# Copied and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/hubert/modeling_hubert.py

# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.hubert.modeling_hubert import HubertEncoderLayer, HubertModel

from ..mincut.mincut_utils import min_cut
from .modules import MLP, init_module


class BYOL(nn.Module):
    def __init__(
        self,
        model_name_or_path="facebook/hubert-base-ls960",
        init_last_layer: int = 3,
        head_out_size: int = 256,
        head_hidden_size: int = 2048,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.init_last_layer = init_last_layer

        self.student = HubertModel.from_pretrained(model_name_or_path)
        self.student_projector = MLP(
            self.student.config.hidden_size,
            head_out_size,
            head_hidden_size,
        )
        self.student_predictor = MLP(
            head_out_size,
            head_out_size,
            head_hidden_size,
            norm_outputs=True,
        )
        self.loss_fn = nn.MSELoss()

        self.reset_parameters(init_last_layer)
        self.make_teacher(head_out_size, head_hidden_size)

    def reset_parameters(self, init_last_layer: int = 3):
        for m in self.student.encoder.layers[-init_last_layer:].modules():
            init_module(m)

        for m in self.student_projector.modules():
            init_module(m)

        for m in self.student_predictor.modules():
            init_module(m)

    def make_teacher(
        self,
        head_out_size: int = 256,
        head_hidden_size: int = 2048,
    ):
        self.teacher_encoder_layers = nn.ModuleList(
            [HubertEncoderLayer(self.student.config) for _ in range(self.student.config.num_hidden_layers)]
        )
        self.teacher_encoder_layers.load_state_dict(self.student.encoder.layers.state_dict())
        self.teacher_encoder_layers.requires_grad_(False)

        self.teacher_projector = MLP(
            self.student.config.hidden_size,
            head_out_size,
            head_hidden_size,
            norm_outputs=True,
        )
        self.teacher_projector.load_state_dict(self.student_projector.state_dict())
        self.teacher_projector.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self):
        for param_s, param_t in zip(self.student.encoder.layers.parameters(), self.teacher_encoder_layers.parameters()):
            param_t.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param_s.detach().data)

        for param_s, param_t in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            param_t.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param_s.detach().data)

    def forward(
        self,
        teacher_input_values: torch.Tensor,
        student_input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # enable dropout
        self.student.feature_projection.train()
        self.student.encoder.layers.train()
        student_hidden_states, padding_mask = self.student_forward(student_input_values, attention_mask)
        student_projection = self.student_projector(student_hidden_states[-1][padding_mask])
        student_prediction = self.student_predictor(student_projection)

        # disable dropout
        self.student.feature_projection.eval()
        self.teacher_encoder_layers.eval()
        with torch.no_grad():
            teacher_hidden_states, padding_mask = self.teacher_forward(teacher_input_values, attention_mask)
            teacher_projection = self.teacher_projector(teacher_hidden_states[-1][padding_mask])

        return self.loss_fn(student_prediction, teacher_projection)

    def student_forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]:
        extract_features = self.student.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        feature_vector_attention_mask = None
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.student._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
            feature_vector_attention_mask = attention_mask

        hidden_states = self.student.feature_projection(extract_features)

        all_hidden_states = ()

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.student.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.student.encoder.layer_norm(hidden_states)
        hidden_states = self.student.encoder.dropout(hidden_states)

        for layer in self.student.encoder.layers:
            all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.student.config.layerdrop) else False
            if not skip_the_layer:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

        all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states, feature_vector_attention_mask

    def teacher_forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]:
        extract_features = self.student.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        feature_vector_attention_mask = None
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.student._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
            feature_vector_attention_mask = attention_mask

        hidden_states = self.student.feature_projection(extract_features)

        all_hidden_states = ()

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.student.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.student.encoder.layer_norm(hidden_states)
        # hidden_states = self.student.encoder.dropout(hidden_states)

        for layer in self.teacher_encoder_layers:
            all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

        all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states, feature_vector_attention_mask

    def freeze_pretrained_modules(self):
        """for warmup"""
        # CNN
        self.student.feature_extractor._freeze_parameters()
        self.student.feature_projection.requires_grad_(False)

        # Transformer
        self.student.encoder.pos_conv_embed.requires_grad_(False)
        self.student.encoder.layer_norm.requires_grad_(False)
        self.student.encoder.layers.requires_grad_(False)
        self.student.encoder.layers[-self.init_last_layer :].requires_grad_(True)

    def defrost_transformer_encoder(self):
        # CNN
        self.student.feature_extractor._freeze_parameters()
        self.student.feature_projection.requires_grad_(False)

        # Transformer
        self.student.encoder.pos_conv_embed.requires_grad_(False)
        self.student.encoder.layer_norm.requires_grad_(False)
        self.student.encoder.layers.requires_grad_(True)


class BYOLForSyllableDiscovery(nn.Module):
    def __init__(
        self,
        checkpoint_path="models/byol/checkpoint",
        quantizer1_path="models/byol/quantizer1.joblib",
        quantizer2_path="models/byol/quantizer2.npy",
        segmentation_layer: int = 8,
    ):
        super().__init__()
        self.segmentation_layer = segmentation_layer

        state_dict = torch.load(checkpoint_path)["model"]
        head_out_size, head_hidden_size = state_dict["student_projector.mlp.3.weight"].shape

        self.model = BYOL(head_out_size=head_out_size, head_hidden_size=head_hidden_size)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.quantizer1 = joblib.load(quantizer1_path) if quantizer1_path else None
        self.quantizer2 = np.load(quantizer2_path) if quantizer2_path else None

    @classmethod
    def from_hf_hub(cls) -> "BYOLForSyllableDiscovery":
        checkpoint_path = hf_hub_download("ryota-komatsu/speaker_disentangled_hubert", "models/byol/checkpoint")
        quantizer1_path = hf_hub_download("ryota-komatsu/speaker_disentangled_hubert", "models/byol/quantizer1.joblib")
        quantizer2_path = hf_hub_download("ryota-komatsu/speaker_disentangled_hubert", "models/byol/quantizer2.npy")
        return cls(
            checkpoint_path,
            quantizer1_path,
            quantizer2_path,
            8,
        )

    @torch.inference_mode()
    def get_hidden_states(self, input_values: torch.Tensor) -> np.ndarray:
        hidden_states, _ = self.model.student_forward(input_values)
        return hidden_states[self.segmentation_layer].squeeze(0).cpu().numpy()

    def forward(self, input_values: torch.Tensor) -> Dict[str, np.ndarray]:
        hidden_states = self.get_hidden_states(input_values)
        if self.quantizer1 is None or self.quantizer2 is None:
            return {"hidden_states": hidden_states}

        frame_similarity = hidden_states @ hidden_states.T
        boundary, segment_features, frame_boundary = min_cut(hidden_states)

        # deduplicated syllabic units
        units = self.quantizer1.predict(segment_features)
        units = self.quantizer2[units]

        # duplicated syllabic units
        repeats = frame_boundary[:, 1] - frame_boundary[:, 0]
        duplicated_units = np.repeat(units, repeats)
        return {
            "units": units,
            "duplicated_units": duplicated_units,
            "boundary": boundary,
            "frame_boundary": frame_boundary,
            "hidden_states": hidden_states,
            "frame_similarity": frame_similarity,
        }


class BYOLForSequenceClassification(nn.Module):
    def __init__(
        self,
        model_name_or_path="models/byol/checkpoint",
        classifier_proj_size: int = 256,
        num_labels: int = 1251,
        segmentation_layer: int = 8,
    ):
        super().__init__()
        state_dict = torch.load(model_name_or_path)["model"]
        student_state_dict = OrderedDict()
        for name in state_dict:
            if name.startswith("student."):
                student_state_dict[name[8:]] = state_dict[name]

        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert.load_state_dict(student_state_dict)
        self.projector = nn.Linear(self.hubert.config.hidden_size, classifier_proj_size)
        self.classifier = nn.Linear(classifier_proj_size, num_labels, bias=False)

        # Initialize weights and apply final processing
        self.reset_parameters()
        self.num_labels = num_labels
        self.segmentation_layer = segmentation_layer

        self.freeze_base_model()
        self.hubert.eval()

    def reset_parameters(self):
        init_module(self.projector)
        init_module(self.classifier)

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        self.hubert.requires_grad_(False)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[1][self.segmentation_layer]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self.hubert._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
