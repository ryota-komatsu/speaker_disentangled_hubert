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

from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
from torch import nn
from transformers.models.hubert.modeling_hubert import HubertEncoderLayer, HubertModel

from ..mincut.mincut_utils import min_cut
from .modules import DINOHead, DINOLoss, init_module


class DINO(nn.Module):
    def __init__(
        self,
        model_name_or_path="facebook/hubert-base-ls960",
        init_last_layer: int = 3,
        head_out_size: int = 4096,
        head_hidden_size: int = 2048,
        head_bottleneck_size: int = 256,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.init_last_layer = init_last_layer

        self.student = HubertModel.from_pretrained(model_name_or_path)
        self.student_head = DINOHead(
            self.student.config.hidden_size,
            head_out_size,
            head_hidden_size,
            head_bottleneck_size,
        )
        self.loss_fn = DINOLoss(head_out_size, teacher_temp, student_temp, center_momentum)

        self.reset_parameters(init_last_layer)
        self.make_teacher(head_out_size, head_hidden_size, head_bottleneck_size)

    def reset_parameters(self, init_last_layer: int = 3):
        for m in self.student.encoder.layers[-init_last_layer:].modules():
            init_module(m)

    def make_teacher(
        self,
        head_out_size: int = 4096,
        head_hidden_size: int = 2048,
        head_bottleneck_size: int = 256,
    ):
        self.teacher_encoder_layers = nn.ModuleList(
            [HubertEncoderLayer(self.student.config) for _ in range(self.student.config.num_hidden_layers)]
        )
        self.teacher_encoder_layers.load_state_dict(self.student.encoder.layers.state_dict())
        self.teacher_encoder_layers.requires_grad_(False)

        self.teacher_head = DINOHead(
            self.student.config.hidden_size,
            head_out_size,
            head_hidden_size,
            head_bottleneck_size,
        )
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.teacher_head.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self):
        for param_s, param_t in zip(self.student.encoder.layers.parameters(), self.teacher_encoder_layers.parameters()):
            param_t.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param_s.detach().data)

        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
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
        student_logits = self.student_head(student_hidden_states[-1][padding_mask])

        # disable dropout
        self.student.feature_projection.eval()
        self.teacher_encoder_layers.eval()
        with torch.no_grad():
            teacher_hidden_states, padding_mask = self.teacher_forward(teacher_input_values, attention_mask)
            teacher_logits = self.teacher_head(teacher_hidden_states[-1][padding_mask])

        return self.loss_fn(student_logits, teacher_logits)

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


class DINOForSyllableDiscovery(nn.Module):
    def __init__(
        self,
        checkpoint_path="models/dino/checkpoint",
        quantizer1_path="models/dino/quantizer1.joblib",
        quantizer2_path="models/dino/quantizer2.npy",
        segmentation_layer: int = 8,
    ):
        super().__init__()
        self.segmentation_layer = segmentation_layer

        state_dict = torch.load(checkpoint_path)["model"]
        head_hidden_size, _ = state_dict["student_head.mlp.0.weight"].shape
        head_out_size, head_bottleneck_size = state_dict[
            "student_head.last_layer.parametrizations.weight.original1"
        ].shape

        self.model = DINO(
            head_out_size=head_out_size,
            head_hidden_size=head_hidden_size,
            head_bottleneck_size=head_bottleneck_size,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.quantizer1 = joblib.load(quantizer1_path) if quantizer1_path else None
        self.quantizer2 = np.load(quantizer2_path) if quantizer2_path else None

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
