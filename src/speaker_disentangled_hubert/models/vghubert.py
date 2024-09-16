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

import pickle
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from ...vghubert.models import audio_encoder
from ..mincut.mincut_utils import min_cut
from .modules import init_module


def load_vghubert(model_dir="models/vg-hubert_3"):
    args_path = Path(model_dir) / "args.pkl"
    bundle_path = Path(model_dir) / "best_bundle.pth"

    with open(args_path, "rb") as f:
        model_args = pickle.load(f)

    model = audio_encoder.AudioEncoder(model_args)
    model.carefully_load_state_dict(torch.load(bundle_path)["dual_encoder"], load_all=True)
    model.eval()
    return model


class VGHubertForSyllableDiscovery(nn.Module):
    def __init__(
        self,
        checkpoint_path="models/vg-hubert_3",
        quantizer1_path="models/vg-hubert_3/quantizer1.joblib",
        quantizer2_path="models/vg-hubert_3/quantizer2.npy",
        segmentation_layer: int = 8,
    ):
        super().__init__()
        self.segmentation_layer = segmentation_layer
        self.model = load_vghubert(checkpoint_path)

        self.quantizer1 = joblib.load(quantizer1_path) if quantizer1_path else None
        self.quantizer2 = np.load(quantizer2_path) if quantizer2_path else None

    @torch.inference_mode()
    def get_hidden_states(self, input_values: torch.Tensor) -> np.ndarray:
        hidden_states = self.model(
            input_values,
            padding_mask=None,
            mask=False,
            tgt_layer=self.segmentation_layer,
            need_attention_weights=False,
            pre_feats=False,
        )["features"]
        return hidden_states.squeeze(0).cpu().numpy()

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


class VGHubertForSequenceClassification(nn.Module):
    def __init__(
        self,
        model_name_or_path="models/vg-hubert_3",
        classifier_proj_size: int = 256,
        num_labels: int = 1251,
        segmentation_layer: int = 8,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.hubert = load_vghubert(model_name_or_path)
        self.projector = nn.Linear(hidden_size, classifier_proj_size)
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
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        padding_mask = attention_mask.bool().logical_not()
        hidden_states = self.hubert(
            input_values,
            padding_mask=padding_mask,
            mask=False,
            tgt_layer=self.segmentation_layer,
            need_attention_weights=False,
            pre_feats=False,
        )["features"]

        hidden_states = self.projector(hidden_states)

        # attention mask
        extra = padding_mask.size(1) % hidden_states.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), hidden_states.size(1), -1)
        padding_mask = padding_mask.all(-1)

        hidden_states[padding_mask] = 0.0
        pooled_output = hidden_states.sum(dim=1) / padding_mask.logical_not().sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
