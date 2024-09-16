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

import sys
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from .modules import init_module

sys.path.append("src/sdhubert")
from ...sdhubert.utils.misc import load_model


class SDHubertForSequenceClassification(nn.Module):
    def __init__(
        self,
        model_name_or_path="models/sdhubert_base/sdhubert_base.pt",
        classifier_proj_size: int = 256,
        num_labels: int = 1251,
        segmentation_layer: int = 9,
    ):
        super().__init__()
        self.hubert = load_model(model_name_or_path)[0]
        self.projector = nn.Linear(self.hubert.speech_model.config.hidden_size, classifier_proj_size)
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

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(
            self.hubert.speech_model.config.conv_kernel, self.hubert.speech_model.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        wavlen = self._get_feat_extract_output_lengths(torch.sum(attention_mask, dim=1)).tolist()
        hidden_states = self.hubert(wav=input_values, wavlen=wavlen, inference_mode=True)["hidden_states"]
        hidden_states = hidden_states[self.segmentation_layer][:, 1:]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self.hubert.speech_model._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
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
        )
