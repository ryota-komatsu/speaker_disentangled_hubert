# copied and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/fastspeech2_conformer/modeling_fastspeech2_conformer.py

# coding=utf-8
# Copyright 2023 The Espnet authors, IMS Toucan authors, and the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional

import torch
from torch import nn

from ..configuration_sylreg_decoder import FlowMatchingConfig


class MLP(nn.Module):
    """
    Multi-layered conv1d with a GLU activation function for Transformer block.
    https://arxiv.org/abs/1905.09263
    """

    def __init__(self, config, kernel_size: int = 3):
        super().__init__()
        self.gate_proj = nn.Conv1d(
            config.hidden_size, config.intermediate_size, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.up_proj = nn.Conv1d(
            config.hidden_size, config.intermediate_size, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.down_proj = nn.Conv1d(
            config.intermediate_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: Optional[torch.BoolTensor] = None):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Batch of input tensors.

        Returns:
            `torch.Tensor`: Batch of output tensors `(batch_size, sequence_length, hidden_size)`.
        """
        hidden_states = hidden_states.transpose(-1, 1)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            hidden_states = hidden_states.masked_fill(~attention_mask, 0.0)

        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)

        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask, 0.0)

        hidden_states = self.down_proj(hidden_states)
        hidden_states = hidden_states.transpose(-1, 1)
        return hidden_states


class FlowMatchingDurationPredictor(nn.Module):
    """
    Duration predictor module.
    https://arxiv.org/abs/1905.09263
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.log_domain_offset = 1.0
        self.conv = nn.Conv1d(config.embedding_dim, 1, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.FloatTensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.

        Returns:
            `torch.Tensor`: Batch of predicted durations in log domain `(batch_size, max_text_length)`.

        """
        # (batch_size, input_dim, max_text_length)
        hidden_states = hidden_states.transpose(1, -1)

        # NOTE: calculate in log domain, (batch_size, max_text_length)
        hidden_states = self.conv(hidden_states).squeeze(1)

        if not self.training:
            # NOTE: calculate in linear domain
            hidden_states = torch.clamp(torch.round(hidden_states.exp() - self.log_domain_offset), min=0).long()

        return hidden_states
