# from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.scale = hidden_size**0.5
        self.eps = eps
        self.to_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.to_weight.weight)

    def forward(self, hidden_states: torch.FloatTensor, time_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                hidden states.
            time_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
                condition for adaptive norm layers.
        """
        time_embeddings = time_embeddings.unsqueeze(1)

        normed = F.normalize(hidden_states, dim=-1, eps=self.eps)
        gamma = self.to_weight(time_embeddings)
        return normed * self.scale * (gamma + 1.0)
