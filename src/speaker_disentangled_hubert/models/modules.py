# Copied and modified from https://github.com/facebookresearch/dino

# Copyright (c) Facebook, Inc. and its affiliates.
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

import torch
import torch.nn.functional as F
from torch import nn


def init_module(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 256,
        hidden_dim: int = 2048,
        norm_outputs: bool = False,
    ):
        super().__init__()
        self.norm_outputs = norm_outputs
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.norm_outputs:
            x = nn.functional.normalize(x, dim=-1, p=2)
        return x


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.apply(self._init_weights)

        self.last_layer = torch.nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def _init_weights(self, m):
        init_module(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int = 4096,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_outputs = []

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()

        loss = F.cross_entropy(student_out, teacher_out)
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output) -> None:
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
