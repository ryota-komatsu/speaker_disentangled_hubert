# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from transformers import PreTrainedModel
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniBigVGANConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import TorchActivation1d

from . import activations
from .utils import get_padding, init_weights


class AMPBlock1(nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        config: Qwen2_5OmniBigVGANConfig,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
    ):
        super().__init__()

        self.config = config

        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)  # Total number of conv layers

        # Activation functions
        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=activations.SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def apply_weight_norm(self):
        for l in self.convs1:
            weight_norm(l)
        for l in self.convs2:
            weight_norm(l)

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_parametrizations(l, "weight")
        for l in self.convs2:
            remove_parametrizations(l, "weight")


class BigVGan(PreTrainedModel):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
        h (AttrDict): Hyperparameters.

    Note:
        - Ensure that the activation function is correctly specified in the hyperparameters (config.activation).
    """

    config: Qwen2_5OmniBigVGANConfig
    base_model_prefix = "vocoder"

    def __init__(self, config: Qwen2_5OmniBigVGANConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # Pre-conv
        self.conv_pre = Conv1d(config.mel_dim, config.upsample_initial_channel, 7, 1, padding=3)

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        ConvTranspose1d(
                            config.upsample_initial_channel // (2**i),
                            config.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(config, ch, k, d))

        # Post-conv
        self.activation_post = TorchActivation1d(activation=activations.SnakeBeta(ch))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        self.post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (`torch.FloatTensor` of shape `(batch_size, sequence_length, model_in_dim)`):
                Spectrograms.
        Returns:
            x (`torch.FloatTensor` of shape `(batch_size, (sequence_length - 1) * 320 + 400)`):
                Waveforms.
        """
        x = x.transpose(2, 1)  # make channel first for Conv1d

        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x.squeeze(1)

    def apply_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    weight_norm(l_i)
            for l in self.resblocks:
                l.apply_weight_norm()
            weight_norm(self.conv_pre)
            weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already applied weight norm. Skipping!")
            pass

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_parametrizations(l_i, "weight")
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_parametrizations(self.conv_pre, "weight")
            remove_parametrizations(self.conv_post, "weight")
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass
