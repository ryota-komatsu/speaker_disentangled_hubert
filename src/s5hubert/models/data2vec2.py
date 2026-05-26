# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
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

import math
from functools import partial
from types import SimpleNamespace
from typing import Callable

import torch
from fairseq.models.wav2vec import ConvFeatureExtractionModel
from fairseq.modules import LayerNorm, SamePad, TransposeLast
from timm.models.vision_transformer import Mlp
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


class Data2Vec2Config(PretrainedConfig):
    def __init__(
        self,
        _name="data2vec_multi",
        depth=7,  # 8
        num_heads=12,
        norm_eps=1e-05,
        norm_affine=True,
        encoder_dropout=0.0,  # 0.1
        post_mlp_drop=0.0,  # 0.1
        attention_dropout=0.0,  # 0.1
        activation_dropout=0.0,
        dropout_input=0.0,
        embed_dim=768,
        mlp_ratio=4.0,
        modalities={
            "audio": {
                "prenet_depth": 4,
                "prenet_dropout": 0.0,  # 0.1
                "use_alibi_encoder": True,
                "learned_alibi_scale": False,
                "num_alibi_heads": 12,
                "extractor_mode": "layer_norm",
                "feature_encoder_spec": "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
                "conv_pos_width": 95,
                "conv_pos_groups": 16,
                "conv_pos_depth": 5,
            }
        },
        **kwargs,
    ):
        self._name = _name
        self.depth = depth
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.norm_affine = norm_affine
        self.encoder_dropout = encoder_dropout
        self.post_mlp_drop = post_mlp_drop
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout_input = dropout_input
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.modalities = modalities
        super().__init__(**kwargs)


class BlockEncoder(nn.Module):
    def __init__(self, blocks, norm_layer, dropout):
        super().__init__()
        self.blocks = blocks
        self.norm = norm_layer
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, padding_mask, alibi_bias, alibi_scale):
        x = self.norm(x)

        x = self.dropout(x)

        for i, blk in enumerate(self.blocks):
            ab = alibi_bias
            if ab is not None and alibi_scale is not None:
                scale = alibi_scale[i] if alibi_scale.size(0) > 1 else alibi_scale.squeeze(0)
                ab = ab * scale.type_as(ab)
            x = blk(x, padding_mask, ab)

        return x


class Data2Vec2EncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        post_mlp_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Data2Vec2Attention(dim, num_heads, attn_drop, drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=mlp_drop)
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        x = self.norm1(x + self.attn(x, padding_mask, alibi_bias))
        x = self.norm2(x + self.post_mlp_dropout(self.mlp(x)))
        return x


class Data2Vec2Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        dtype = q.dtype

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias

        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_alibi_bias(batch_size, time_steps, heads, dtype, device):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(heads))

    # prepare alibi position linear bias. Note that wav2vec2 is non
    # autoregressive model so we want a symmetric mask with 0 on the
    # diagonal and other wise linear decreasing valuees
    pos_bias = torch.abs(torch.arange(time_steps).unsqueeze(0) - torch.arange(time_steps).unsqueeze(1)) * -1

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(heads, -1, -1)
    alibi_bias = alibi_bias.to(dtype=dtype, device=device).repeat(batch_size, 1, 1)
    alibi_bias = alibi_bias.view(batch_size, heads, time_steps, time_steps)
    return alibi_bias


class AudioEncoder(nn.Module):
    def __init__(
        self,
        modality_cfg,
        embed_dim: int,
        make_block: Callable[[], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
    ):
        super().__init__()
        self.modality_cfg = modality_cfg
        self.feature_enc_layers = eval(modality_cfg.feature_encoder_spec)
        feature_embed_dim = self.feature_enc_layers[-1][0]

        self.local_encoder = ConvFeatureExtractionModel(
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=modality_cfg.extractor_mode,
            conv_bias=False,
        )

        self.project_features = nn.Sequential(
            TransposeLast(),
            nn.LayerNorm(feature_embed_dim),
            nn.Linear(feature_embed_dim, embed_dim),
        )

        num_pos_layers = modality_cfg.conv_pos_depth
        k = max(3, modality_cfg.conv_pos_width // num_pos_layers)

        self.relative_positional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=modality_cfg.conv_pos_groups,
                    ),
                    SamePad(k),
                    TransposeLast(),
                    LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(num_pos_layers)
            ],
            TransposeLast(),
        )

        self.context_encoder = BlockEncoder(
            nn.ModuleList(make_block() for _ in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim),
            modality_cfg.prenet_dropout,
        )

        self.get_alibi_bias = get_alibi_bias if modality_cfg.use_alibi_encoder else None

        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            self.alibi_scale = nn.Parameter(
                torch.ones(1, 1, self.modality_cfg.num_alibi_heads, 1, 1, dtype=torch.float),
                requires_grad=modality_cfg.learned_alibi_scale,
            )

    def local_features(self, features):
        x = self.local_encoder(features)
        x = self.project_features(x)
        return x

    def convert_padding_mask(self, x, padding_mask):
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length - kernel_size) / stride + 1)

            for i in range(len(self.feature_enc_layers)):
                input_lengths = _conv_out_length(
                    input_lengths,
                    self.feature_enc_layers[i][1],
                    self.feature_enc_layers[i][2],
                )

            return input_lengths.to(torch.long)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
            else:
                padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        return padding_mask


class Data2VecMultiModel(PreTrainedModel):
    config_class = Data2Vec2Config
    base_model_prefix = "model"

    def make_modality_encoder(
        self,
        cfg,
        embed_dim: int,
        make_block: Callable[[], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
    ) -> AudioEncoder:
        return AudioEncoder(cfg, embed_dim, make_block, norm_layer)

    def __init__(self, config: Data2Vec2Config):
        super().__init__(config)
        self.config = config

        make_layer_norm = partial(nn.LayerNorm, eps=config.norm_eps, elementwise_affine=config.norm_affine)

        def make_block():
            return Data2Vec2EncoderLayer(
                config.embed_dim,
                config.num_heads,
                config.mlp_ratio,
                drop=config.encoder_dropout,
                attn_drop=config.attention_dropout,
                mlp_drop=config.activation_dropout,
                post_mlp_drop=config.post_mlp_drop,
                norm_layer=make_layer_norm,
            )

        self.modality_encoders = nn.ModuleDict()
        mod_cfg = SimpleNamespace(**config.modalities["audio"])
        self.modality_encoders["AUDIO"] = self.make_modality_encoder(
            mod_cfg,
            config.embed_dim,
            make_block,
            make_layer_norm,
        )

        self.blocks = nn.ModuleList([make_block() for _ in range(config.depth)])

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_layer: int | None = None,
    ):
        """
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Raw speech waveform.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                1: non-padding
                0: padding
        """
        padding_mask = attention_mask.bool().logical_not() if attention_mask is not None else None

        feature_extractor = self.modality_encoders["AUDIO"]

        x = feature_extractor.local_features(input_values)

        if padding_mask is not None:
            padding_mask = feature_extractor.convert_padding_mask(x, padding_mask)

        orig_B, orig_T, _ = x.shape

        x = x + feature_extractor.relative_positional_encoder(x)

        alibi_bias = None
        alibi_scale = feature_extractor.alibi_scale

        if feature_extractor.get_alibi_bias is not None:
            alibi_bias = feature_extractor.get_alibi_bias(
                batch_size=orig_B,
                time_steps=orig_T,
                heads=feature_extractor.modality_cfg.num_alibi_heads,
                dtype=torch.float32,
                device=x.device,
            )

            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None

        x = feature_extractor.context_encoder(
            x,
            padding_mask,
            alibi_bias,
            alibi_scale[: feature_extractor.modality_cfg.prenet_depth] if alibi_scale is not None else None,
        )

        alibi_scale = (
            alibi_scale[feature_extractor.modality_cfg.prenet_depth :]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale
        )

        layer_results = []
        for i, blk in enumerate(self.blocks):
            ab = alibi_bias
            if ab is not None and alibi_scale is not None:
                scale = alibi_scale[i] if alibi_scale.size(0) > 1 else alibi_scale.squeeze(0)
                ab = ab * scale.type_as(ab)

            x = blk(x, padding_mask=padding_mask, alibi_bias=ab)
            layer_results.append(x)
            if output_layer is not None and i == len(self.blocks) + output_layer:
                break

        return layer_results, padding_mask.logical_not() if padding_mask is not None else None

    def freeze_pretrained_modules(self):
        # CNN
        self.modality_encoders["AUDIO"].requires_grad_(False)

        # Transformer
        self.modality_encoders["AUDIO"].context_encoder.requires_grad_(True)
        self.blocks.requires_grad_(True)

        self.modality_encoders["AUDIO"].alibi_scale.requires_grad_(False)
