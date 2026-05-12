# Copied and modified from https://github.com/lucidrains/voicebox-pytorch/blob/main/voicebox_pytorch/voicebox_pytorch.py

# MIT License
#
# Copyright (c) 2023 Phil Wang
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

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.models.fastspeech2_conformer.modeling_fastspeech2_conformer import length_regulator
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import SinusPositionEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb
from transformers.utils import ModelOutput

from ..bigvgan.bigvgan import BigVGan, BigVGanConfig
from ..bigvgan.data import dynamic_range_compression_torch
from .configs import FlowMatchingConfig, FlowMatchingWithBigVGanConfig
from .modules.fastspeech import MLP, FlowMatchingDurationPredictor
from .modules.norm import AdaptiveRMSNorm


class Attention(nn.Module):
    """
    https://arxiv.org/abs/2302.05442
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.BoolTensor] = None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        bsz, heads, q_len, _ = query_states.shape

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if attention_mask is not None and attention_mask.ndim != 4:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if attention_mask is not None:
            attention_mask = attention_mask.expand(-1, heads, q_len, -1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class AttentionV2(nn.Module):
    """
    https://arxiv.org/abs/2302.05442
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.BoolTensor] = None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        bsz, heads, q_len, _ = query_states.shape

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if attention_mask is not None and attention_mask.ndim != 4:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if attention_mask is not None:
            attention_mask = attention_mask.expand(-1, heads, q_len, -1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class TokenEncoderAttention(AttentionV2):
    """
    https://arxiv.org/abs/2302.05442
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__(config)
        self.head_dim = getattr(config, "head_dim", config.embedding_dim // config.num_attention_heads)
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.k_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.v_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.o_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)


class TokenEncoderMLP(MLP):
    """
    Multi-layered conv1d with a GLU activation function for Transformer block.
    https://arxiv.org/abs/1905.09263
    """

    def __init__(self, config: FlowMatchingConfig, kernel_size: int = 3):
        super().__init__(config, kernel_size)
        self.gate_proj = nn.Conv1d(
            config.embedding_dim, config.embedding_dim * 2, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.up_proj = nn.Conv1d(
            config.embedding_dim, config.embedding_dim * 2, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.down_proj = nn.Conv1d(
            config.embedding_dim * 2, config.embedding_dim, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.act_fn = nn.SiLU()


class TokenEncoderLayer(nn.Module):
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.self_attn = TokenEncoderAttention(config)
        self.mlp = TokenEncoderMLP(config)
        self.input_layernorm = nn.RMSNorm(config.embedding_dim, elementwise_affine=False)
        self.post_attention_layernorm = nn.RMSNorm(config.embedding_dim, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        attn_input = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(attn_input, position_embeddings, attention_mask) + hidden_states

        mlp_input = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(mlp_input, attention_mask) + hidden_states
        return hidden_states


class AdaLNZero(nn.Module):
    """
    https://arxiv.org/abs/2212.09748
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size * 6, bias=False)
        nn.init.zeros_(self.linear.weight)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, hidden_states: torch.FloatTensor, time_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                hidden states.
            time_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
                condition for adaptive norm layers.
        """
        time_embeddings = time_embeddings.unsqueeze(1)
        time_embeddings = self.linear(time_embeddings)
        attn_scale, attn_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = torch.chunk(time_embeddings, 6, dim=2)
        return self.norm(hidden_states) * (attn_scale + 1.0) + attn_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate


class AdaLNZeroFinal(nn.Module):
    """
    https://arxiv.org/abs/2212.09748
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        nn.init.zeros_(self.linear.weight)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, hidden_states: torch.FloatTensor, time_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                hidden states.
            time_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
                condition for adaptive norm layers.
        """
        time_embeddings = time_embeddings.unsqueeze(1)
        time_embeddings = self.linear(time_embeddings)
        scale, shift = torch.chunk(time_embeddings, 2, dim=2)
        return self.norm(hidden_states) * (scale + 1.0) + shift


class DiTLayer(nn.Module):
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = AdaptiveRMSNorm(config.hidden_size)
        self.post_attention_layernorm = AdaptiveRMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        time_embeddings: torch.FloatTensor,
    ):
        attn_input = self.input_layernorm(hidden_states, time_embeddings)
        hidden_states = self.self_attn(attn_input, position_embeddings, attention_mask) + hidden_states

        mlp_input = self.post_attention_layernorm(hidden_states, time_embeddings)
        hidden_states = self.mlp(mlp_input, attention_mask) + hidden_states
        return hidden_states


class DiTLayerV2(nn.Module):
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.self_attn = AttentionV2(config)
        self.mlp = MLP(config)
        self.input_layernorm = AdaLNZero(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        time_embeddings: torch.FloatTensor,
    ):
        attn_input, attn_gate, mlp_scale, mlp_shift, mlp_gate = self.input_layernorm(hidden_states, time_embeddings)
        hidden_states = self.self_attn(attn_input, position_embeddings, attention_mask) * attn_gate + hidden_states

        mlp_input = self.post_attention_layernorm(hidden_states) * (mlp_scale + 1.0) + mlp_shift
        hidden_states = self.mlp(mlp_input, attention_mask) * mlp_gate + hidden_states
        return hidden_states


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_size)
        self.mlp = nn.Sequential(nn.Linear(freq_embed_size, hidden_size), nn.SiLU())

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                diffusion timesteps.
        Returns:
            embeddings (`torch.Tensor` of shape `(batch_size, hidden_size)`):
                condition for adaptive norm layers.
        """
        embeddings = self.time_embed(timesteps)
        embeddings = embeddings.to(timesteps.dtype)
        embeddings = self.mlp(embeddings)
        return embeddings


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: FlowMatchingConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        rope_init_fn = self.compute_default_rope_parameters

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: FlowMatchingConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.embedding_dim // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FlowMatchingModel(PreTrainedModel):
    config_class = FlowMatchingConfig

    def __init__(self, config: FlowMatchingConfig):
        super().__init__(config)
        self.time_cond_mlp = TimestepEmbedding(config.hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.embedding_dim, padding_idx=config.vocab_size)
        self.to_embed = nn.Linear(config.num_mel_bins + config.embedding_dim, config.hidden_size)

        self.layers = nn.ModuleList([DiTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        self.to_pred = nn.Linear(config.hidden_size, config.num_mel_bins, bias=False)
        self.duration_predictor = FlowMatchingDurationPredictor(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        spectrogram_labels: torch.FloatTensor,
        duration_labels: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Batch of padded durations.
        """
        mask = (spectrogram_labels != -100).any(dim=-1)
        bsz, seq_len, _ = spectrogram_labels.shape
        spectrogram_labels = (spectrogram_labels - self.config.mean) / self.config.std

        # main conditional flow logic is below
        x0 = torch.randn_like(spectrogram_labels)
        timesteps = torch.rand((bsz,), device=self.device)
        t = timesteps.unsqueeze(1).unsqueeze(2)
        xt = (1 - t) * x0 + t * spectrogram_labels
        xt = xt.masked_fill(~mask.unsqueeze(2), 0)
        ut = spectrogram_labels - x0

        # phoneme or semantic conditioning embedding
        inputs_embeds = self.embed_tokens(input_ids)

        # forward duration predictor
        duration_predictions = self.duration_predictor(inputs_embeds)

        # use groundtruth in training
        inputs_embeds = length_regulator(inputs_embeds, duration_labels)

        attention_mask = input_ids.ne(self.config.vocab_size)
        duration_predictions = duration_predictions.masked_select(attention_mask)
        duration_labels_ = duration_labels.masked_select(attention_mask)
        duration_labels_ = torch.log(duration_labels_.float() + self.duration_predictor.log_domain_offset)
        duration_loss = F.mse_loss(duration_predictions, duration_labels_)

        time_embeddings = self.time_cond_mlp(timesteps)

        # drop condition for classifier-free guidance
        dropout_mask = torch.rand(bsz, 1, 1, device=inputs_embeds.device) < self.config.cfg_dropout
        dropout_mask = dropout_mask.expand_as(inputs_embeds)
        inputs_embeds.masked_fill_(dropout_mask, 0.0)

        hidden_states = torch.cat([xt, inputs_embeds], dim=-1)
        hidden_states = self.to_embed(hidden_states)

        # rotary embeddings
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # going through the attention layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, position_embeddings, time_embeddings)

        hidden_states = self.norm(hidden_states)
        vt = self.to_pred(hidden_states)

        loss = F.mse_loss(vt[mask], ut[mask]) + duration_loss
        return ModelOutput(loss=loss)

    @torch.inference_mode()
    def sample(self, input_ids: torch.LongTensor) -> ModelOutput:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.

        Returns:
            x1 (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_mel_bins)`):
                Synthesized log mel-spectrograms.
        """
        mask = input_ids.ne(self.config.vocab_size)

        inputs_embeds = self.embed_tokens(input_ids)

        # forward duration predictor
        duration_predictions = self.duration_predictor(inputs_embeds)
        duration_predictions = duration_predictions.masked_fill(~mask, 0.0)

        inputs_embeds = length_regulator(inputs_embeds, duration_predictions)

        # update mask
        lengths = duration_predictions.sum(dim=1, keepdim=True)  # (bsz, 1)
        mask = torch.arange(0, lengths.max(), device=lengths.device).unsqueeze(0) < lengths

        bsz, seq_len, _ = inputs_embeds.shape
        xt = torch.randn(bsz, seq_len, self.config.num_mel_bins, device=inputs_embeds.device)
        expand_mask = torch.cat([mask, mask])

        # rotary embeddings
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        for t in torch.arange(0, 1, self.config.dt, device=self.device):
            time_embeddings = self.time_cond_mlp(t.unsqueeze(0).expand(2 * bsz))

            # concat source signal, semantic / phoneme conditioning embed, and conditioning
            # and project
            hidden_states_cond = torch.cat([xt, inputs_embeds], dim=-1)
            hidden_states_uncond = torch.cat([xt, torch.zeros_like(inputs_embeds)], dim=-1)
            hidden_states = torch.cat([hidden_states_cond, hidden_states_uncond])
            hidden_states = self.to_embed(hidden_states)

            # going through the attention layers
            for layer in self.layers:
                hidden_states = layer(hidden_states, expand_mask, position_embeddings, time_embeddings)

            hidden_states = self.norm(hidden_states)

            # classifier free guidance
            vt = self.to_pred(hidden_states)
            vt_cond, vt_uncond = torch.chunk(vt, 2)
            vt = vt_cond + self.config.cfg_strength * (vt_cond - vt_uncond)

            # Euler method
            xt = xt + vt * self.config.dt

        x1 = xt * self.config.std + self.config.mean
        x1[~mask] = dynamic_range_compression_torch(torch.tensor(0))

        return ModelOutput(spectrogram=x1, durations=duration_predictions)


class FlowMatchingModelV2(PreTrainedModel):
    config_class = FlowMatchingConfig

    def __init__(self, config: FlowMatchingConfig):
        super().__init__(config)
        self.time_cond_mlp = TimestepEmbedding(config.hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.embedding_dim, padding_idx=config.vocab_size)
        self.to_embed = nn.Linear(config.num_mel_bins + config.embedding_dim, config.hidden_size)

        self.encoder_layers = nn.ModuleList([TokenEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.encoder_norm = nn.RMSNorm(config.embedding_dim, elementwise_affine=False)
        self.encoder_rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList([DiTLayerV2(config) for _ in range(config.num_hidden_layers)])
        self.norm = AdaLNZeroFinal(config.hidden_size)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        self.to_pred = nn.Linear(config.hidden_size, config.num_mel_bins, bias=False)
        self.duration_predictor = FlowMatchingDurationPredictor(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        spectrogram_labels: torch.FloatTensor,
        duration_labels: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Batch of padded durations.
        """
        mask = (spectrogram_labels != -100).any(dim=-1)
        bsz, seq_len, _ = spectrogram_labels.shape
        spectrogram_labels = (spectrogram_labels - self.config.mean) / self.config.std

        # main conditional flow logic is below
        x0 = torch.randn_like(spectrogram_labels)
        timesteps = torch.rand((bsz,), device=self.device)
        t = timesteps.unsqueeze(1).unsqueeze(2)
        xt = (1 - t) * x0 + t * spectrogram_labels
        xt = xt.masked_fill(~mask.unsqueeze(2), 0)
        ut = spectrogram_labels - x0

        # phoneme or semantic conditioning embedding
        inputs_embeds = self.embed_tokens(input_ids)

        # forward duration predictor
        duration_predictions = self.duration_predictor(inputs_embeds)

        # Encoder: rotary embeddings
        encoder_position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        encoder_position_embeddings = self.encoder_rotary_emb(inputs_embeds, encoder_position_ids)

        # Encoder: going through the attention layers
        encoder_hidden_states = inputs_embeds
        encoder_mask = input_ids.ne(self.config.vocab_size)
        for layer in self.encoder_layers:
            encoder_hidden_states = layer(encoder_hidden_states, encoder_mask, encoder_position_embeddings)

        encoder_hidden_states = self.encoder_norm(encoder_hidden_states)

        # use groundtruth in training
        inputs_embeds = length_regulator(encoder_hidden_states, duration_labels)

        attention_mask = input_ids.ne(self.config.vocab_size)
        duration_predictions = duration_predictions.masked_select(attention_mask)
        duration_labels_ = duration_labels.masked_select(attention_mask)
        duration_labels_ = torch.log(duration_labels_.float() + self.duration_predictor.log_domain_offset)
        duration_loss = F.mse_loss(duration_predictions, duration_labels_)

        time_embeddings = self.time_cond_mlp(timesteps)

        # drop condition for classifier-free guidance
        dropout_mask = torch.rand(bsz, 1, 1, device=inputs_embeds.device) < self.config.cfg_dropout
        dropout_mask = dropout_mask.expand_as(inputs_embeds)
        inputs_embeds.masked_fill_(dropout_mask, 0.0)

        hidden_states = torch.cat([xt, inputs_embeds], dim=-1)
        hidden_states = self.to_embed(hidden_states)

        # rotary embeddings
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # going through the attention layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, position_embeddings, time_embeddings)

        hidden_states = self.norm(hidden_states, time_embeddings)
        vt = self.to_pred(hidden_states)

        loss = F.mse_loss(vt[mask], ut[mask]) + duration_loss
        return ModelOutput(loss=loss)

    @torch.inference_mode()
    def sample(self, input_ids: torch.LongTensor) -> ModelOutput:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.

        Returns:
            x1 (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_mel_bins)`):
                Synthesized log mel-spectrograms.
        """
        mask = input_ids.ne(self.config.vocab_size)

        inputs_embeds = self.embed_tokens(input_ids)

        # forward duration predictor
        duration_predictions = self.duration_predictor(inputs_embeds)
        duration_predictions = duration_predictions.masked_fill(~mask, 0.0)

        # Encoder: rotary embeddings
        encoder_position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        encoder_position_embeddings = self.encoder_rotary_emb(inputs_embeds, encoder_position_ids)

        # Encoder: going through the attention layers
        encoder_hidden_states = inputs_embeds
        encoder_mask = input_ids.ne(self.config.vocab_size)
        for layer in self.encoder_layers:
            encoder_hidden_states = layer(encoder_hidden_states, encoder_mask, encoder_position_embeddings)

        encoder_hidden_states = self.encoder_norm(encoder_hidden_states)

        inputs_embeds = length_regulator(encoder_hidden_states, duration_predictions)

        # update mask
        lengths = duration_predictions.sum(dim=1, keepdim=True)  # (bsz, 1)
        mask = torch.arange(0, lengths.max(), device=lengths.device).unsqueeze(0) < lengths

        bsz, seq_len, _ = inputs_embeds.shape
        xt = torch.randn(bsz, seq_len, self.config.num_mel_bins, device=inputs_embeds.device)
        expand_mask = torch.cat([mask, mask])

        # rotary embeddings
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        for t in torch.arange(0, 1, self.config.dt, device=self.device):
            time_embeddings = self.time_cond_mlp(t.unsqueeze(0).expand(2 * bsz))

            # concat source signal, semantic / phoneme conditioning embed, and conditioning
            # and project
            hidden_states_cond = torch.cat([xt, inputs_embeds], dim=-1)
            hidden_states_uncond = torch.cat([xt, torch.zeros_like(inputs_embeds)], dim=-1)
            hidden_states = torch.cat([hidden_states_cond, hidden_states_uncond])
            hidden_states = self.to_embed(hidden_states)

            # going through the attention layers
            for layer in self.layers:
                hidden_states = layer(hidden_states, expand_mask, position_embeddings, time_embeddings)

            hidden_states = self.norm(hidden_states, time_embeddings)

            # classifier free guidance
            vt = self.to_pred(hidden_states)
            vt_cond, vt_uncond = torch.chunk(vt, 2)
            vt = vt_cond + self.config.cfg_strength * (vt_cond - vt_uncond)

            # Euler method
            xt = xt + vt * self.config.dt

        x1 = xt * self.config.std + self.config.mean
        x1[~mask] = dynamic_range_compression_torch(torch.tensor(0))

        return ModelOutput(spectrogram=x1, durations=duration_predictions)


class FlowMatchingWithBigVGan(PreTrainedModel):
    config_class = FlowMatchingWithBigVGanConfig

    def __init__(self, config: FlowMatchingWithBigVGanConfig):
        super().__init__(config)
        self.model = FlowMatchingModel(config.model_config)
        self.vocoder = BigVGan(config.vocoder_config)

    @classmethod
    def load_pretrained(
        cls,
        model_path,
        vocoder_path,
    ) -> "FlowMatchingWithBigVGan":
        model_config = FlowMatchingConfig.from_pretrained(model_path)
        vocoder_config = BigVGanConfig.from_pretrained(vocoder_path)
        config = FlowMatchingWithBigVGanConfig(model_config.to_dict(), vocoder_config.to_dict())

        model = cls(config)
        model.model = FlowMatchingModel.from_pretrained(model_path)
        model.vocoder = BigVGan.from_pretrained(vocoder_path)
        return model

    @torch.inference_mode()
    def forward(self, input_ids: torch.LongTensor) -> ModelOutput:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input syllabic unit sequence.

        Returns:
            waveform (`list` of `torch.FloatTensor` of shape `(1, (spectrogram_length - 1) * 320 + 400)`):
                Synthesized waveforms.
        """
        outputs = self.model.sample(input_ids)
        waveform = self.vocoder(outputs.spectrogram)
        return ModelOutput(waveform=waveform)
