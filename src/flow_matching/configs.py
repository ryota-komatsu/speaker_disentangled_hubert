from typing import Dict, Optional

from transformers import PretrainedConfig

from ..bigvgan.bigvgan import BigVGanConfig


class FlowMatchingConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 16384,
        num_mel_bins: int = 80,
        embedding_dim: int = 768,
        hidden_size: int = 512,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 2,
        intermediate_size: int = 768,
        attention_dropout: float = 0.0,
        cfg_dropout: float = 0.2,
        mean: float = -5.8843,
        std: float = 2.2615,
        rope_theta: float = 10000.0,
        max_position_embeddings=None,
        dt: float = 0.1,
        cfg_strength: float = 0.7,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.cfg_dropout = cfg_dropout
        self.mean = mean
        self.std = std
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.dt = dt
        self.cfg_strength = cfg_strength
        self.rms_norm_eps = rms_norm_eps
        super().__init__(**kwargs)


class FlowMatchingWithBigVGanConfig(PretrainedConfig):
    model_type = "flow_matching_with_bigvgan"
    sub_configs = {"model_config": FlowMatchingConfig, "vocoder_config": BigVGanConfig}

    def __init__(
        self,
        model_config: Optional[Dict] = None,
        vocoder_config: Optional[Dict] = None,
        **kwargs,
    ):
        if model_config is None:
            model_config = {}

        if vocoder_config is None:
            vocoder_config = {}

        self.model_config = FlowMatchingConfig(**model_config)
        self.vocoder_config = BigVGanConfig(**vocoder_config)
        super().__init__(**kwargs)
