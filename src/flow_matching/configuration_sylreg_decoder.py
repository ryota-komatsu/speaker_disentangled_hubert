from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniBigVGANConfig


class FlowMatchingConfig(PreTrainedConfig):
    vocab_size: int = 8192
    num_mel_bins: int = 80
    embedding_dim: int = 768
    hidden_size: int = 512
    num_hidden_layers: int = 4
    num_encoder_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 1024
    attention_dropout: float = 0.0
    cfg_dropout: float = 0.2
    mean: float = -5.8843
    std: float = 2.2615
    rope_theta: float = 10000.0
    max_position_embeddings: None = None
    dt: float = 0.1
    cfg_strength: float = 0.7
    rope_parameters: RopeParameters | dict | None = None


class FlowMatchingWithBigVGanConfig(PreTrainedConfig):
    model_type = "flow_matching_with_bigvgan"
    sub_configs = {"model_config": FlowMatchingConfig, "vocoder_config": Qwen2_5OmniBigVGANConfig}

    model_config: dict | PreTrainedConfig | None = None
    vocoder_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.model_config is None:
            self.model_config = FlowMatchingConfig()
        elif isinstance(self.model_config, dict):
            self.model_config = FlowMatchingConfig(**self.model_config)

        if self.vocoder_config is None:
            self.vocoder_config = Qwen2_5OmniBigVGANConfig()
        elif isinstance(self.vocoder_config, dict):
            self.vocoder_config = Qwen2_5OmniBigVGANConfig(**self.vocoder_config)

        super().__post_init__(**kwargs)
