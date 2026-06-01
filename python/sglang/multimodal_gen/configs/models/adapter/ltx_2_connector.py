from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.adapter.base import (
    AdapterArchConfig,
    AdapterConfig,
)


@dataclass
class LTX2ConnectorArchConfig(AdapterArchConfig):
    audio_connector_attention_head_dim: int = 128
    audio_connector_num_attention_heads: int = 30
    audio_connector_num_layers: int = 2
    audio_connector_num_learnable_registers: int = 128
    audio_gated_attn: bool = False
    audio_hidden_dim: int = 0
    audio_feature_extractor_out_features: int = 0
    caption_channels: int = 3840
    causal_temporal_positioning: bool = False
    connector_rope_base_seq_len: int = 4096
    connector_apply_gated_attention: bool = False
    feature_extractor_in_features: int = 0
    per_modality_projections: bool = False
    proj_bias: bool = False
    rope_double_precision: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "split"
    text_proj_in_factor: int = 49
    video_feature_extractor_out_features: int = 0
    video_connector_attention_head_dim: int = 128
    video_connector_num_attention_heads: int = 30
    video_connector_num_layers: int = 2
    video_connector_num_learnable_registers: int = 128
    video_gated_attn: bool = False
    video_hidden_dim: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.video_hidden_dim <= 0:
            self.video_hidden_dim = (
                self.video_connector_num_attention_heads
                * self.video_connector_attention_head_dim
            )
        if self.audio_hidden_dim <= 0:
            self.audio_hidden_dim = (
                self.audio_connector_num_attention_heads
                * self.audio_connector_attention_head_dim
            )


@dataclass
class LTX2ConnectorConfig(AdapterConfig):

    arch_config: AdapterArchConfig = field(default_factory=LTX2ConnectorArchConfig)

    prefix: str = "LTX2"
