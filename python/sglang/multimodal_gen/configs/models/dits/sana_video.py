# SPDX-License-Identifier: Apache-2.0
#
# Architecture/config for the SANA-Video 3D DiT (SanaVideoTransformer3DModel).
# Defaults match the Efficient-Large-Model/SANA-Video_2B_480p_diffusers
# checkpoint (inner_dim = 20*112 = 2240, 20 layers, mlp_ratio 3.0, linear
# self-attn + 3D RoPE + softmax cross-attn + conv FFN). For the 720p checkpoint,
# override sample_size=22 (and the VAE in the pipeline config).

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class SanaVideoArchConfig(DiTArchConfig):
    patch_size: tuple = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 20
    attention_head_dim: int = 112
    num_attention_heads: int = 20
    num_cross_attention_heads: int = 20
    cross_attention_head_dim: int = 112
    cross_attention_dim: int = 2240
    caption_channels: int = 2304
    mlp_ratio: float = 3.0
    qk_norm: str = "rms_norm_across_heads"
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6
    attention_bias: bool = False
    sample_size: int = 30  # 480p; 720p checkpoint uses 22
    rope_max_seq_len: int = 1024

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^transformer\.(.*)$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class SanaVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SanaVideoArchConfig)
    prefix: str = "SanaVideo"
