# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-Video text-to-video (T2V)."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SanaVideoSamplingParams(SamplingParams):
    """Defaults match the SANA-Video 480p diffusers example (832x480, 81 frames,
    50 steps, guidance 6.0, 16 fps)."""

    data_type: DataType = DataType.VIDEO
    num_frames: int = 81
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    height: int | None = 480
    width: int | None = 832
    fps: int = 16
    negative_prompt: str = (
        "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
        "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
        "frames out of sync, inconsistent character shapes, temporal artifacts, "
        "jitter, and ghosting effects, creating a disorienting visual experience."
    )
