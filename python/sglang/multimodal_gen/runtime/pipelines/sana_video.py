# SPDX-License-Identifier: Apache-2.0
"""SANA-Video text-to-video pipeline (sglang multimodal_gen).

Minimal wiring: standard stages (text-encode -> latent-prep -> denoise ->
decode). The DPMSolverMultistep scheduler and Gemma2 encoder + Wan/LTX-2 VAE are
loaded from the checkpoint per the pipeline config. The 3D RoPE is computed
inside SanaVideoTransformer3DModel, so no model-specific stage is required.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SanaVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    """SANA-Video T2V pipeline. pipeline_name matches model_index.json _class_name."""

    pipeline_name = "SanaVideoPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # add_standard_t2i_stages drives text-encode -> latent-prep -> denoise ->
        # decode; 5D video latents come from PipelineConfig.prepare_latent_shape
        # and the video VAE handles 5D decode (same path Wan T2V uses).
        self.add_standard_t2i_stages()


EntryClass = SanaVideoPipeline
