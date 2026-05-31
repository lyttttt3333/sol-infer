#!/usr/bin/env python3
"""Runtime-only benchmark for the official LTX-2.3 HQ two-stage pipeline.

The official CLI builds and frees several models inside the pipeline call. This
wrapper keeps the official pipeline semantics and hyperparameters, but reports a
runtime-only total that excludes model/component build time. Timed segments are:
prompt encoding compute, stage-1 denoising, latent upsample, stage-2 denoising,
video decode, and audio decode. Video file encoding is outside the timed region.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode
from sglang.multimodal_gen.runtime.utils.dit_activation_dump import ActivationDumpContext


DEFAULT_PROMPT = (
    "A cinematic 10 second aerial shot of an antique brass clockwork train crossing "
    "a snowy mountain bridge at sunrise, steam drifting through golden light, "
    "smooth camera movement, high detail"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timer(name: str, timings: dict[str, float]):
    sync()
    start = time.perf_counter()
    yield
    sync()
    timings[name] = time.perf_counter() - start


def build_with_timer(name: str, timings: dict[str, float], fn):
    with timer(name, timings):
        return fn()


class Stage1DebugEarlyExit(RuntimeError):
    def __init__(self, calls: int):
        super().__init__(f"stage1 debug early exit after {calls} denoiser calls")
        self.calls = int(calls)


class Stage1DebugDenoiser:
    def __init__(
        self,
        inner,
        dump_dir: Path | None,
        max_calls: int = 2,
        activation_dump_dir: Path | None = None,
        activation_dump_calls: int = 1,
        *,
        stage_name: str = "stage1",
        stop_after_env: str = "OFFICIAL_LTX2_STOP_AFTER_STAGE1_DENOISER_CALLS",
    ):
        self.inner = inner
        self.dump_dir = dump_dir
        self.max_calls = max_calls
        self.activation_dump_dir = activation_dump_dir
        self.activation_dump_calls = activation_dump_calls
        self.stage_name = stage_name
        self.stop_after_env = stop_after_env
        self.call_index = 0

    @staticmethod
    def _cpu(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu()
        return value

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
        if __import__("os").environ.get("OFFICIAL_LTX2_ATTENTION_DEBUG_DIR"):
            for module_name, module in transformer.named_modules():
                try:
                    setattr(module, "_ltx2_debug_name", module_name)
                except Exception:
                    pass
        activation_ctx = nullcontext()
        activation_dump_start_call = int(
            os.environ.get("OFFICIAL_LTX2_DIT_ACTIVATION_DUMP_START_CALL", "0") or 0
        )
        if (
            self.activation_dump_dir is not None
            and activation_dump_start_call <= self.call_index < activation_dump_start_call + self.activation_dump_calls
        ):
            activation_target = getattr(transformer, "_model", transformer)
            activation_target = getattr(activation_target, "velocity_model", activation_target)
            activation_ctx = ActivationDumpContext(
                activation_target,
                self.activation_dump_dir,
                prefix=f"official_{self.stage_name}_call_{self.call_index:02d}",
                name_pattern=__import__("os").environ.get(
                    "OFFICIAL_LTX2_DIT_ACTIVATION_DUMP_PATTERN", ""
                ),
                max_events=int(
                    __import__("os").environ.get(
                        "OFFICIAL_LTX2_DIT_ACTIVATION_DUMP_MAX_EVENTS", "1000"
                    )
                    or 1000
                ),
                include_root=__import__("os").environ.get(
                    "OFFICIAL_LTX2_DIT_ACTIVATION_DUMP_INCLUDE_ROOT", "0"
                ).lower()
                in ("1", "true", "yes", "on"),
                save_tensors=__import__("os").environ.get(
                    "OFFICIAL_LTX2_DIT_ACTIVATION_DUMP_SAVE_TENSORS", "0"
                ).lower()
                in ("1", "true", "yes", "on"),
                max_tensor_events=int(
                    __import__("os").environ.get(
                        "OFFICIAL_LTX2_DIT_ACTIVATION_DUMP_MAX_TENSOR_EVENTS", "20"
                    )
                    or 20
                ),
                max_sample=int(
                    __import__("os").environ.get(
                        "OFFICIAL_LTX2_DIT_ACTIVATION_DUMP_MAX_SAMPLE", "8"
                    )
                    or 8
                ),
                hash_tensors=__import__("os").environ.get(
                    "OFFICIAL_LTX2_DIT_ACTIVATION_DUMP_HASH_TENSORS", "1"
                ).lower()
                not in ("0", "false", "no", "off"),
            )
        with activation_ctx:
            video_result, audio_result = self.inner(
                transformer, video_state, audio_state, sigmas, step_index
            )
        if self.dump_dir is not None and self.call_index < self.max_calls:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "call_index": self.call_index,
                    "step_index": int(step_index),
                    "sigmas": sigmas.detach().cpu(),
                    "video_latents_in": self._cpu(video_state.latent if video_state is not None else None),
                    "audio_latents_in": self._cpu(audio_state.latent if audio_state is not None else None),
                    "video_denoised": self._cpu(getattr(video_result, "denoised", None) if video_result is not None else None),
                    "audio_denoised": self._cpu(getattr(audio_result, "denoised", None) if audio_result is not None else None),
                    "video_cond": self._cpu(getattr(video_result, "cond", None) if video_result is not None else None),
                    "audio_cond": self._cpu(getattr(audio_result, "cond", None) if audio_result is not None else None),
                    "video_uncond": self._cpu(getattr(video_result, "uncond", None) if video_result is not None else None),
                    "audio_uncond": self._cpu(getattr(audio_result, "uncond", None) if audio_result is not None else None),
                    "video_mod": self._cpu(getattr(video_result, "mod", None) if video_result is not None else None),
                    "audio_mod": self._cpu(getattr(audio_result, "mod", None) if audio_result is not None else None),
                    "video_ptb": self._cpu(getattr(video_result, "ptb", None) if video_result is not None else None),
                    "audio_ptb": self._cpu(getattr(audio_result, "ptb", None) if audio_result is not None else None),
                },
                self.dump_dir / f"official_{self.stage_name}_denoiser_call_{self.call_index:02d}.pt",
            )
        self.call_index += 1
        stop_after = int(os.environ.get(self.stop_after_env, "0") or 0)
        if stop_after > 0 and self.call_index >= stop_after:
            raise Stage1DebugEarlyExit(self.call_index)
        return video_result, audio_result


class DumpingGaussianNoiser:
    def __init__(self, generator: torch.Generator, dump_dir: Path | None = None):
        self.generator = generator
        self.dump_dir = dump_dir
        self.call_index = 0
        self.names = (
            "stage1_video",
            "stage1_audio",
            "stage2_video",
            "stage2_audio",
        )

    def __call__(self, latent_state, noise_scale: float = 1.0):
        noise = torch.randn(
            *latent_state.latent.shape,
            device=latent_state.latent.device,
            dtype=latent_state.latent.dtype,
            generator=self.generator,
        )
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
        result = replace(latent_state, latent=latent.to(latent_state.latent.dtype))
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            name = self.names[self.call_index] if self.call_index < len(self.names) else f"call{self.call_index}"
            torch.save(
                {
                    "latents": result.latent.detach().cpu(),
                    "noise": noise.detach().cpu(),
                    "clean_latents": latent_state.latent.detach().cpu(),
                    "denoise_mask": latent_state.denoise_mask.detach().cpu(),
                    "noise_scale": float(noise_scale),
                    "shape": list(result.latent.shape),
                    "dtype": str(result.latent.dtype),
                    "call_index": self.call_index,
                    "name": name,
                },
                self.dump_dir / f"official_{name}.pt",
            )
        self.call_index += 1
        return result


def video_tools_for(pixel_shape: VideoPixelShape) -> VideoLatentTools:
    return VideoLatentTools(
        VideoLatentPatchifier(patch_size=1),
        VideoLatentShape.from_pixel_shape(pixel_shape),
        pixel_shape.fps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--distilled-lora", required=True)
    parser.add_argument("--spatial-upsampler-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--output-video-path", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--num-frames", type=int, default=241)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=15)
    parser.add_argument("--stage1-lora-strength", type=float, default=0.25)
    parser.add_argument("--stage2-lora-strength", type=float, default=0.5)
    parser.add_argument("--video-cfg-guidance-scale", type=float, default=3.0)
    parser.add_argument("--video-stg-guidance-scale", type=float, default=0.0)
    parser.add_argument("--video-rescale-scale", type=float, default=0.45)
    parser.add_argument("--a2v-guidance-scale", type=float, default=3.0)
    parser.add_argument("--audio-cfg-guidance-scale", type=float, default=7.0)
    parser.add_argument("--audio-stg-guidance-scale", type=float, default=0.0)
    parser.add_argument("--audio-rescale-scale", type=float, default=1.0)
    parser.add_argument("--v2a-guidance-scale", type=float, default=3.0)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--dump-noise-dir", default="")
    parser.add_argument("--dump-context-dir", default="")
    parser.add_argument("--dump-stage1-debug-dir", default="")
    parser.add_argument("--dump-stage1-debug-calls", type=int, default=2)
    parser.add_argument("--dump-stage2-debug-dir", default="")
    parser.add_argument("--dump-stage2-debug-calls", type=int, default=4)
    parser.add_argument("--dump-dit-activations-dir", default="")
    parser.add_argument("--dump-dit-activations-calls", type=int, default=1)
    parser.add_argument("--stop-after-context", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    timings: dict[str, float] = {}
    build_timings: dict[str, float] = {}
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            str(Path(args.distilled_lora)),
            1.0,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]

    pipeline = build_with_timer(
        "build_pipeline_s",
        build_timings,
        lambda: TI2VidTwoStagesHQPipeline(
            checkpoint_path=args.checkpoint_path,
            distilled_lora=distilled_lora,
            distilled_lora_strength_stage_1=args.stage1_lora_strength,
            distilled_lora_strength_stage_2=args.stage2_lora_strength,
            spatial_upsampler_path=args.spatial_upsampler_path,
            gemma_root=args.gemma_root,
            loras=(),
            offload_mode=OffloadMode.NONE,
        ),
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    dump_noise_dir = Path(args.dump_noise_dir) if args.dump_noise_dir else None
    noiser = DumpingGaussianNoiser(generator=generator, dump_dir=dump_noise_dir)
    dtype = torch.bfloat16
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    text_encoder = build_with_timer(
        "build_text_encoder_s",
        build_timings,
        pipeline.prompt_encoder._build_text_encoder,
    )
    embeddings_processor = build_with_timer(
        "build_embeddings_processor_s",
        build_timings,
        pipeline.prompt_encoder._build_embeddings_processor,
    )
    with timer("text_encoding_s", timings):
        raw_outputs = [
            text_encoder.encode(args.prompt),
            text_encoder.encode(args.negative_prompt),
        ]
        ctx_p, ctx_n = [
            embeddings_processor.process_hidden_states(hs, mask)
            for hs, mask in raw_outputs
        ]
    del text_encoder, embeddings_processor
    torch.cuda.empty_cache()

    v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
    v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
    dump_context_dir = Path(args.dump_context_dir) if args.dump_context_dir else None
    if dump_context_dir is not None:
        dump_context_dir.mkdir(parents=True, exist_ok=True)
        raw_prompt_embeds, raw_prompt_mask = raw_outputs[0]
        raw_negative_prompt_embeds, raw_negative_prompt_mask = raw_outputs[1]

        def cpu_tree(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, tuple):
                return tuple(cpu_tree(v) for v in value)
            if isinstance(value, list):
                return [cpu_tree(v) for v in value]
            if isinstance(value, dict):
                return {k: cpu_tree(v) for k, v in value.items()}
            return value

        torch.save(
            {
                "raw_prompt_embeds": cpu_tree(raw_prompt_embeds),
                "raw_prompt_attention_mask": cpu_tree(raw_prompt_mask),
                "raw_negative_prompt_embeds": cpu_tree(raw_negative_prompt_embeds),
                "raw_negative_attention_mask": cpu_tree(raw_negative_prompt_mask),
                "video_context_pos": v_context_p.detach().cpu(),
                "audio_context_pos": a_context_p.detach().cpu(),
                "video_context_neg": v_context_n.detach().cpu(),
                "audio_context_neg": a_context_n.detach().cpu(),
            },
            dump_context_dir / "official_contexts.pt",
        )
    if args.stop_after_context:
        summary = {
            "variant": "official_hq_context_only",
            "context_dump": str(dump_context_dir / "official_contexts.pt") if dump_context_dir is not None else "",
            "timings_s": timings,
            "excluded_build_timings_s": build_timings,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.frame_rate,
            "num_inference_steps_stage1": args.num_inference_steps,
            "distilled_lora_strength_stage1": args.stage1_lora_strength,
            "distilled_lora_strength_stage2": args.stage2_lora_strength,
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    stepper = Res2sDiffusionStep()
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg_guidance_scale,
        stg_scale=args.video_stg_guidance_scale,
        rescale_scale=args.video_rescale_scale,
        modality_scale=args.a2v_guidance_scale,
        skip_step=0,
        stg_blocks=(),
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_guidance_scale,
        stg_scale=args.audio_stg_guidance_scale,
        rescale_scale=args.audio_rescale_scale,
        modality_scale=args.v2a_guidance_scale,
        skip_step=0,
        stg_blocks=(),
    )

    stage1_shape = VideoPixelShape(
        batch=1,
        frames=args.num_frames,
        width=args.width // 2,
        height=args.height // 2,
        fps=args.frame_rate,
    )
    empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage1_shape).to_torch_shape())
    stage1_sigmas = pipeline._scheduler.execute(
        latent=empty_latent,
        steps=args.num_inference_steps,
    ).to(dtype=torch.float32, device=pipeline.device)

    stage1_debug_dir = (
        Path(args.dump_stage1_debug_dir) if args.dump_stage1_debug_dir else None
    )
    stage1_denoiser = Stage1DebugDenoiser(
        GuidedDenoiser(
            v_context=v_context_p,
            a_context=a_context_p,
            video_guider=MultiModalGuider(
                params=video_guider_params,
                negative_context=v_context_n,
            ),
            audio_guider=MultiModalGuider(
                params=audio_guider_params,
                negative_context=a_context_n,
            ),
        ),
        stage1_debug_dir,
        args.dump_stage1_debug_calls,
        Path(args.dump_dit_activations_dir) if args.dump_dit_activations_dir else None,
        args.dump_dit_activations_calls,
    )

    with pipeline.stage_1.model_context(video_tools=video_tools_for(stage1_shape)) as transformer:
        with timer("stage1_denoise_s", timings):
            old_res2s_debug_dir = os.environ.get("OFFICIAL_LTX2_RES2S_DEBUG_DIR")
            if stage1_debug_dir is not None:
                os.environ["OFFICIAL_LTX2_RES2S_DEBUG_DIR"] = str(stage1_debug_dir / "res2s")
            early_exit_calls = None
            try:
                video_state, audio_state = pipeline.stage_1.run(
                    transformer=transformer,
                    denoiser=stage1_denoiser,
                    sigmas=stage1_sigmas,
                    noiser=noiser,
                    stepper=stepper,
                    width=stage1_shape.width,
                    height=stage1_shape.height,
                    frames=args.num_frames,
                    fps=args.frame_rate,
                    video=ModalitySpec(context=v_context_p, conditionings=[]),
                    audio=ModalitySpec(context=a_context_p),
                    loop=res2s_audio_video_denoising_loop,
                    max_batch_size=args.max_batch_size,
                )
            except Stage1DebugEarlyExit as exc:
                early_exit_calls = exc.calls
            finally:
                if old_res2s_debug_dir is None:
                    os.environ.pop("OFFICIAL_LTX2_RES2S_DEBUG_DIR", None)
                else:
                    os.environ["OFFICIAL_LTX2_RES2S_DEBUG_DIR"] = old_res2s_debug_dir
    if early_exit_calls is not None:
        summary = {
            "variant": "official_hq_stage1_debug_early_exit",
            "early_exit_after_stage1_denoiser_calls": int(early_exit_calls),
            "dump_stage1_debug_dir": str(stage1_debug_dir) if stage1_debug_dir is not None else "",
            "dump_dit_activations_dir": args.dump_dit_activations_dir,
            "timings_s": timings,
            "excluded_build_timings_s": build_timings,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.frame_rate,
            "num_inference_steps_stage1": args.num_inference_steps,
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    if stage1_debug_dir is not None:
        stage1_debug_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "video_stage1_final": video_state.latent.detach().cpu(),
                "audio_stage1_final": audio_state.latent.detach().cpu(),
            },
            stage1_debug_dir / "official_stage1_final.pt",
        )
    torch.cuda.empty_cache()

    video_encoder = build_with_timer(
        "build_upsample_video_encoder_s",
        build_timings,
        lambda: pipeline.upsampler._encoder_builder.build(
            device=pipeline.device, dtype=pipeline.dtype
        ).eval(),
    )
    upsampler = build_with_timer(
        "build_latent_upsampler_s",
        build_timings,
        lambda: pipeline.upsampler._upsampler_builder.build(
            device=pipeline.device, dtype=pipeline.dtype
        ).eval(),
    )
    with timer("latent_upsample_s", timings):
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=upsampler,
        )
    if stage1_debug_dir is not None:
        torch.save(
            {
                "video_post_upsample_unpacked": upscaled_video_latent.detach().cpu(),
            },
            stage1_debug_dir / "official_stage1_post_upsample.pt",
        )
    del video_encoder, upsampler
    torch.cuda.empty_cache()

    stage2_shape = VideoPixelShape(
        batch=1,
        frames=args.num_frames,
        width=args.width,
        height=args.height,
        fps=args.frame_rate,
    )
    stage2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(
        dtype=torch.float32, device=pipeline.device
    )
    stage2_debug_dir = (
        Path(args.dump_stage2_debug_dir) if args.dump_stage2_debug_dir else None
    )
    stage2_denoiser = Stage1DebugDenoiser(
        SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
        stage2_debug_dir,
        args.dump_stage2_debug_calls,
        None,
        0,
        stage_name="stage2",
        stop_after_env="OFFICIAL_LTX2_STOP_AFTER_STAGE2_DENOISER_CALLS",
    )
    with pipeline.stage_2.model_context(video_tools=video_tools_for(stage2_shape)) as transformer:
        with timer("stage2_denoise_s", timings):
            old_res2s_debug_dir = os.environ.get("OFFICIAL_LTX2_RES2S_DEBUG_DIR")
            if stage2_debug_dir is not None:
                os.environ["OFFICIAL_LTX2_RES2S_DEBUG_DIR"] = str(stage2_debug_dir / "res2s")
            try:
                video_state, audio_state = pipeline.stage_2.run(
                    transformer=transformer,
                    denoiser=stage2_denoiser,
                    sigmas=stage2_sigmas,
                    noiser=noiser,
                    stepper=stepper,
                    width=args.width,
                    height=args.height,
                    frames=args.num_frames,
                    fps=args.frame_rate,
                    video=ModalitySpec(
                        context=v_context_p,
                        conditionings=[],
                        noise_scale=stage2_sigmas[0].item(),
                        initial_latent=upscaled_video_latent,
                    ),
                    audio=ModalitySpec(
                        context=a_context_p,
                        noise_scale=stage2_sigmas[0].item(),
                        initial_latent=audio_state.latent,
                    ),
                    loop=res2s_audio_video_denoising_loop,
                )
            finally:
                if old_res2s_debug_dir is None:
                    os.environ.pop("OFFICIAL_LTX2_RES2S_DEBUG_DIR", None)
                else:
                    os.environ["OFFICIAL_LTX2_RES2S_DEBUG_DIR"] = old_res2s_debug_dir
    if stage2_debug_dir is not None:
        stage2_debug_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "video_stage2_final": video_state.latent.detach().cpu(),
                "audio_stage2_final": audio_state.latent.detach().cpu(),
            },
            stage2_debug_dir / "official_stage2_final.pt",
        )
    torch.cuda.empty_cache()

    video_decoder = build_with_timer(
        "build_video_decoder_s",
        build_timings,
        lambda: pipeline.video_decoder._decoder_builder.build(
            device=pipeline.device, dtype=pipeline.dtype
        ).eval(),
    )
    with timer("video_decode_s", timings):
        video_chunks = [
            chunk.detach()
            for chunk in video_decoder.decode_video(
                video_state.latent, tiling_config, generator
            )
        ]
    del video_decoder
    torch.cuda.empty_cache()

    audio_decoder = build_with_timer(
        "build_audio_decoder_s",
        build_timings,
        lambda: pipeline.audio_decoder._decoder_builder.build(
            device=pipeline.device, dtype=pipeline.dtype
        ).eval(),
    )
    vocoder = build_with_timer(
        "build_vocoder_s",
        build_timings,
        lambda: pipeline.audio_decoder._vocoder_builder.build(
            device=pipeline.device, dtype=pipeline.dtype
        ).eval(),
    )
    with timer("audio_decode_s", timings):
        audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
    del audio_decoder, vocoder
    torch.cuda.empty_cache()

    runtime_only_total_s = sum(timings.values())
    output_path = Path(args.output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_start = time.perf_counter()
    encode_video(
        video=iter(video_chunks),
        fps=int(args.frame_rate),
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=video_chunks_number,
    )
    encode_video_s = time.perf_counter() - encode_start

    summary: dict[str, Any] = {
        "variant": "official_hq_runtime_only_loaded_components",
        "pipeline_source": "https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py",
        "output_video": str(output_path),
        "runtime_only_total_s": runtime_only_total_s,
        "timings_s": timings,
        "excluded_build_timings_s": build_timings,
        "encode_video_s_not_counted": encode_video_s,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.frame_rate,
        "num_inference_steps_stage1": args.num_inference_steps,
        "stage2_sigmas": [float(x) for x in STAGE_2_DISTILLED_SIGMAS],
        "distilled_lora_strength_stage1": args.stage1_lora_strength,
        "distilled_lora_strength_stage2": args.stage2_lora_strength,
        "video_cfg_guidance_scale": args.video_cfg_guidance_scale,
        "video_stg_guidance_scale": args.video_stg_guidance_scale,
        "video_rescale_scale": args.video_rescale_scale,
        "audio_cfg_guidance_scale": args.audio_cfg_guidance_scale,
        "audio_stg_guidance_scale": args.audio_stg_guidance_scale,
        "audio_rescale_scale": args.audio_rescale_scale,
        "dump_noise_dir": str(dump_noise_dir) if dump_noise_dir is not None else "",
        "dump_context_dir": str(dump_context_dir) if dump_context_dir is not None else "",
        "dump_stage2_debug_dir": str(stage2_debug_dir) if stage2_debug_dir is not None else "",
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
