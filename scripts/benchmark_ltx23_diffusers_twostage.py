#!/usr/bin/env python3
"""Benchmark LTX-2.3 two-stage generation through the Diffusers LTX2 pipeline.

This script intentionally stops at video VAE decode. It does not run video
postprocess, audio VAE decode, vocoder, or file saving.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2LatentUpsamplePipeline,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.utils import export_to_video


DEFAULT_PRETRAINED_MODEL_ID = "diffusers/LTX-2.3-Diffusers"

DEFAULT_MODEL_DIR = os.environ.get("LTX23_MODEL_DIR")
DEFAULT_RUNTIME_MODEL_DIR = os.environ.get("LTX23_DIFFUSERS_RUNTIME_DIR", "outputs/ltx23-diffusers-official-runtime")
DEFAULT_PROMPT = (
    "A cinematic aerial shot of clouds moving across a mountain ridge at sunrise"
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
DEFAULT_STAGE2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def cuda_timer(name: str, timings: dict[str, float]):
    sync()
    start = time.perf_counter()
    yield
    sync()
    timings[name] = time.perf_counter() - start


class ForwardTimer:
    def __init__(self, module: torch.nn.Module, enabled: bool = True):
        self.module = module
        self.enabled = enabled
        self.orig_forward = module.forward
        self.phase = "unset"
        self.times: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def __enter__(self) -> "ForwardTimer":
        if not self.enabled:
            return self
        timer = self

        def wrapped_forward(*args: Any, **kwargs: Any):
            sync()
            start = time.perf_counter()
            out = timer.orig_forward(*args, **kwargs)
            sync()
            elapsed = time.perf_counter() - start
            timer.times[timer.phase] = timer.times.get(timer.phase, 0.0) + elapsed
            timer.calls[timer.phase] = timer.calls.get(timer.phase, 0) + 1
            return out

        self.module.forward = wrapped_forward  # type: ignore[method-assign]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled:
            self.module.forward = self.orig_forward  # type: ignore[method-assign]


def parse_dtype(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def load_latent_upsampler(path: str, dtype: torch.dtype) -> LTX2LatentUpsamplerModel:
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
    config = json.loads(metadata.get("config", "{}"))
    use_rational_resampler = bool(config.get("rational_resampler", False))
    rational_scale = float(config.get("spatial_scale", 2.0))

    upsampler_kwargs = dict(
        in_channels=int(config.get("in_channels", 128)),
        mid_channels=int(config.get("mid_channels", 1024)),
        num_blocks_per_stage=int(config.get("num_blocks_per_stage", 4)),
        dims=int(config.get("dims", 3)),
        spatial_upsample=bool(config.get("spatial_upsample", True)),
        temporal_upsample=bool(config.get("temporal_upsample", False)),
        rational_spatial_scale=rational_scale,
    )
    if "use_rational_resampler" in inspect.signature(LTX2LatentUpsamplerModel.__init__).parameters:
        upsampler_kwargs["use_rational_resampler"] = use_rational_resampler
    elif not use_rational_resampler:
        upsampler_kwargs["rational_spatial_scale"] = None

    model = LTX2LatentUpsamplerModel(**upsampler_kwargs)
    state_dict = load_file(path, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(dtype=dtype)


def load_model_component(
    cls: type[torch.nn.Module],
    model_dir: str,
    subfolder: str,
    weight_name: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    config = cls.load_config(model_dir, subfolder=subfolder)
    model = cls.from_config(config)
    state_dict = load_file(os.path.join(model_dir, subfolder, weight_name), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(dtype=dtype)



def load_pipe(model_dir: str, dtype: torch.dtype, runtime_model_dir: str | None = None) -> LTX2Pipeline:
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_dir, subfolder="scheduler"
    )
    vae = load_model_component(
        AutoencoderKLLTX2Video, model_dir, "vae", "model.safetensors", dtype
    )
    audio_vae = load_model_component(
        AutoencoderKLLTX2Audio,
        model_dir,
        "audio_vae",
        "diffusion_pytorch_model.safetensors",
        dtype,
    )
    transformer = load_model_component(
        LTX2VideoTransformer3DModel,
        model_dir,
        "transformer",
        "model.safetensors",
        dtype,
    )
    connectors = load_model_component(
        LTX2TextConnectors, model_dir, "connectors", "model.safetensors", dtype
    )
    vocoder = load_model_component(
        LTX2Vocoder, model_dir, "vocoder", "model.safetensors", dtype
    )
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        os.path.join(model_dir, "text_encoder"), torch_dtype=dtype
    )
    tokenizer = GemmaTokenizerFast.from_pretrained(os.path.join(model_dir, "tokenizer"))
    return LTX2Pipeline(
        scheduler=scheduler,
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        transformer=transformer,
        vocoder=vocoder,
    )


def maybe_set_adapter(pipe: LTX2Pipeline, adapter_name: str, weight: float) -> None:
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters([adapter_name], adapter_weights=[float(weight)])
        return
    raise RuntimeError("Pipeline does not expose set_adapters; cannot set LoRA weight.")


def has_adapter(pipe: LTX2Pipeline, adapter_name: str) -> bool:
    if hasattr(pipe, "get_list_adapters"):
        adapters = pipe.get_list_adapters()
        if isinstance(adapters, dict):
            return any(adapter_name in names for names in adapters.values())
        return adapter_name in adapters
    if hasattr(pipe, "get_active_adapters"):
        return adapter_name in pipe.get_active_adapters()
    return False


def supported_call_kwargs(pipe: LTX2Pipeline, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(pipe.__call__).parameters
    return {key: value for key, value in kwargs.items() if key in params and value is not None}



def _save_latent_dump(path: Path, tensor: torch.Tensor, tensor_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": tensor.detach().cpu(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "tensor_name": tensor_name,
        },
        path,
    )


@contextmanager
def dump_stage1_initial_latents(pipe: LTX2Pipeline, dump_dir: Path | None):
    if dump_dir is None:
        yield
        return

    dump_dir.mkdir(parents=True, exist_ok=True)
    orig_prepare_latents = pipe.prepare_latents
    orig_prepare_audio_latents = pipe.prepare_audio_latents
    saved = {"video": False, "audio": False}

    def wrapped_prepare_latents(*args: Any, **kwargs: Any):
        out = orig_prepare_latents(*args, **kwargs)
        if not saved["video"]:
            _save_latent_dump(
                dump_dir / "diffusers_stage1_video_initial.pt", out, "video"
            )
            saved["video"] = True
        return out

    def wrapped_prepare_audio_latents(*args: Any, **kwargs: Any):
        out = orig_prepare_audio_latents(*args, **kwargs)
        if not saved["audio"]:
            _save_latent_dump(
                dump_dir / "diffusers_stage1_audio_initial.pt", out, "audio"
            )
            saved["audio"] = True
        return out

    pipe.prepare_latents = wrapped_prepare_latents  # type: ignore[method-assign]
    pipe.prepare_audio_latents = wrapped_prepare_audio_latents  # type: ignore[method-assign]
    try:
        yield
    finally:
        pipe.prepare_latents = orig_prepare_latents  # type: ignore[method-assign]
        pipe.prepare_audio_latents = orig_prepare_audio_latents  # type: ignore[method-assign]



@contextmanager
def dump_stage2_renoise_latents(pipe: LTX2Pipeline, dump_dir: Path | None):
    if dump_dir is None:
        yield
        return

    dump_dir.mkdir(parents=True, exist_ok=True)
    orig_create_noised_state = pipe._create_noised_state
    call_index = {"value": 0}

    def wrapped_create_noised_state(
        latents: torch.Tensor,
        noise_scale: float | torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        out = orig_create_noised_state(latents, noise_scale, generator)
        try:
            scale = float(noise_scale.item() if isinstance(noise_scale, torch.Tensor) else noise_scale)
        except Exception:
            scale = 0.0
        if scale != 0.0:
            tensor_name = "video" if call_index["value"] == 0 else "audio"
            noise = (out.float() - latents.float() * (1.0 - scale)) / scale
            _save_latent_dump(
                dump_dir / f"diffusers_stage2_{tensor_name}_noise.pt",
                noise,
                f"{tensor_name}_noise",
            )
            _save_latent_dump(
                dump_dir / f"diffusers_stage2_{tensor_name}_initial.pt",
                out,
                f"{tensor_name}_initial",
            )
        call_index["value"] += 1
        return out

    pipe._create_noised_state = wrapped_create_noised_state  # type: ignore[method-assign]
    try:
        yield
    finally:
        pipe._create_noised_state = orig_create_noised_state  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, required=DEFAULT_MODEL_DIR is None)
    parser.add_argument("--runtime-model-dir", default=DEFAULT_RUNTIME_MODEL_DIR)
    parser.add_argument("--pretrained-model-id", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-video-path", default="")
    parser.add_argument("--dump-stage1-initial-latents-dir", default="")
    parser.add_argument("--dump-stage2-renoise-dir", default="")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--num-frames", type=int, default=241)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--stage2-guidance-scale", type=float, default=None)
    parser.add_argument("--stg-scale", type=float, default=0.0)
    parser.add_argument("--modality-scale", type=float, default=1.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--audio-guidance-scale", type=float, default=None)
    parser.add_argument("--audio-stg-scale", type=float, default=None)
    parser.add_argument("--audio-modality-scale", type=float, default=None)
    parser.add_argument("--audio-guidance-rescale", type=float, default=None)
    parser.add_argument("--spatio-temporal-guidance-blocks", type=int, nargs="*", default=None)
    parser.add_argument("--use-cross-timestep", action="store_true")
    parser.add_argument("--stage1-steps", type=int, default=30)
    parser.add_argument("--stage2-steps", type=int, default=3)
    parser.add_argument("--stage2-sigmas", type=float, nargs="+", default=DEFAULT_STAGE2_SIGMAS)
    parser.add_argument("--distilled-lora-path", default="")
    parser.add_argument("--stage1-lora-strength", type=float, default=0.0)
    parser.add_argument("--stage2-lora-strength", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-vae-tiling", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--actual-runs", type=int, default=1)
    parser.add_argument("--compile-transformer", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", choices=["none", "true", "false"], default="false")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = parse_dtype(args.dtype)
    model_dir = os.path.abspath(args.model_dir)
    lora_path = (
        os.path.abspath(args.distilled_lora_path)
        if args.distilled_lora_path
        else os.path.join(
            model_dir, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
    )
    if (
        args.stage1_lora_strength != 0.0 or args.stage2_lora_strength != 0.0
    ) and not os.path.exists(lora_path):
        raise FileNotFoundError(
            "Distilled LoRA is required for the requested LoRA strength but was "
            f"not found at {lora_path}. Pass --distilled-lora-path or materialize "
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors."
        )
    upsampler_path = os.path.join(model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    timings: dict[str, float] = {}
    with cuda_timer("load_components_s", timings):
        if args.pretrained_model_id:
            pipe = LTX2Pipeline.from_pretrained(
                args.pretrained_model_id,
                torch_dtype=dtype,
                local_files_only=args.local_files_only,
            )
        else:
            runtime_model_dir = os.path.abspath(args.runtime_model_dir) if args.runtime_model_dir else None
            pipe = load_pipe(model_dir, dtype=dtype, runtime_model_dir=runtime_model_dir)
        latent_upsampler = load_latent_upsampler(upsampler_path, dtype=dtype)
        upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=pipe.vae, latent_upsampler=latent_upsampler
        )
        pipe.to(device=args.device, dtype=dtype)
        upsample_pipe.to(device=args.device, dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        upsample_pipe.set_progress_bar_config(disable=True)
        if args.enable_vae_tiling:
            pipe.vae.enable_tiling()
            upsample_pipe.vae.enable_tiling()
        if args.compile_transformer:
            compile_kwargs: dict[str, Any] = {
                "backend": args.compile_backend,
                "fullgraph": args.compile_fullgraph,
            }
            if args.compile_mode:
                compile_kwargs["mode"] = args.compile_mode
            if args.compile_dynamic != "none":
                compile_kwargs["dynamic"] = args.compile_dynamic == "true"
            pipe.transformer.compile(**compile_kwargs)
        stage1_scheduler_config = dict(pipe.scheduler.config)
        stage2_scheduler_config = dict(pipe.scheduler.config)

    half_height = args.height // 2
    half_width = args.width // 2

    def run_once(run_label: str) -> dict[str, Any]:
        run_timings: dict[str, float] = {}
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        should_save_video = bool(args.output_video_path) and run_label == "actual"
        with ForwardTimer(pipe.transformer, enabled=not args.compile_transformer) as forward_timer:
            if hasattr(pipe, "get_active_adapters") and pipe.get_active_adapters():
                maybe_set_adapter(pipe, "distilled", args.stage1_lora_strength)
            if args.stage1_lora_strength != 0.0:
                if not has_adapter(pipe, "distilled"):
                    with cuda_timer(f"{run_label}.load_stage1_lora_s", run_timings):
                        pipe.load_lora_weights(lora_path, adapter_name="distilled")
                maybe_set_adapter(pipe, "distilled", args.stage1_lora_strength)

            stage1_dump_dir = (
                Path(args.dump_stage1_initial_latents_dir)
                if args.dump_stage1_initial_latents_dir and run_label == "actual"
                else None
            )
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                stage1_scheduler_config
            )
            forward_timer.phase = "stage1"
            with dump_stage1_initial_latents(pipe, stage1_dump_dir):
                with cuda_timer(f"{run_label}.stage1_pipeline_s", run_timings):
                    stage1_video_latents, stage1_audio_latents = pipe(
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                        height=half_height,
                        width=half_width,
                        num_frames=args.num_frames,
                        frame_rate=args.fps,
                        num_inference_steps=args.stage1_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        **supported_call_kwargs(
                            pipe,
                            {
                                "stg_scale": args.stg_scale,
                                "modality_scale": args.modality_scale,
                                "guidance_rescale": args.guidance_rescale,
                                "audio_guidance_scale": args.audio_guidance_scale,
                                "audio_stg_scale": args.audio_stg_scale,
                                "audio_modality_scale": args.audio_modality_scale,
                                "audio_guidance_rescale": args.audio_guidance_rescale,
                                "spatio_temporal_guidance_blocks": args.spatio_temporal_guidance_blocks,
                                "use_cross_timestep": args.use_cross_timestep,
                            },
                        ),
                        output_type="latent",
                        return_dict=False,
                    )

            with cuda_timer(f"{run_label}.latent_upsample_s", run_timings):
                upsampled_video_latents = upsample_pipe(
                    latents=stage1_video_latents,
                    latents_normalized=False,
                    height=half_height,
                    width=half_width,
                    num_frames=args.num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]

            if args.stage1_lora_strength == 0.0 and not has_adapter(pipe, "distilled"):
                with cuda_timer(f"{run_label}.load_stage2_lora_s", run_timings):
                    pipe.load_lora_weights(lora_path, adapter_name="distilled")
            maybe_set_adapter(pipe, "distilled", args.stage2_lora_strength)

            stage2_dump_dir = (
                Path(args.dump_stage2_renoise_dir)
                if args.dump_stage2_renoise_dir and run_label == "actual"
                else None
            )
            forward_timer.phase = "stage2"
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                stage2_scheduler_config,
                use_dynamic_shifting=False,
                shift_terminal=None,
            )
            with dump_stage2_renoise_latents(pipe, stage2_dump_dir):
                with cuda_timer(f"{run_label}.stage2_pipeline_s", run_timings):
                    stage2_video_latents, stage2_audio_latents = pipe(
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        frame_rate=args.fps,
                        num_inference_steps=args.stage2_steps,
                        sigmas=args.stage2_sigmas,
                        guidance_scale=(
                            args.stage2_guidance_scale
                            if args.stage2_guidance_scale is not None
                            else args.guidance_scale
                        ),
                        noise_scale=float(args.stage2_sigmas[0]),
                        latents=upsampled_video_latents,
                        audio_latents=stage1_audio_latents,
                        generator=generator,
                        output_type="latent",
                        return_dict=False,
                    )

            with cuda_timer(f"{run_label}.video_vae_decode_s", run_timings):
                decode_timestep = None
                if pipe.vae.config.timestep_conditioning:
                    decode_timestep = torch.tensor(
                        [0.0], device=args.device, dtype=stage2_video_latents.dtype
                    )
                _video = pipe.vae.decode(
                    stage2_video_latents.to(pipe.vae.dtype),
                    decode_timestep,
                    return_dict=False,
                )[0]

            if should_save_video:
                output_video_path = Path(args.output_video_path)
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                with cuda_timer(f"{run_label}.video_postprocess_save_s", run_timings):
                    frames = pipe.video_processor.postprocess_video(_video, output_type="np")[0]
                    export_to_video(
                        frames,
                        str(output_video_path),
                        fps=int(args.fps),
                        quality=8,
                        macro_block_size=1,
                    )

        stage1_key = f"{run_label}.stage1_pipeline_s"
        stage2_key = f"{run_label}.stage2_pipeline_s"
        decode_key = f"{run_label}.video_vae_decode_s"
        strict_pipeline_s = run_timings[stage1_key] + run_timings[stage2_key] + run_timings[decode_key]
        strict_transformer_s = (
            forward_timer.times.get("stage1", 0.0)
            + forward_timer.times.get("stage2", 0.0)
            + run_timings[decode_key]
        )
        return {
            "timings_s": run_timings,
            "transformer_forward_s": dict(forward_timer.times),
            "transformer_forward_calls": dict(forward_timer.calls),
            "strict_pipeline_s": strict_pipeline_s,
            "strict_transformer_plus_decode_s": strict_transformer_s,
            "output_shapes": {
                "stage1_video_latents": list(stage1_video_latents.shape),
                "stage1_audio_latents": list(stage1_audio_latents.shape),
                "upsampled_video_latents": list(upsampled_video_latents.shape),
                "stage2_video_latents": list(stage2_video_latents.shape),
                "stage2_audio_latents": list(stage2_audio_latents.shape),
            },
        }

    if args.warmup:
        warmup_result = run_once("warmup")
        del warmup_result
        torch.cuda.empty_cache()

    actual_results = []
    for actual_idx in range(args.actual_runs):
        run_label = "actual" if actual_idx == args.actual_runs - 1 else f"actual_warm{actual_idx + 1}"
        actual_results.append(run_once(run_label))
    result = actual_results[-1]
    if len(actual_results) > 1:
        result["preceding_actual_results"] = actual_results[:-1]
    result.update(
        {
            "load_timings_s": timings,
            "model_dir": model_dir,
            "runtime_model_dir": os.path.abspath(args.runtime_model_dir) if args.runtime_model_dir else None,
            "pretrained_model_id": args.pretrained_model_id,
            "lora_path": lora_path,
            "upsampler_path": upsampler_path,
            "stage2_scheduler_reset": {
                "use_dynamic_shifting": False,
                "shift_terminal": None,
            },
            "diffusers_version": __import__("diffusers").__version__,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
            "params": vars(args),
        }
    )
    out_path = output_dir / "perf_diffusers.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
