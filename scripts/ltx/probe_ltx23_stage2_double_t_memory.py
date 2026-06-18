#!/usr/bin/env python3
"""Probe LTX-2.3 stage-2 DiT memory with doubled temporal tokens.

This is a synthetic forward-only check. It loads the stage-2 transformer and
constructs dummy tensors matching the 1080p HQ stage-2 token layout. It does not
run stage 1, decode, or save a generated video.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def _set_default_cache_dirs(repo_root: Path) -> None:
    cache_root = repo_root / "outputs" / ".cache"
    defaults = {
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch_extensions",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "SGLANG_DIFFUSION_CACHE_ROOT": cache_root / "sgl_diffusion",
        "TMPDIR": repo_root / "outputs" / ".tmp",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _enable_current_kernel_env(*, te_nvfp4: bool) -> None:
    # Match the current "best" stage-2 kernel/runtime path as closely as a
    # standalone transformer forward can: KWL flags, prompt/RoPE caching,
    # stage-2 PISA, and optionally TE NVFP4 video FFN.
    flags = {
        "SGLANG_LTX2_OFFICIAL_FA4_ATTENTION": "1",
        "SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN": "1",
        "SGLANG_LTX2_SHARE_GUIDANCE_PREFIX": "1",
        "SGLANG_LTX2_FUSED_QK_ROPE": "1",
        "SGLANG_LTX2_FUSED_RMS_ADALN": "1",
        "SGLANG_LTX2_FUSED_ADALN": "1",
        "SGLANG_LTX2_FUSED_QKNORM_ROPE": "1",
        "SGLANG_LTX2_FUSED_DUAL_MODULATE": "1",
        "SGLANG_LTX2_FUSED_ADA_VALUES_ALL": "1",
        "SGLANG_LTX2_FUSED_RESIDUAL_GATE": "1",
        "SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU": "1",
        "SGLANG_LTX2_COMPILE_GATE_TO_OUT": "1",
        "SGLANG_LTX2_FUSED_AUDIO_QKVG": "1",
        "SGLANG_ENABLE_FUSED_QKNORM_ROPE": "1",
        "SGLANG_LTX2_PREPROJECT_PROMPTS": "1",
        "SGLANG_LTX2_CACHE_ROPE_EMB": "1",
    }
    if te_nvfp4:
        flags.update(
            {
                "SGLANG_LTX2_TE_NVFP4_VIDEO_FFN": "1",
                "SGLANG_LTX2_TE_NVFP4_DISABLE_RHT": "1",
                "SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
                "SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
            }
        )
    for key, value in flags.items():
        os.environ.setdefault(key, value)


def _init_dist(master_port: int) -> None:
    from sglang.multimodal_gen.runtime.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(0)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        cfg_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        dp_size=1,
        distributed_init_method=f"tcp://127.0.0.1:{master_port}",
        dist_timeout=3600,
    )


def _build_server_args(model_path: str, master_port: int):
    from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
        LTX2PipelineConfig,
    )
    from sglang.multimodal_gen.runtime.server_args import Backend, ServerArgs

    return ServerArgs(
        model_path=model_path,
        pipeline_config=LTX2PipelineConfig(),
        backend=Backend.AUTO,
        pipeline_class_name="LTX2TwoStageHQPipeline",
        component_attention_backends="transformer=fa,transformer_2=piecewise_attn,text_encoder=torch_sdpa",
        attention_backend_config=(
            "piecewise_sparsity=0.9,"
            "piecewise_block_size=64,"
            "piecewise_only_video_self_attention=true,"
            "piecewise_stage1_schedule=false,"
            "piecewise_stage1_dense_steps=0,"
            "piecewise_stage1_start_sparsity=0.9,"
            "piecewise_stage1_end_sparsity=0.9,"
            "piecewise_dense_layers=none,"
            "piecewise_stage1_dense_layers=none,"
            "piecewise_stage2_dense_layers=none,"
            "piecewise_approx_remainder=true,"
            "piecewise_route_mode=score,"
            "piecewise_dense_fallback=fa"
        ),
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        dp_size=1,
        cfg_parallel_degree=1,
        master_port=master_port,
        performance_mode="speed",
        ltx2_two_stage_device_mode="resident",
        warmup=False,
        pin_cpu_memory=True,
    )


def _load_stage2_transformer(model_path: str, server_args):
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        PipelineComponentLoader,
    )

    transformer_path = str(Path(model_path) / "transformer")
    model, _ = PipelineComponentLoader.load_component(
        component_name="transformer_2",
        component_model_path=transformer_path,
        transformers_or_diffusers="diffusers",
        server_args=server_args,
    )
    return model.eval()


def _make_inputs(model, *, t_frames: int, height_tokens: int, width_tokens: int):
    arch = model.config.arch_config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch = 1
    video_tokens = int(t_frames) * int(height_tokens) * int(width_tokens)
    audio_tokens = 251
    text_tokens = 256
    # Stage-2 LTX-2.3 receives connector-projected prompt embeddings. The
    # checkpoint config still records raw Gemma packed dim as caption_channels
    # (3840), but caption_proj_before_connector=true means the DiT block prompt
    # attention/modulation path expects video/audio context dims.
    video_context_dim = int(getattr(arch, "cross_attention_dim", model.hidden_size))
    audio_context_dim = int(
        getattr(arch, "audio_cross_attention_dim", model.audio_hidden_size)
    )
    sigma = 0.421875

    return {
        "hidden_states": torch.randn(
            batch,
            video_tokens,
            int(arch.in_channels),
            device=device,
            dtype=dtype,
        ),
        "audio_hidden_states": torch.randn(
            batch,
            audio_tokens,
            int(arch.audio_in_channels),
            device=device,
            dtype=dtype,
        ),
        "encoder_hidden_states": torch.randn(
            batch,
            text_tokens,
            video_context_dim,
            device=device,
            dtype=dtype,
        ),
        "audio_encoder_hidden_states": torch.randn(
            batch,
            text_tokens,
            audio_context_dim,
            device=device,
            dtype=dtype,
        ),
        "timestep": torch.full((batch,), sigma, device=device, dtype=torch.float32),
        "audio_timestep": torch.full((batch,), sigma, device=device, dtype=torch.float32),
        "prompt_timestep": torch.full((batch,), sigma, device=device, dtype=torch.float32),
        "audio_prompt_timestep": torch.full(
            (batch,), sigma, device=device, dtype=torch.float32
        ),
        "encoder_attention_mask": None,
        "audio_encoder_attention_mask": None,
        "num_frames": int(t_frames),
        "height": int(height_tokens),
        "width": int(width_tokens),
        "fps": 24.0,
        "audio_num_frames": audio_tokens,
        "video_coords": None,
        "audio_coords": None,
        "video_self_attention_mask": None,
        "audio_self_attention_mask": None,
        "a2v_cross_attention_mask": None,
        "v2a_cross_attention_mask": None,
        "return_latents": False,
        "return_dict": False,
        "legacy_ltx23_one_stage_semantics": False,
    }


def _run_case(model, *, name: str, t_frames: int, height_tokens: int, width_tokens: int):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_alloc = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    inputs = _make_inputs(
        model, t_frames=t_frames, height_tokens=height_tokens, width_tokens=width_tokens
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    status = "ok"
    error = None
    try:
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs = model(**inputs)
        if isinstance(outputs, tuple):
            for item in outputs:
                if torch.is_tensor(item):
                    item.detach()
        elif torch.is_tensor(outputs):
            outputs.detach()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "name": name,
        "status": status,
        "error": error,
        "t_frames": t_frames,
        "height_tokens": height_tokens,
        "width_tokens": width_tokens,
        "video_tokens": t_frames * height_tokens * width_tokens,
        "elapsed_s": elapsed,
        "before_alloc_gib": before_alloc / 1024**3,
        "before_reserved_gib": before_reserved / 1024**3,
        "peak_alloc_gib": peak_alloc / 1024**3,
        "peak_reserved_gib": peak_reserved / 1024**3,
        "device_total_gib": torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493",
    )
    parser.add_argument("--master-port", type=int, default=30997)
    parser.add_argument("--baseline-t", type=int, default=31)
    parser.add_argument("--test-t", type=int, default=62)
    parser.add_argument("--height-tokens", type=int, default=34)
    parser.add_argument("--width-tokens", type=int, default=60)
    parser.add_argument("--te-nvfp4", action="store_true")
    parser.add_argument(
        "--output-json",
        default="outputs/ltx23-stage2-double-t-memory/probe.json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _set_default_cache_dirs(repo_root)
    _enable_current_kernel_env(te_nvfp4=args.te_nvfp4)
    _init_dist(args.master_port)

    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    server_args = _build_server_args(args.model_path, args.master_port)
    set_global_server_args(server_args)
    model = _load_stage2_transformer(args.model_path, server_args)

    results = {
        "model_path": args.model_path,
        "te_nvfp4": bool(args.te_nvfp4),
        "dtype": str(next(model.parameters()).dtype),
        "cases": [],
    }
    for name, t_frames in (("baseline_t", args.baseline_t), ("double_t", args.test_t)):
        result = _run_case(
            model,
            name=name,
            t_frames=t_frames,
            height_tokens=args.height_tokens,
            width_tokens=args.width_tokens,
        )
        results["cases"].append(result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if result["status"] != "ok":
            break

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}", flush=True)
    return 0 if all(c["status"] == "ok" for c in results["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
