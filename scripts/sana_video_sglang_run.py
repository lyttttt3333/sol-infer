#!/usr/bin/env python3
"""Stage-validate + run the newly-ported SANA-Video model in the sglang
multimodal_gen runtime. Each stage prints a marker so the slurm log pinpoints
where any wiring bug is, before/after the heavy model load."""
from __future__ import annotations

import argparse
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
    ap.add_argument("--prompt", default=(
        "A cat and a dog baking a cake together in a kitchen. The cat is carefully "
        "measuring flour, while the dog is stirring the batter with a wooden spoon. "
        "The kitchen is cozy, with sunlight streaming through the window. motion score: 30."
    ))
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--guidance-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="/home/yitongl/sana_video/outputs/sglang_sana_480p")
    ap.add_argument("--label", default="")
    ap.add_argument("--compile", action="store_true", help="enable torch.compile (kernel-fusion toggle)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="disable warmup (warmup is ON by default for clean steady-state timing)")
    ap.add_argument("--teacache", type=float, default=0.0,
                    help="TeaCache rel-L1 threshold (0=off; higher=more skips / 跳步幅度)")
    ap.add_argument("--coeffs", default="",
                    help="TeaCache calibrated polynomial (highest-degree-first, comma-sep) "
                         "mapping input-relL1 -> output-relL1")
    ap.add_argument("--calib", default="",
                    help="TeaCache calibration: write per-step (branch input_relL1 output_relL1) "
                         "to this path; forces dense so the pairs describe true per-step drift")
    ap.add_argument("--nvfp4", action="store_true",
                    help="enable selective W4A4 NVFP4 on attn GEMMs")
    ap.add_argument("--ffn-lp", default="", choices=["", "fp4", "fp8"],
                    help="low-precision conv-FFN conv_inverted (fp4/fp8; the one high-N/K GEMM)")
    ap.add_argument("--nvfp4-layers", default="",
                    help="which blocks get NVFP4 (e.g. '0-9'); default all")
    ap.add_argument("--skip-from", type=int, default=0,
                    help="late-step skip: run steps [0,N) fully, reuse residual after (0=off); "
                         "composition-preserving fine-grained speed knob")
    ap.add_argument("--taylorseer", type=int, default=0,
                    help="TaylorSeer order/n_derivatives (0=off, 1=linear, 2=quadratic forecast)")
    ap.add_argument("--ts-interval", type=int, default=2, help="TaylorSeer compute interval")
    ap.add_argument("--ts-warmup", type=int, default=3, help="TaylorSeer warmup steps (always computed)")
    ap.add_argument("--easycache", type=float, default=0.0,
                    help="EasyCache reuse threshold (0=off; higher=more skips); "
                         "calibration-free adaptive step skipping")
    ap.add_argument("--ec-warmup", type=int, default=3, help="EasyCache warmup steps (always computed)")
    args = ap.parse_args()
    if args.label:
        args.output = f"{args.output}_{args.label}"
    # Set technique env BEFORE building the model (DiT reads them in __init__ /
    # post_load_weights); inherited by the local scheduler subprocess.
    import os as _os
    _os.environ["SGLANG_SANA_TEACACHE_THRESH"] = str(args.teacache)
    if args.coeffs:
        _os.environ["SGLANG_SANA_TEACACHE_COEFFS"] = args.coeffs
    if args.calib:
        _os.environ["SGLANG_SANA_TEACACHE_CALIB"] = args.calib
    if args.nvfp4:
        _os.environ["SGLANG_SANA_NVFP4"] = "1"
    if args.ffn_lp:
        _os.environ["SGLANG_SANA_FFN_LP"] = args.ffn_lp
    if args.nvfp4_layers:
        _os.environ["SGLANG_SANA_NVFP4_LAYERS"] = args.nvfp4_layers
    if args.skip_from:
        _os.environ["SGLANG_SANA_SKIP_FROM_STEP"] = str(args.skip_from)
    if args.taylorseer:
        _os.environ["SGLANG_SANA_TAYLORSEER_ORDER"] = str(args.taylorseer)
        _os.environ["SGLANG_SANA_TAYLORSEER_INTERVAL"] = str(args.ts_interval)
        _os.environ["SGLANG_SANA_TAYLORSEER_WARMUP"] = str(args.ts_warmup)
    if args.easycache:
        _os.environ["SGLANG_SANA_EASYCACHE_THRESH"] = str(args.easycache)
        _os.environ["SGLANG_SANA_EASYCACHE_WARMUP"] = str(args.ec_warmup)

    print(f"==== STAGE 1: import new SANA-Video modules ====", flush=True)
    try:
        from sglang.multimodal_gen.runtime.models.dits.sana_video import (  # noqa: F401
            SanaVideoTransformer3DModel,
        )
        from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (  # noqa: F401
            SanaVideoPipelineConfig,
        )
        from sglang.multimodal_gen.configs.sample.sana_video import (  # noqa: F401
            SanaVideoSamplingParams,
        )
        from sglang.multimodal_gen.runtime.pipelines.sana_video import (  # noqa: F401
            SanaVideoPipeline,
        )
        print("IMPORT_OK", flush=True)
    except Exception:
        traceback.print_exc()
        print("IMPORT_FAIL", flush=True)
        sys.exit(1)

    print(f"==== STAGE 2: resolve local snapshot + pipeline registry ====", flush=True)
    import glob
    import os

    def _resolve_local(model):
        if os.path.isdir(model):
            return model
        cache = os.environ.get(
            "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
        )
        safe = "models--" + model.replace("/", "--")
        snaps = sorted(glob.glob(os.path.join(cache, safe, "snapshots", "*")))
        return snaps[-1] if snaps else model

    model_path = _resolve_local(args.model)
    print("model_path:", model_path, flush=True)
    try:
        from sglang.multimodal_gen.registry import (
            _PIPELINE_REGISTRY,
            _discover_and_register_pipelines,
        )

        _discover_and_register_pipelines()
        print(
            "SanaVideoPipeline registered:",
            "SanaVideoPipeline" in _PIPELINE_REGISTRY,
            flush=True,
        )
        print(
            "sana pipelines:",
            [k for k in _PIPELINE_REGISTRY if "ana" in k.lower()],
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        print("REGISTRY_WARN", flush=True)

    print(f"==== STAGE 3: build DiffGenerator ====", flush=True)
    try:
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        gen = DiffGenerator.from_pretrained(
            model_path=model_path,
            pipeline_class_name="SanaVideoPipeline",
            num_gpus=1,
            dit_cpu_offload=False,
            enable_torch_compile=args.compile,
            warmup=not args.no_warmup,
        )
        print(f"BUILD_OK (compile={args.compile}, warmup={not args.no_warmup})", flush=True)
    except Exception:
        traceback.print_exc()
        print("BUILD_FAIL", flush=True)
        sys.exit(3)

    print(f"==== STAGE 4: generate ====", flush=True)
    try:
        t = time.time()
        res = gen.generate(
            sampling_params_kwargs=dict(
                prompt=args.prompt,
                num_frames=args.frames,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                output_file_name=args.output,
                save_output=True,
                fps=16,
            )
        )
        print(f"GENERATE_OK in {time.time() - t:.1f}s: {res}", flush=True)
        print("RUN_COMPLETE_MARKER", flush=True)
    except Exception:
        traceback.print_exc()
        print("GENERATE_FAIL", flush=True)
        sys.exit(4)


if __name__ == "__main__":
    main()
