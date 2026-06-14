#!/usr/bin/env python3
"""Run a fixed 16-prompt validation set through SANA-Video for a single config
(dense / easycache / lateskip / ...). Builds the DiffGenerator ONCE and loops all
prompts so the model load + warmup is amortized. Per-prompt output is named
<base>_<label>_pNN so dense vs cache clips line up for compare_sana_validation.py.

Cache state resets per generation automatically (timestep jump-up), so the 16
prompts are independent. Seed is base_seed+i (matched across configs per prompt)."""
from __future__ import annotations

import argparse
import os as _os
import sys
import time
import traceback

# 16 diverse validation prompts (animals / humans+action / nature / urban /
# objects / weather), SANA-Video "motion score" convention appended.
PROMPTS = [
    "A cat and a dog baking a cake together in a cozy kitchen, sunlight through the window.",
    "A majestic eagle soaring over snow-capped mountains at sunrise.",
    "A bustling Tokyo street at night with glowing neon signs and people walking in the rain.",
    "A close-up of a blooming red rose with dew drops, gently swaying in the breeze.",
    "A surfer riding a large ocean wave, water spraying into the air.",
    "A steam locomotive traveling through a lush green valley, smoke billowing.",
    "A child flying a colorful kite on a sunny sandy beach.",
    "A chef tossing vegetables in a flaming wok in a professional kitchen.",
    "A pod of dolphins leaping out of clear blue ocean water.",
    "Clouds drifting over a golden wheat field at sunset, timelapse.",
    "A humanoid robot walking through a futuristic city with flying cars.",
    "A waterfall cascading down mossy rocks in a tropical rainforest.",
    "A hot air balloon drifting slowly over a patchwork of green farmland.",
    "A giant panda eating bamboo in a misty bamboo forest.",
    "A vintage convertible car driving along a coastal highway at sunset.",
    "A figure skater spinning gracefully on a glistening ice rink.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
    ap.add_argument("--label", required=True, help="config label, e.g. dense / ec010 / ls20")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--guidance-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0, help="base seed; prompt i uses seed+i")
    ap.add_argument("--motion-score", type=int, default=30)
    ap.add_argument("--nprompts", type=int, default=16, help="run only the first N prompts")
    ap.add_argument("--output-base", default="/home/yitongl/sana_video/outputs/sglang_sana_480p_val")
    # cache toggles (same env contract as sana_video_sglang_run.py; OFF==dense)
    ap.add_argument("--teacache", type=float, default=0.0)
    ap.add_argument("--coeffs", default="")
    ap.add_argument("--skip-from", type=int, default=0)
    ap.add_argument("--taylorseer", type=int, default=0)
    ap.add_argument("--ts-interval", type=int, default=2)
    ap.add_argument("--ts-warmup", type=int, default=3)
    ap.add_argument("--easycache", type=float, default=0.0)
    ap.add_argument("--ec-warmup", type=int, default=3)
    ap.add_argument("--ffn-lp", default="", choices=["", "fp4", "fp8"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    # technique env BEFORE building the model (DiT reads it in __init__)
    _os.environ["SGLANG_SANA_TEACACHE_THRESH"] = str(args.teacache)
    if args.coeffs:
        _os.environ["SGLANG_SANA_TEACACHE_COEFFS"] = args.coeffs
    if args.skip_from:
        _os.environ["SGLANG_SANA_SKIP_FROM_STEP"] = str(args.skip_from)
    if args.taylorseer:
        _os.environ["SGLANG_SANA_TAYLORSEER_ORDER"] = str(args.taylorseer)
        _os.environ["SGLANG_SANA_TAYLORSEER_INTERVAL"] = str(args.ts_interval)
        _os.environ["SGLANG_SANA_TAYLORSEER_WARMUP"] = str(args.ts_warmup)
    if args.easycache:
        _os.environ["SGLANG_SANA_EASYCACHE_THRESH"] = str(args.easycache)
        _os.environ["SGLANG_SANA_EASYCACHE_WARMUP"] = str(args.ec_warmup)
    if args.ffn_lp:
        _os.environ["SGLANG_SANA_FFN_LP"] = args.ffn_lp

    import glob

    def _resolve_local(model):
        if _os.path.isdir(model):
            return model
        cache = _os.environ.get("HF_HUB_CACHE", _os.path.expanduser("~/.cache/huggingface/hub"))
        safe = "models--" + model.replace("/", "--")
        snaps = sorted(glob.glob(_os.path.join(cache, safe, "snapshots", "*")))
        return snaps[-1] if snaps else model

    print("==== build DiffGenerator ====", flush=True)
    try:
        from sglang.multimodal_gen.runtime.models.dits.sana_video import (  # noqa: F401
            SanaVideoTransformer3DModel,
        )
        from sglang.multimodal_gen.runtime.pipelines.sana_video import (  # noqa: F401
            SanaVideoPipeline,
        )
        from sglang.multimodal_gen.registry import _discover_and_register_pipelines
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        _discover_and_register_pipelines()
        gen = DiffGenerator.from_pretrained(
            model_path=_resolve_local(args.model),
            pipeline_class_name="SanaVideoPipeline",
            num_gpus=1,
            dit_cpu_offload=False,
            enable_torch_compile=args.compile,
            warmup=not args.no_warmup,
        )
        print(f"BUILD_OK label={args.label} compile={args.compile}", flush=True)
    except Exception:
        traceback.print_exc()
        print("BUILD_FAIL", flush=True)
        sys.exit(3)

    prompts = PROMPTS[: max(1, args.nprompts)]
    print(f"==== generate {len(prompts)} validation prompts ====", flush=True)
    times = []
    for i, p in enumerate(prompts):
        prompt = f"{p} motion score: {args.motion_score}."
        out = f"{args.output_base}_{args.label}_p{i:02d}"
        try:
            t = time.time()
            gen.generate(
                sampling_params_kwargs=dict(
                    prompt=prompt,
                    num_frames=args.frames,
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + i,
                    output_file_name=out,
                    save_output=True,
                    fps=16,
                )
            )
            dt = time.time() - t
            times.append(dt)
            print(f"PROMPT_OK p{i:02d} {dt:.1f}s :: {p[:50]}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"PROMPT_FAIL p{i:02d}", flush=True)
    if times:
        mean = sum(times) / len(times)
        # steady-state mean excludes p00 (absorbs residual JIT even with warmup)
        ss = times[1:] or times
        print(f"SUMMARY label={args.label} n={len(times)} mean={mean:.1f}s "
              f"steady_mean={sum(ss)/len(ss):.1f}s", flush=True)
    print("VAL_RUN_COMPLETE_MARKER", flush=True)


if __name__ == "__main__":
    main()
