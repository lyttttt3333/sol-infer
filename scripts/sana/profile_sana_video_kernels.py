#!/usr/bin/env python3
"""Operator/kernel-level profile of SANA-Video via torch.profiler.

Two scoped captures per model (warmup excluded):
  (1) DiT denoise loop -- a profiler with a schedule, stepped by the pipeline's
      callback_on_step_end, capturing ~5 steady-state steps.
  (2) VAE decode       -- the single decode call wrapped in its own profiler.
Prints key_averages() sorted by self CUDA time (the per-kernel %), and writes a
Chrome trace per scope for drill-down (chrome://tracing or perfetto.dev).
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from diffusers import SanaVideoPipeline

PROMPT = ("A cat and a dog baking a cake together in a kitchen. The cat is carefully "
          "measuring flour, while the dog is stirring the batter with a wooden spoon. "
          "The kitchen is cozy, with sunlight streaming through the window.")
NEG = ("A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden "
       "disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, "
       "inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, "
       "creating a disorienting visual experience.")
PROF_DIR = "/home/yitongl/sana_video/profiles"


def render(prof, row_limit=30):
    """key_averages table, robust to the self_cuda/self_device sort-key rename."""
    ka = prof.key_averages()
    for key in ("self_cuda_time_total", "self_device_time_total"):
        try:
            return ka.table(sort_by=key, row_limit=row_limit)
        except Exception:
            continue
    return ka.table(row_limit=row_limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--guidance-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vae-tiling", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.makedirs(PROF_DIR, exist_ok=True)
    tag = args.label or args.model.split("/")[-1]
    prompt = PROMPT + " motion score: 30."
    print(f"\n===== KERNEL PROFILE {tag}  {args.width}x{args.height} {args.frames}f "
          f"vae_tiling={args.vae_tiling} =====", flush=True)

    pipe = SanaVideoPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.to("cuda")
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    torch.cuda.synchronize()

    def call(steps, **extra):
        return pipe(
            prompt=prompt, negative_prompt=NEG, height=args.height, width=args.width,
            frames=args.frames, guidance_scale=args.guidance_scale, num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed), **extra,
        ).frames[0]

    # warmup (excluded)
    _ = call(2)
    torch.cuda.synchronize()

    ACT = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # ---- (1) DiT denoise loop: capture ~5 steady-state steps ----
    sch = schedule(wait=2, warmup=3, active=5, repeat=1)
    dit_prof = profile(activities=ACT, schedule=sch, record_shapes=False, with_stack=False)
    dit_prof.start()

    def cb(p, step, t, kw):
        dit_prof.step()
        return kw

    _ = call(12, callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    dit_prof.stop()
    print(f"\n----- [{tag}] DiT DENOISE LOOP — top kernels by self CUDA time -----", flush=True)
    print(render(dit_prof, 30), flush=True)
    try:
        dit_prof.export_chrome_trace(f"{PROF_DIR}/sana_dit_{tag}.json")
        print(f"[trace] {PROF_DIR}/sana_dit_{tag}.json", flush=True)
    except Exception as e:
        print(f"[trace] DiT export failed: {e!r}", flush=True)

    # ---- (2) VAE decode: wrap the single decode call in its own profiler ----
    cap = {}
    orig_decode = pipe.vae.decode

    def decode_prof(*a, **k):
        with profile(activities=ACT, record_shapes=False) as p:
            r = orig_decode(*a, **k)
            torch.cuda.synchronize()
        cap["table"] = render(p, 25)
        try:
            p.export_chrome_trace(f"{PROF_DIR}/sana_vae_{tag}.json")
            cap["trace"] = f"{PROF_DIR}/sana_vae_{tag}.json"
        except Exception as e:
            cap["trace"] = f"(export failed: {e!r})"
        return r

    pipe.vae.decode = decode_prof
    _ = call(4)  # decode runs once -> profiled
    pipe.vae.decode = orig_decode
    print(f"\n----- [{tag}] VAE DECODE — top kernels by self CUDA time -----", flush=True)
    print(cap.get("table", "(decode not captured)"), flush=True)
    print(f"[trace] {cap.get('trace', 'n/a')}", flush=True)

    print(f"KERNEL_PROFILE_DONE {tag}", flush=True)


if __name__ == "__main__":
    main()
