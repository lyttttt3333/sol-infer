#!/usr/bin/env python3
"""Stage-level time profile of SANA-Video generation.

Attributes wall-clock (with cuda.synchronize on both sides) to each high-level
pipeline component by wrapping the bound methods the pipeline calls internally:
  encode_prompt            -> text_encode (Gemma2)
  transformer.forward      -> DiT denoise (summed over all steps; CFG may 2x)
  vae.decode               -> VAE decode (tiled if --vae-tiling)
  video_processor.postprocess_video -> tensor->PIL/np postprocess (CPU)
"other" = total infer - the above (scheduler.step, guidance combine, latent
prep/denorm, misc). A warmup run is excluded so step-0 compile/alloc doesn't
pollute the steady-state numbers.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from diffusers import SanaVideoPipeline
from diffusers.utils import export_to_video

PROMPT = ("A cat and a dog baking a cake together in a kitchen. The cat is carefully "
          "measuring flour, while the dog is stirring the batter with a wooden spoon. "
          "The kitchen is cozy, with sunlight streaming through the window.")
NEG = ("A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden "
       "disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, "
       "inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, "
       "creating a disorienting visual experience.")


class StageTimer:
    def __init__(self):
        self.t = {}
        self.n = {}

    def wrap(self, obj, attr, name):
        if not hasattr(obj, attr):
            print(f"[prof] WARN: cannot wrap {name}: {attr} not found", flush=True)
            return
        orig = getattr(obj, attr)

        def wrapped(*a, **k):
            torch.cuda.synchronize()
            s = time.perf_counter()
            r = orig(*a, **k)
            torch.cuda.synchronize()
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - s)
            self.n[name] = self.n.get(name, 0) + 1
            return r

        setattr(obj, attr, wrapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vae-tiling", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    tag = args.label or args.model.split("/")[-1]
    prompt = PROMPT + " motion score: 30."

    def call(steps):
        return pipe(
            prompt=prompt, negative_prompt=NEG, height=args.height, width=args.width,
            frames=args.frames, guidance_scale=args.guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).frames[0]

    print(f"\n===== PROFILE {tag}  {args.width}x{args.height} {args.frames}f {args.steps}steps "
          f"vae_tiling={args.vae_tiling} =====", flush=True)

    t0 = time.time()
    pipe = SanaVideoPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.to("cuda")
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    torch.cuda.synchronize()
    load_s = time.time() - t0

    # warmup (excluded): primes cudnn autotune / allocator at this resolution
    _ = call(2)
    torch.cuda.synchronize()

    prof = StageTimer()
    prof.wrap(pipe, "encode_prompt", "text_encode")
    prof.wrap(pipe.transformer, "forward", "transformer")
    prof.wrap(pipe.vae, "decode", "vae_decode")
    prof.wrap(pipe.video_processor, "postprocess_video", "postprocess")

    torch.cuda.synchronize()
    t1 = time.time()
    video = call(args.steps)
    torch.cuda.synchronize()
    infer_s = time.time() - t1

    out = f"/home/yitongl/sana_video/outputs/_profile_{tag}.mp4"
    t2 = time.time()
    export_to_video(video, out, fps=16)
    export_s = time.time() - t2

    te = prof.t.get("text_encode", 0.0)
    tf = prof.t.get("transformer", 0.0)
    vd = prof.t.get("vae_decode", 0.0)
    pp = prof.t.get("postprocess", 0.0)
    other = max(infer_s - (te + tf + vd + pp), 0.0)

    rows = [
        ("text_encode (Gemma2)", te, prof.n.get("text_encode", 0)),
        ("transformer denoise (DiT)", tf, prof.n.get("transformer", 0)),
        ("vae_decode" + (" (tiled)" if args.vae_tiling else ""), vd, prof.n.get("vae_decode", 0)),
        ("postprocess (->PIL, CPU)", pp, prof.n.get("postprocess", 0)),
        ("other (sched/guidance/denorm)", other, None),
    ]
    print(f"[load] {load_s:6.2f}s   [INFER total] {infer_s:6.2f}s   [export mp4] {export_s:5.2f}s", flush=True)
    print(f"{'component':36s} {'sec':>8s} {'% infer':>9s} {'calls':>7s}", flush=True)
    for name, sec, cnt in rows:
        c = "" if cnt is None else str(cnt)
        print(f"{name:36s} {sec:8.3f} {100 * sec / infer_s:8.1f}% {c:>7}", flush=True)
    nfwd = prof.n.get("transformer", 0)
    if nfwd:
        print(f"  -> DiT per-forward {1000 * tf / nfwd:6.1f} ms over {nfwd} calls "
              f"({nfwd / args.steps:.0f}x/step => CFG {'batched' if nfwd == args.steps else 'sequential'})", flush=True)
    print("PROFILE_JSON " + json.dumps({
        "tag": tag, "wxh": f"{args.width}x{args.height}", "frames": args.frames, "steps": args.steps,
        "vae_tiling": args.vae_tiling, "load_s": round(load_s, 3), "infer_s": round(infer_s, 3),
        "export_s": round(export_s, 3), "text_encode_s": round(te, 3),
        "transformer_s": round(tf, 3), "transformer_calls": nfwd,
        "vae_decode_s": round(vd, 3), "postprocess_s": round(pp, 3), "other_s": round(other, 3),
    }), flush=True)
    print("PROFILE_DONE", flush=True)


if __name__ == "__main__":
    main()
