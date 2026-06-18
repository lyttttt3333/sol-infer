#!/usr/bin/env python3
"""Self-check SANA-Video acceleration quality vs the dense baseline.

All clips share seed/prompt, so a quality-preserving config should be nearly
identical to dense. Numpy-only metrics (no GPU / no sglang import) + extracts a
mid-frame PNG per config for visual inspection:
  MAE/255      mean abs pixel diff vs dense (lower = closer)
  corr         per-frame pixel correlation vs dense (1.0 = same structure/semantics)
  sharp_ratio  gradient-energy ratio vs dense (1.0 = same visual detail; <1 = blurrier)
"""
from __future__ import annotations

import os
import sys

import imageio.v2 as imageio
import numpy as np

BASE = "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/outputs"
PREFIX = "home_yitongl_sana_video_outputs_sglang_sana_480p"
FRAME_DIR = "/home/yitongl/sana_video/outputs/sglang/quality_frames"
FIDX = 40  # mid frame


def load(label: str):
    p = f"{BASE}/{PREFIX}_{label}.mp4"
    if not os.path.exists(p):
        return None
    r = imageio.get_reader(p)
    return np.stack([np.asarray(f, dtype=np.float32) for f in r])  # [T,H,W,3]


def sharp(x: np.ndarray) -> float:
    return float(np.abs(np.diff(x, axis=2)).mean() + np.abs(np.diff(x, axis=1)).mean())


def compare(cand: np.ndarray, ref: np.ndarray):
    T = min(len(cand), len(ref))
    c, r = cand[:T], ref[:T]
    mae = float(np.abs(c - r).mean())
    cf, rf = c.reshape(T, -1), r.reshape(T, -1)
    corr = float(np.mean([np.corrcoef(cf[i], rf[i])[0, 1] for i in range(T)]))
    return mae, corr, sharp(c) / max(sharp(r), 1e-6)


def main():
    labels = sys.argv[1:] or [
        "fine_tc002", "fine_tc004", "fine_tc006", "fine_tc008",
        "tc001", "tc002", "tc003", "tc005",
    ]
    os.makedirs(FRAME_DIR, exist_ok=True)
    ref = load("dense_warm")
    if ref is None:
        print("ERROR: dense baseline (dense_warm) missing")
        sys.exit(1)
    fi = min(FIDX, len(ref) - 1)
    imageio.imwrite(f"{FRAME_DIR}/f{fi}_00dense.png", ref[fi].astype(np.uint8))
    print(f"{'config':14s} {'MAE/255':>8s} {'corr':>7s} {'sharp':>7s}   verdict")
    print(f"{'dense (ref)':14s} {0.0:8.2f} {1.000:7.3f} {1.000:7.3f}   baseline")
    for lb in labels:
        c = load(lb)
        if c is None:
            print(f"{lb:14s} (missing)")
            continue
        mae, corr, sr = compare(c, ref)
        imageio.imwrite(f"{FRAME_DIR}/f{fi}_{lb}.png", c[min(fi, len(c) - 1)].astype(np.uint8))
        verdict = (
            "near-identical" if corr >= 0.985 and mae <= 4
            else "close" if corr >= 0.96 and mae <= 9
            else "drifting" if corr >= 0.90
            else "DIFFERENT"
        )
        print(f"{lb:14s} {mae:8.2f} {corr:7.3f} {sr:7.3f}   {verdict}")
    print(f"\nframes -> {FRAME_DIR}/f{fi}_*.png")


if __name__ == "__main__":
    main()
