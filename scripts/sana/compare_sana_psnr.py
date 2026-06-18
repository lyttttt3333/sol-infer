#!/usr/bin/env python3
"""PSNR of each config vs the dense baseline across the 16-prompt validation set.

Per-frame PSNR = 10*log10(255^2 / MSE_frame), averaged over frames (standard video
PSNR convention), then aggregated over the 16 prompts (mean / std / min).
Higher dB = closer to dense. Reads the validation clips produced by
sana_video_validation_run.py (<prefix>_val_<label>_pNN.mp4).

Usage: python scripts/sana/compare_sana_psnr.py dense ec010 ec020 ls28 ls20
       (first arg = baseline label; rest = candidates)
"""
from __future__ import annotations

import sys

import imageio.v2 as imageio
import numpy as np

OUT = "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/outputs"
PREFIX = "home_yitongl_sana_video_outputs_sglang_sana_480p_val"
N = 16


def load(label, i):
    p = f"{OUT}/{PREFIX}_{label}_p{i:02d}.mp4"
    try:
        r = imageio.get_reader(p)
        return np.stack([np.asarray(f, dtype=np.float32) for f in r])
    except Exception:
        return None


def psnr(cand, ref):
    T = min(len(cand), len(ref))
    c, r = cand[:T], ref[:T]
    mse = ((c - r) ** 2).reshape(T, -1).mean(axis=1)        # per-frame MSE
    mse = np.maximum(mse, 1e-8)
    return float((10.0 * np.log10(255.0 ** 2 / mse)).mean())  # mean per-frame PSNR (dB)


def main():
    args = sys.argv[1:] or ["dense", "ec010", "ec020", "ls28", "ls20"]
    baseline, cands = args[0], args[1:]
    print(f"PSNR vs {baseline}  |  {N}-prompt validation (dB; higher = closer)\n")

    print(f"{'prompt':>7s} " + " ".join(f"{c:>8s}" for c in cands))
    agg = {c: [] for c in cands}
    for i in range(N):
        ref = load(baseline, i)
        row = f"  p{i:02d}  "
        for c in cands:
            cv = load(c, i)
            if ref is None or cv is None:
                row += f" {'--':>8s}"
                continue
            v = psnr(cv, ref)
            agg[c].append(v)
            row += f" {v:>8.2f}"
        print(row)

    print(f"\n{'config':>8s} {'mean_dB':>8s} {'std':>6s} {'min_dB':>7s}  n")
    for c in cands:
        a = np.array(agg[c])
        if len(a):
            print(f"{c:>8s} {a.mean():>8.2f} {a.std():>6.2f} {a.min():>7.2f}  {len(a)}")
        else:
            print(f"{c:>8s}   (no clips found)")


if __name__ == "__main__":
    main()
