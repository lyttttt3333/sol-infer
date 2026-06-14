#!/usr/bin/env python3
"""Aggregate SANA-Video quality across the 16-prompt validation set: each
candidate config vs the dense baseline, per-prompt + mean/std/min.

Clips are produced by sana_video_validation_run.py as
  <prefix>_<label>_pNN.mp4
all sharing seed+prompt per index, so a quality-preserving config tracks dense.

Usage: python scripts/compare_sana_validation.py dense ec010 ec020 ls20
       (first arg = baseline label; rest = candidates)
"""
from __future__ import annotations

import sys

import imageio.v2 as imageio
import numpy as np

BASE = "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/outputs"
PREFIX = "home_yitongl_sana_video_outputs_sglang_sana_480p_val"
NPROMPTS = 16


def load(label, i):
    p = f"{BASE}/{PREFIX}_{label}_p{i:02d}.mp4"
    try:
        r = imageio.get_reader(p)
        return np.stack([np.asarray(f, dtype=np.float32) for f in r])
    except Exception:
        return None


def sharp(x):
    return float(np.abs(np.diff(x, axis=2)).mean() + np.abs(np.diff(x, axis=1)).mean())


def metrics(cand, ref):
    T = min(len(cand), len(ref))
    c, r = cand[:T], ref[:T]
    mae = float(np.abs(c - r).mean())
    cf, rf = c.reshape(T, -1), r.reshape(T, -1)
    corr = float(np.mean([np.corrcoef(cf[i], rf[i])[0, 1] for i in range(T)]))
    return mae, corr, sharp(c) / max(sharp(r), 1e-6)


def main():
    args = sys.argv[1:] or ["dense", "ec010", "ec020", "ls20"]
    baseline, cands = args[0], args[1:]
    print(f"baseline = {baseline}  |  {NPROMPTS}-prompt validation\n")

    # per-prompt corr table
    hdr = f"{'prompt':>7s} " + " ".join(f"{c:>9s}" for c in cands)
    print(hdr)
    agg = {c: {"corr": [], "mae": [], "sharp": []} for c in cands}
    for i in range(NPROMPTS):
        ref = load(baseline, i)
        row = f"  p{i:02d}  "
        for c in cands:
            cv = load(c, i)
            if ref is None or cv is None:
                row += f" {'--':>9s}"
                continue
            mae, corr, sh = metrics(cv, ref)
            agg[c]["corr"].append(corr)
            agg[c]["mae"].append(mae)
            agg[c]["sharp"].append(sh)
            row += f" {corr:>9.3f}"
        print(row)

    print(f"\n{'config':>9s} {'mean_corr':>9s} {'std':>6s} {'min':>6s} {'mean_MAE':>8s} {'mean_sharp':>10s}  n")
    for c in cands:
        cr = agg[c]["corr"]
        if not cr:
            print(f"{c:>9s}   (no clips found)")
            continue
        arr = np.array(cr)
        print(f"{c:>9s} {arr.mean():>9.3f} {arr.std():>6.3f} {arr.min():>6.3f} "
              f"{np.mean(agg[c]['mae']):>8.2f} {np.mean(agg[c]['sharp']):>10.3f}  {len(cr)}")


if __name__ == "__main__":
    main()
