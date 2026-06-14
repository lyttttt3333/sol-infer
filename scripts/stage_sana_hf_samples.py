#!/usr/bin/env python3
"""Stage SANA-Video acceleration samples with descriptive, browsable names +
a MANIFEST for HF upload. Speedup = end-to-end wall-clock vs dense (34.4s);
corr = per-frame pixel correlation to the dense baseline (1.0 = identical)."""
from __future__ import annotations

import os
import shutil

SRC = "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/outputs"
PRE = "home_yitongl_sana_video_outputs_sglang_sana_480p"
DST = "/home/yitongl/sana_video/hf_samples"

DENSE_S = 34.4  # dense_warm end-to-end GENERATE_OK (1.00x reference)

# (label, method, family, time_s, corr, note)
ROWS = [
    ("dense_warm",  "Dense baseline (50 steps)",           "baseline",   34.4, 1.000, "reference"),
    ("compile_warm","Fusion (torch.compile)",              "fusion",     24.2, 1.000, "LOSSLESS; ~1.61x on DiT, est ~1.42x e2e"),
    # --- late-skip: THE WINNER (fixed schedule + shared common-mode residual) ---
    ("ls3_28", "Late-skip N=28 (skip last 22)",  "lateskip", 22.6, 0.966, "near-lossless"),
    ("ls3_24", "Late-skip N=24 (skip last 26)",  "lateskip", 19.9, 0.953, "*** sweet spot ***"),
    ("ls3_20", "Late-skip N=20 (skip last 30)",  "lateskip", 18.3, 0.936, "great"),
    ("ls3_16", "Late-skip N=16 (skip last 34)",  "lateskip", 15.6, 0.912, "aggressive"),
    ("ls3_12", "Late-skip N=12 (skip last 38)",  "lateskip", 13.5, 0.871, "very aggressive"),
    # --- EasyCache (calibration-free adaptive): Pareto-dominated by late-skip ---
    ("ec_010", "EasyCache thr=0.1",  "easycache", 26.5, 0.969, "barely skips (~1.0x)"),
    ("ec_020", "EasyCache thr=0.2",  "easycache", 21.1, 0.887, "adaptive"),
    ("ec_040", "EasyCache thr=0.4",  "easycache", 18.1, 0.683, "skips early steps -> degrades"),
    ("ec_080", "EasyCache thr=0.8",  "easycache", 15.0, 0.617, "skips early steps -> degrades"),
    # --- TaylorSeer (forecast): loses to constant-hold ---
    ("ts_o1_i2",   "TaylorSeer uniform order1 interval2", "taylorseer", 21.8, 0.618, "uniform skip; overshoots"),
    ("ts_o1_i4",   "TaylorSeer uniform order1 interval4", "taylorseer", 15.4, 0.232, "uniform skip; collapses"),
    ("tsL_w45_o1", "TaylorSeer late-forecast w45 order1", "taylorseer", 31.6, 0.946, "late-only; ~ties hold but slow"),
    ("tsL_w40_o2", "TaylorSeer late-forecast w40 order2", "taylorseer", 29.1, 0.640, "order2 overshoots badly"),
    # --- TeaCache (uncalibrated; premise fails: input->drift corr 0.13) ---
    ("tc001", "TeaCache thr=0.01 (uncalibrated)", "teacache", 19.5, 0.532, "signal not predictive"),
    # --- NVFP4 W4A4: net-negative, excluded ---
    ("nvfp4_all", "NVFP4 W4A4 (all attn)", "nvfp4", 37.1, None, "0.93x SLOWER -> excluded"),
]


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    lines = [
        "# SANA-Video (480p, 832x480x81f, 50 steps) — acceleration ablation samples",
        "",
        "All clips share prompt + seed, so a quality-preserving method should look",
        "near-identical to `00_dense`. **Speedup = end-to-end wall-clock vs dense (34.4s).**",
        "**corr = per-frame pixel correlation to dense (1.0 = identical structure/semantics).**",
        "",
        "| # | file | method | speedup | corr | note |",
        "|---|------|--------|--------:|-----:|------|",
    ]
    n = 0
    for label, method, fam, t, corr, note in ROWS:
        src = f"{SRC}/{PRE}_{label}.mp4"
        if not os.path.exists(src):
            print(f"SKIP missing: {label}")
            continue
        spd = DENSE_S / t
        spd_s = f"{spd:.2f}x" if t else "-"
        corr_s = f"{corr:.3f}" if corr is not None else "n/a"
        cs = f"corr{corr:.3f}" if corr is not None else "exclude"
        out = f"{n:02d}_{fam}_{label}_{spd:.2f}x_{cs}.mp4"
        shutil.copy2(src, f"{DST}/{out}")
        lines.append(f"| {n:02d} | `{out}` | {method} | {spd_s} | {corr_s} | {note} |")
        n += 1
    lines += [
        "",
        "## Verdict",
        "",
        "- **Late-skip wins** (fixed schedule + shared common-mode residual): corr 0.953 @ 1.73x, 0.912 @ 2.21x.",
        "- **All predictive/adaptive caches fail** — TeaCache (input->drift corr 0.13), FBCache (0.32),",
        "  TaylorSeer (forecast overshoots), EasyCache (Pareto-dominated). No cheap signal predicts",
        "  SANA-Video's per-step drift, so the skip schedule can't be learned; only the structural",
        "  prior \"never skip the early composition-forming steps\" works = late-skip.",
        "- **Fusion** (torch.compile) is lossless ~1.4x e2e; **NVFP4** is net-negative (excluded).",
        "- Key fix: under unbatched CFG, reuse ONE shared residual (guidance term cancels) — never",
        "  per-branch (re-injects 6x-amplified stale guidance -> artifacts).",
    ]
    with open(f"{DST}/MANIFEST.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"staged {n} clips + MANIFEST.md -> {DST}")


if __name__ == "__main__":
    main()
