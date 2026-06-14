#!/usr/bin/env python3
"""Stage the 16-prompt SANA-Video validation set for HF upload: compute each
config's per-prompt corr vs dense, copy clips into a flat browsable layout
(pNN_<k><config>_corrX.mp4 — sorts by prompt, configs adjacent), and write a
MANIFEST with the prompts, per-prompt corr table, aggregate, and verdict."""
from __future__ import annotations

import os
import shutil

import imageio.v2 as imageio
import numpy as np

OUT = "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/outputs"
PRE = "home_yitongl_sana_video_outputs_sglang_sana_480p_val"
DST = "/home/yitongl/sana_video/hf_validation"
N = 16

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

# (order, label, description, steady_time_s); dense=baseline
CONFIGS = [
    (0, "dense", "Dense baseline (50 steps)",        29.4),
    (1, "ec010", "EasyCache thr=0.1",                19.6),
    (2, "ec020", "EasyCache thr=0.2",                14.2),
    (3, "ls28",  "Late-skip N=28",                   17.6),
    (4, "ls20",  "Late-skip N=20",                   13.2),
]
DENSE_S = 29.4


def load(label, i):
    p = f"{OUT}/{PRE}_{label}_p{i:02d}.mp4"
    try:
        r = imageio.get_reader(p)
        return np.stack([np.asarray(f, dtype=np.float32) for f in r]), p
    except Exception:
        return None, p


def sharp(x):
    return float(np.abs(np.diff(x, axis=2)).mean() + np.abs(np.diff(x, axis=1)).mean())


def corr_sharp(cand, ref):
    T = min(len(cand), len(ref))
    c, r = cand[:T], ref[:T]
    cf, rf = c.reshape(T, -1), r.reshape(T, -1)
    corr = float(np.mean([np.corrcoef(cf[i], rf[i])[0, 1] for i in range(T)]))
    return corr, sharp(c) / max(sharp(r), 1e-6)


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    agg = {lbl: {"corr": [], "sharp": []} for _, lbl, _, _ in CONFIGS if lbl != "dense"}
    percorr = {lbl: [None] * N for _, lbl, _, _ in CONFIGS}

    for i in range(N):
        ref, _ = load("dense", i)
        for order, lbl, _, _ in CONFIGS:
            arr, src = load(lbl, i)
            if arr is None:
                print(f"MISSING {lbl} p{i:02d}")
                continue
            if lbl == "dense":
                tag = "corr1.000"
                percorr[lbl][i] = 1.0
            elif ref is not None:
                cr, sh = corr_sharp(arr, ref)
                tag = f"corr{cr:.3f}"
                percorr[lbl][i] = cr
                agg[lbl]["corr"].append(cr)
                agg[lbl]["sharp"].append(sh)
            else:
                tag = "corrNA"
            shutil.copy2(src, f"{DST}/p{i:02d}_{order}{lbl}_{tag}.mp4")

    # MANIFEST
    L = ["# SANA-Video 16-prompt validation — EasyCache vs dense baseline (+ late-skip)",
         "",
         "480p, 832x480x81f, 50 steps. Files: `pNN_<k><config>_corrX.mp4` (sorts by prompt;",
         "configs adjacent). corr = per-frame pixel correlation to dense (1.0 = identical).",
         "Speedup = end-to-end wall-clock vs dense (29.4s steady-state).", "",
         "## Configs", "", "| k | config | method | speedup |", "|---|--------|--------|--------:|"]
    for order, lbl, desc, t in CONFIGS:
        L.append(f"| {order} | `{lbl}` | {desc} | {DENSE_S/t:.2f}x |")
    L += ["", "## Per-prompt corr vs dense", "",
          "| prompt | " + " | ".join(l for _, l, _, _ in CONFIGS if l != "dense") + " | text |",
          "|---|" + "---|" * (len(CONFIGS) - 1) + "---|"]
    for i in range(N):
        cells = " | ".join(f"{percorr[l][i]:.3f}" if percorr[l][i] is not None else "--"
                           for _, l, _, _ in CONFIGS if l != "dense")
        L.append(f"| p{i:02d} | {cells} | {PROMPTS[i][:48]} |")
    L += ["", "## Aggregate (16 prompts)", "",
          "| config | mean corr | std | min | mean sharp |", "|---|---:|---:|---:|---:|"]
    for _, lbl, _, _ in CONFIGS:
        if lbl == "dense":
            continue
        a = np.array(agg[lbl]["corr"])
        if len(a):
            L.append(f"| `{lbl}` | {a.mean():.3f} | {a.std():.3f} | {a.min():.3f} | "
                     f"{np.mean(agg[lbl]['sharp']):.3f} |")
    L += ["", "## Verdict", "",
          "- **Late-skip Pareto-dominates EasyCache.** ls20 (2.23x, mean 0.921) beats ec020",
          "  (2.07x, 0.867) on speed AND corr AND sharpness; ls28 (1.67x) matches/beats ec010",
          "  (1.50x, 0.950) at similar speed.",
          "- **EasyCache is less consistent** — higher std (0.067 @ thr0.2 vs late-skip 0.037)",
          "  and a lower worst-prompt floor (min 0.709 vs 0.845); it misfires on high-motion",
          "  scenes (p02 Tokyo-rain, p10 cloud-timelapse) where its weak drift signal mis-times skips.",
          "- Root cause (whole cache family): no cheap signal predicts SANA-Video's per-step drift,",
          "  so adaptive scheduling can't beat the structural prior 'never skip early' = late-skip."]
    with open(f"{DST}/MANIFEST.md", "w") as f:
        f.write("\n".join(L) + "\n")

    print(f"staged -> {DST} ({len(os.listdir(DST))} files)")
    print("\nAGGREGATE mean corr / std / min:")
    for _, lbl, _, _ in CONFIGS:
        if lbl == "dense":
            continue
        a = np.array(agg[lbl]["corr"])
        if len(a):
            print(f"  {lbl:>6s}: {a.mean():.3f} / {a.std():.3f} / {a.min():.3f}  (sharp {np.mean(agg[lbl]['sharp']):.3f})")


if __name__ == "__main__":
    main()
