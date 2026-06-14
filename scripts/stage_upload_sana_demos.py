#!/usr/bin/env python3
"""Stage + upload a paired (dense vs <cand>) SANA-Video demo set to HF, on
cpu_datamover (decode-heavy). Generalizes the ecc demo staging.

Usage: stage_upload_sana_demos.py <cand_label> <repo_subdir> <title> [speedup_str]
e.g.   stage_upload_sana_demos.py fullopt sana-video-fullopt-2.77x "EasyCache+fusion+linattn-bf16" "2.77x (10.6s)"
"""
from __future__ import annotations

import os
import shutil
import sys

import imageio.v2 as imageio
import numpy as np
from huggingface_hub import HfApi

OUT = "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/outputs"
PRE = "home_yitongl_sana_video_outputs_sglang_sana_480p_val"
REPO = "yitongl/ltx23-shares"
N = 16
PROMPTS = [
    "cat & dog baking a cake", "eagle over snowy mountains", "Tokyo street at night, rain",
    "blooming red rose, dew", "surfer riding a wave", "steam locomotive in valley",
    "child flying a kite, beach", "chef tossing a flaming wok", "dolphins leaping",
    "clouds over wheat field (timelapse)", "robot in futuristic city", "waterfall, mossy rocks",
    "hot-air balloon over farmland", "panda eating bamboo", "vintage car, coast", "figure skater spinning",
]


def load(label, i):
    p = f"{OUT}/{PRE}_{label}_p{i:02d}.mp4"
    try:
        return np.stack([np.asarray(f, dtype=np.float32) for f in imageio.get_reader(p)])
    except Exception:
        return None


def metrics(c, r):
    T = min(len(c), len(r)); c, r = c[:T], r[:T]
    cf, rf = c.reshape(T, -1), r.reshape(T, -1)
    corr = float(np.mean([np.corrcoef(cf[i], rf[i])[0, 1] for i in range(T)]))
    mse = np.maximum(((c - r) ** 2).reshape(T, -1).mean(1), 1e-8)
    return corr, float((10 * np.log10(255.0 ** 2 / mse)).mean())


def main():
    cand, sub, title = sys.argv[1], sys.argv[2], sys.argv[3]
    speed = sys.argv[4] if len(sys.argv) > 4 else ""
    dst = f"/home/yitongl/sana_video/hf_{cand}"
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    rows, corrs, psnrs = [], [], []
    for i in range(N):
        d, c = load("dense", i), load(cand, i)
        if d is None or c is None:
            print(f"MISSING p{i:02d} dense={d is not None} {cand}={c is not None}")
            continue
        corr, psnr = metrics(c, d)
        corrs.append(corr); psnrs.append(psnr)
        shutil.copy2(f"{OUT}/{PRE}_dense_p{i:02d}.mp4", f"{dst}/p{i:02d}_0dense.mp4")
        shutil.copy2(f"{OUT}/{PRE}_{cand}_p{i:02d}.mp4",
                     f"{dst}/p{i:02d}_1{cand}_corr{corr:.3f}_psnr{psnr:.1f}.mp4")
        rows.append(f"| p{i:02d} | {corr:.3f} | {psnr:.2f} | {PROMPTS[i]} |")

    L = [f"# SANA-Video demo set — {title}" + (f"  ({speed} end-to-end)" if speed else ""), "",
         "480p, 832x480x81f, 50 steps. Each prompt: `_0dense` (baseline) vs "
         f"`_1{cand}`. corr/PSNR vs dense (per-frame; under-rate temporal quality).", "",
         "| prompt | corr | PSNR dB | text |", "|---|---:|---:|---|"] + rows
    if corrs:
        L += ["", f"**mean corr {np.mean(corrs):.3f} (min {np.min(corrs):.3f}), "
              f"mean PSNR {np.mean(psnrs):.2f} dB** over {len(corrs)} prompts."]
    with open(f"{dst}/MANIFEST.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"staged {len(rows)} pairs + MANIFEST -> {dst}")

    token = open("/home/yitongl/.cache/huggingface/token").read().strip()
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    url = api.upload_folder(folder_path=dst, path_in_repo=sub, repo_id=REPO, repo_type="dataset",
                            commit_message=f"Add SANA-Video demo set: {title} ({speed})")
    print("UPLOADED_URL:", url, flush=True)
    if corrs:
        print(f"SUMMARY mean_corr={np.mean(corrs):.3f} min_corr={np.min(corrs):.3f} mean_psnr={np.mean(psnrs):.2f}")


if __name__ == "__main__":
    main()
