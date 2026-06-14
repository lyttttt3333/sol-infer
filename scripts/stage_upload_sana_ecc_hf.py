#!/usr/bin/env python3
"""Stage + upload the EasyCache0.1+fusion (2.24x) demo set: 16 prompts, paired
dense vs ec010+compile, with corr+PSNR-vs-dense in each filename + a MANIFEST.
Runs on cpu_datamover (decode-heavy). Uploads to sana-video-easycache-compile/.

Layout: pNN_0dense.mp4 / pNN_1easycache-compile_<corr>_<psnr>.mp4 (sorted by prompt)."""
from __future__ import annotations

import os
import shutil

import imageio.v2 as imageio
import numpy as np
from huggingface_hub import HfApi

OUT = "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/outputs"
PRE = "home_yitongl_sana_video_outputs_sglang_sana_480p_val"
DST_DIR = "/home/yitongl/sana_video/hf_ecc"
REPO = "yitongl/ltx23-shares"
REPO_PATH = "sana-video-easycache-compile"
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
    if os.path.isdir(DST_DIR):
        shutil.rmtree(DST_DIR)
    os.makedirs(DST_DIR)
    rows, corrs, psnrs = [], [], []
    for i in range(N):
        d = load("dense", i); e = load("ecc", i)
        if d is None or e is None:
            print(f"MISSING p{i:02d} dense={d is not None} ecc={e is not None}")
            continue
        corr, psnr = metrics(e, d)
        corrs.append(corr); psnrs.append(psnr)
        shutil.copy2(f"{OUT}/{PRE}_dense_p{i:02d}.mp4", f"{DST_DIR}/p{i:02d}_0dense.mp4")
        shutil.copy2(f"{OUT}/{PRE}_ecc_p{i:02d}.mp4",
                     f"{DST_DIR}/p{i:02d}_1easycache-compile_corr{corr:.3f}_psnr{psnr:.1f}.mp4")
        rows.append(f"| p{i:02d} | {corr:.3f} | {psnr:.2f} | {PROMPTS[i]} |")

    L = ["# SANA-Video — EasyCache 0.1 + kernel fusion (the 2.24x end-to-end full-opt)", "",
         "480p, 832x480x81f, 50 steps. **EasyCache(thr 0.1) skip + torch.compile fusion, stacked.**",
         "13.1s/video steady-state vs 29.4s dense = **2.24x end-to-end**. Fusion is lossless, so",
         "quality = EasyCache-alone. Each prompt: `_0dense` (baseline) vs `_1easycache-compile`.",
         "corr/PSNR = vs dense (per-frame; note they under-rate temporal quality).", "",
         "| prompt | corr | PSNR dB | text |", "|---|---:|---:|---|"] + rows
    if corrs:
        L += ["", f"**mean corr {np.mean(corrs):.3f} (min {np.min(corrs):.3f}), "
              f"mean PSNR {np.mean(psnrs):.2f} dB** over {len(corrs)} prompts."]
    with open(f"{DST_DIR}/MANIFEST.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"staged {len(rows)} prompt-pairs + MANIFEST -> {DST_DIR}")

    token = open("/home/yitongl/.cache/huggingface/token").read().strip()
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    url = api.upload_folder(folder_path=DST_DIR, path_in_repo=REPO_PATH,
                            repo_id=REPO, repo_type="dataset",
                            commit_message="Add EasyCache0.1+fusion (2.24x) demo set: 16 prompts, dense-paired")
    print("UPLOADED_URL:", url, flush=True)


if __name__ == "__main__":
    main()
