#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -J ltx23-base-same-noise
#SBATCH -o outputs/slurm/ltx23-base-same-noise-%j.out
#SBATCH -e outputs/slurm/ltx23-base-same-noise-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

ROOT="${ROOT:-outputs/ltx23-kwl-vs-diffusers-same-noise-1080p10s}"
SHARED_DIR="$ROOT/shared_noise"
OUT_DIR="$ROOT/sglang_baseline_same_noise"
KWL_DIR="$ROOT/kwl_same_noise"
mkdir -p outputs/slurm "$OUT_DIR"

PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"
NEGATIVE_PROMPT="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."

export SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH="$SHARED_DIR/diffusers_stage1_video_initial.pt"
export SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH="$SHARED_DIR/diffusers_stage1_audio_initial.pt"
export SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH="$SHARED_DIR/diffusers_stage2_video_noise.pt"
export SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH="$SHARED_DIR/diffusers_stage2_audio_noise.pt"
export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$OUT_DIR/latents"
export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$OUT_DIR/latents"

.conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path Lightricks/LTX-2.3 \
  --backend auto \
  --pipeline-class-name LTX2TwoStagePipeline \
  --num-gpus 1 \
  --performance-mode speed \
  --ltx2-two-stage-device-mode resident \
  --height 1088 \
  --width 1920 \
  --num-frames 241 \
  --fps 24 \
  --seed 42 \
  --num-inference-steps 30 \
  --guidance-scale 3.0 \
  --guidance-rescale 0.7 \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --prompt "$PROMPT" \
  --return-file-paths-only true \
  --output-file-path "$OUT_DIR/out.mp4" \
  --perf-dump-path "$OUT_DIR/perf.json"

.conda/ltx23/bin/python scripts/make_side_by_side_video.py \
  --left "$KWL_DIR/out.mp4" \
  --right "$OUT_DIR/out.mp4" \
  --left-label "KWL same noise" \
  --right-label "SGLang baseline same noise" \
  --out "$ROOT/kwl-vs-sglang-baseline-same-noise-side-by-side.mp4"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .conda/ltx23/bin/python - "$ROOT" <<PY
import json
import sys
from pathlib import Path
import cv2
import numpy as np
cv2.setNumThreads(1)
root = Path(sys.argv[1])
cap = cv2.VideoCapture(str(root / "kwl-vs-sglang-baseline-same-noise-side-by-side.mp4"))
if not cap.isOpened():
    raise RuntimeError("open side-by-side failed")
metrics = []
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    left = frame[:, : w // 2].astype(np.float32)
    right = frame[:, w // 2 :].astype(np.float32)
    diff = np.abs(left - right)
    metrics.append({"frame": idx, "mean_abs": float(diff.mean()), "rmse": float(np.sqrt((diff * diff).mean()))})
    idx += 1
cap.release()
summary = {
    "frames": len(metrics),
    "mean_abs_avg": float(np.mean([m["mean_abs"] for m in metrics])),
    "rmse_avg": float(np.mean([m["rmse"] for m in metrics])),
    "mean_abs_samples": [metrics[i] for i in [0, 60, 120, 180, 240] if i < len(metrics)],
}
(root / "kwl_vs_sglang_baseline_pixel_diff_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
