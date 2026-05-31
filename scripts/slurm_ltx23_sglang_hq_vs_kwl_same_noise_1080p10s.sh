#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -J ltx23-hq-kwl-samenoise
#SBATCH -o outputs/slurm/ltx23-hq-kwl-samenoise-%j.out
#SBATCH -e outputs/slurm/ltx23-hq-kwl-samenoise-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-sglang-hq-vs-kwl-same-noise-1080p10s}"
SHARED_DIR="$ROOT/shared_noise"
FORCE="${FORCE:-1}"
WARMUP="${WARMUP:-true}"
WARMUP_STEPS="${WARMUP_STEPS:-15}"
COMPARE_MP4="$ROOT/sglang-hq-vs-kwl-same-noise-side-by-side.mp4"
SUMMARY_JSON="$ROOT/benchmark_summary.json"
DIFF_JSON="$ROOT/frame_diff_metrics.json"
mkdir -p "$ROOT" "$SHARED_DIR"

echo "[start] $(date -Is) root=$ROOT force=$FORCE warmup=$WARMUP warmup_steps=$WARMUP_STEPS"
echo "[phase1] dense SGLang HQ dumps stage1/stage2 noise -> $SHARED_DIR"
(
  export CUDA_VISIBLE_DEVICES=0
  export SGLANG_HQ_VARIANT=dense
  export ROOT="$ROOT"
  export OUT_DIR="$ROOT/dense"
  export FORCE="$FORCE"
  export WARMUP="$WARMUP"
  export WARMUP_STEPS="$WARMUP_STEPS"
  export MASTER_PORT=30005
  export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$SHARED_DIR"
  export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$SHARED_DIR"
  unset SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH
  unset SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH
  unset SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH
  unset SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH
  bash scripts/run_ltx23_sglang_hq_1080p10s.sh
) > "$ROOT/dense.log" 2>&1

for required in \
  "$SHARED_DIR/sglang_stage1_video_initial.pt" \
  "$SHARED_DIR/sglang_stage1_audio_initial.pt" \
  "$SHARED_DIR/sglang_stage2_video_noise.pt" \
  "$SHARED_DIR/sglang_stage2_audio_noise.pt"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] dense did not create required noise artifact: $required" >&2
    exit 2
  fi
done

echo "[phase2] KWL reads dense stage1/stage2 noise"
(
  export CUDA_VISIBLE_DEVICES=1
  export SGLANG_HQ_VARIANT=kwl
  export ROOT="$ROOT"
  export OUT_DIR="$ROOT/kwl_same_noise"
  export FORCE="$FORCE"
  export WARMUP="$WARMUP"
  export WARMUP_STEPS="$WARMUP_STEPS"
  export MASTER_PORT=30015
  export SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH="$SHARED_DIR/sglang_stage1_video_initial.pt"
  export SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH="$SHARED_DIR/sglang_stage1_audio_initial.pt"
  export SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH="$SHARED_DIR/sglang_stage2_video_noise.pt"
  export SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH="$SHARED_DIR/sglang_stage2_audio_noise.pt"
  export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$ROOT/kwl_same_noise/latents"
  export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$ROOT/kwl_same_noise/latents"
  bash scripts/run_ltx23_sglang_hq_1080p10s.sh
) > "$ROOT/kwl_same_noise.log" 2>&1

for required in "$ROOT/dense/out.mp4" "$ROOT/kwl_same_noise/out.mp4" "$ROOT/dense/perf.json" "$ROOT/kwl_same_noise/perf.json"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing required output: $required" >&2
    exit 3
  fi
done

.conda/ltx23/bin/python scripts/make_side_by_side_video.py \
  --left "$ROOT/dense/out.mp4" \
  --right "$ROOT/kwl_same_noise/out.mp4" \
  --out "$COMPARE_MP4" \
  --left-label "SGLang HQ same noise" \
  --right-label "SGLang HQ + KWL same noise"

.conda/ltx23/bin/python - "$ROOT" "$SUMMARY_JSON" "$COMPARE_MP4" "$DIFF_JSON" <<'PYSUM'
import json
import math
import sys
from pathlib import Path

import cv2

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
compare_mp4 = Path(sys.argv[3])
diff_path = Path(sys.argv[4])

def load_perf(name: str):
    path = root / name / "perf.json"
    data = json.loads(path.read_text())
    steps = data.get("steps", []) or []
    stage_durations_s = {}
    for item in steps:
        stage = str(item.get("name", "unknown"))
        stage_durations_s[stage] = stage_durations_s.get(stage, 0.0) + float(item.get("duration_ms", 0.0)) / 1000.0
    denoise_steps = data.get("denoise_steps_ms", []) or []
    denoise_total_s = sum(float(item.get("duration_ms", 0.0)) for item in denoise_steps) / 1000.0
    return {
        "perf_json": str(path),
        "output_video": str(root / name / "out.mp4"),
        "total_s": float(data.get("total_duration_ms", 0.0)) / 1000.0,
        "denoise_total_s": denoise_total_s,
        "denoise_step_count": len(denoise_steps),
        "stage_durations_s": stage_durations_s,
    }

def frame_diff(left: Path, right: Path):
    cap_l = cv2.VideoCapture(str(left))
    cap_r = cv2.VideoCapture(str(right))
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError(f"failed to open videos: {left}, {right}")
    frames = 0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_pix = 0
    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            break
        if frame_l.shape != frame_r.shape:
            frame_r = cv2.resize(frame_r, (frame_l.shape[1], frame_l.shape[0]))
        diff = frame_l.astype("float32") - frame_r.astype("float32")
        sum_abs += float(abs(diff).sum())
        sum_sq += float((diff * diff).sum())
        sum_pix += int(diff.size)
        frames += 1
    cap_l.release()
    cap_r.release()
    mse = sum_sq / sum_pix if sum_pix else 0.0
    mad = sum_abs / sum_pix if sum_pix else 0.0
    psnr = 10.0 * math.log10((255.0 * 255.0) / mse) if mse > 0 else float("inf")
    return {"frames": frames, "mean_abs_diff": mad, "mse": mse, "psnr_db": psnr}

dense = load_perf("dense")
kwl = load_perf("kwl_same_noise")
diff = frame_diff(root / "dense/out.mp4", root / "kwl_same_noise/out.mp4")
diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n")
summary = {
    "root": str(root),
    "side_by_side_video": str(compare_mp4),
    "frame_diff_metrics": str(diff_path),
    "dense": dense,
    "kwl_same_noise": kwl,
    "speedup_kwl_vs_dense_total": (dense["total_s"] / kwl["total_s"]) if kwl["total_s"] else None,
    "speedup_kwl_vs_dense_denoise": (dense["denoise_total_s"] / kwl["denoise_total_s"]) if kwl["denoise_total_s"] else None,
    "visual_diff": diff,
    "semantics": {
        "pipeline_class_name": "LTX2TwoStageHQPipeline",
        "stage1_steps": 15,
        "stage2_sigmas": [0.909375, 0.725, 0.421875, 0.0],
        "stage2_steps": 3,
        "stage1_lora_strength": 0.25,
        "stage2_lora_strength": 0.5,
        "same_stage1_initial_latents": True,
        "same_stage2_renoise": True,
        "sparse_attention": False,
        "nvfp4_fp4": False,
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PYSUM

echo "[done] side-by-side: $COMPARE_MP4"
echo "[done] summary: $SUMMARY_JSON"
echo "[done] frame diff: $DIFF_JSON"
