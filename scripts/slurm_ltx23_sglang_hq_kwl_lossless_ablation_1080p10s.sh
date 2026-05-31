#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -J ltx23-kwl-lossless-ablate
#SBATCH -o outputs/slurm/ltx23-kwl-lossless-ablate-%j.out
#SBATCH -e outputs/slurm/ltx23-kwl-lossless-ablate-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-sglang-hq-kwl-lossless-ablation-1080p10s}"
SHARED_DIR="$ROOT/shared_noise"
FORCE="${FORCE:-1}"
WARMUP="${WARMUP:-true}"
WARMUP_STEPS="${WARMUP_STEPS:-15}"
mkdir -p "$ROOT" "$SHARED_DIR"

echo "[start] $(date -Is) root=$ROOT force=$FORCE warmup=$WARMUP warmup_steps=$WARMUP_STEPS"

need_dense=0
if [[ "$FORCE" == "1" ]]; then
  need_dense=1
elif [[ ! -s "$ROOT/dense/out.mp4" || ! -s "$ROOT/dense/perf.json" ]]; then
  need_dense=1
elif [[ ! -s "$SHARED_DIR/sglang_stage1_video_initial.pt" || ! -s "$SHARED_DIR/sglang_stage1_audio_initial.pt" || ! -s "$SHARED_DIR/sglang_stage2_video_noise.pt" || ! -s "$SHARED_DIR/sglang_stage2_audio_noise.pt" ]]; then
  need_dense=1
fi

if [[ "$need_dense" == "1" ]]; then
  echo "[phase1] dense SGLang HQ dumps fixed stage1/stage2 noise"
  (
    export CUDA_VISIBLE_DEVICES=0
    export SGLANG_HQ_VARIANT=dense
    export ROOT="$ROOT"
    export OUT_DIR="$ROOT/dense"
    export FORCE=1
    export WARMUP="$WARMUP"
    export WARMUP_STEPS="$WARMUP_STEPS"
    export MASTER_PORT=30100
    export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$SHARED_DIR"
    export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$SHARED_DIR"
    unset SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH
    unset SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH
    unset SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH
    unset SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH
    bash scripts/run_ltx23_sglang_hq_1080p10s.sh
  ) > "$ROOT/dense.log" 2>&1
else
  echo "[phase1] reusing existing dense/noise artifacts"
fi

for required in \
  "$ROOT/dense/out.mp4" \
  "$ROOT/dense/perf.json" \
  "$SHARED_DIR/sglang_stage1_video_initial.pt" \
  "$SHARED_DIR/sglang_stage1_audio_initial.pt" \
  "$SHARED_DIR/sglang_stage2_video_noise.pt" \
  "$SHARED_DIR/sglang_stage2_audio_noise.pt"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing required artifact: $required" >&2
    exit 2
  fi
done

ALL_DIT_OFF=(
  SGLANG_HQ_KWL_FUSED_QK_ROPE=0
  SGLANG_HQ_KWL_FUSED_RMS_ADALN=0
  SGLANG_HQ_KWL_FUSED_ADALN=0
  SGLANG_HQ_KWL_FUSED_QKNORM_ROPE=0
  SGLANG_HQ_KWL_FUSED_DUAL_MODULATE=0
  SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL=0
  SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE=0
  SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU=0
  SGLANG_HQ_KWL_FUSED_AUDIO_QKVG=0
  SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE=0
)
NO_RUNTIME_COMPILE=(
  SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT=0
  SGLANG_HQ_KWL_COMPILE_TILED_VAE=0
)

run_kwl_variant() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  local share_block0="$4"
  local share_prefix="$5"
  shift 5
  echo "[launch] $name gpu=$gpu share_block0=$share_block0 share_prefix=$share_prefix extra=$*"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SGLANG_HQ_VARIANT=kwl
    export ROOT="$ROOT"
    export OUT_DIR="$ROOT/$name"
    export FORCE="$FORCE"
    export WARMUP="$WARMUP"
    export WARMUP_STEPS="$WARMUP_STEPS"
    export MASTER_PORT="$port"
    export SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN="$share_block0"
    export SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX="$share_prefix"
    export SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH="$SHARED_DIR/sglang_stage1_video_initial.pt"
    export SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH="$SHARED_DIR/sglang_stage1_audio_initial.pt"
    export SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH="$SHARED_DIR/sglang_stage2_video_noise.pt"
    export SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH="$SHARED_DIR/sglang_stage2_audio_noise.pt"
    export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$ROOT/$name/latents"
    export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$ROOT/$name/latents"
    while [[ $# -gt 0 ]]; do
      export "$1"
      shift
    done
    bash scripts/run_ltx23_sglang_hq_1080p10s.sh
  ) > "$ROOT/$name.log" 2>&1
}

wait_wave() {
  local status=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "[error] at least one ablation run failed" >&2
    exit "$status"
  fi
}

echo "[phase2] broad KWL semantic ablations"
run_kwl_variant kwl_all 0 30110 1 1 & pid0=$!
run_kwl_variant kwl_no_share 1 30111 0 0 & pid1=$!
run_kwl_variant kwl_dit_fused_only 2 30112 0 0 "${NO_RUNTIME_COMPILE[@]}" & pid2=$!
run_kwl_variant kwl_vae_compile_only 3 30113 0 0 "${ALL_DIT_OFF[@]}" SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT=0 SGLANG_HQ_KWL_COMPILE_TILED_VAE=1 & pid3=$!
wait_wave "$pid0" "$pid1" "$pid2" "$pid3"

echo "[phase3] grouped DiT/compile ablations"
run_kwl_variant kwl_compile_gate_only 0 30120 0 0 "${ALL_DIT_OFF[@]}" SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT=1 SGLANG_HQ_KWL_COMPILE_TILED_VAE=0 & pid0=$!
run_kwl_variant kwl_qknorm_rope_only 1 30121 0 0 "${ALL_DIT_OFF[@]}" "${NO_RUNTIME_COMPILE[@]}" SGLANG_HQ_KWL_FUSED_QK_ROPE=1 SGLANG_HQ_KWL_FUSED_QKNORM_ROPE=1 SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE=1 & pid1=$!
run_kwl_variant kwl_adaln_group_only 2 30122 0 0 "${ALL_DIT_OFF[@]}" "${NO_RUNTIME_COMPILE[@]}" SGLANG_HQ_KWL_FUSED_RMS_ADALN=1 SGLANG_HQ_KWL_FUSED_ADALN=1 SGLANG_HQ_KWL_FUSED_DUAL_MODULATE=1 SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL=1 SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE=1 & pid2=$!
run_kwl_variant kwl_ffn_audio_only 3 30123 0 0 "${ALL_DIT_OFF[@]}" "${NO_RUNTIME_COMPILE[@]}" SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU=1 SGLANG_HQ_KWL_FUSED_AUDIO_QKVG=1 & pid3=$!
wait_wave "$pid0" "$pid1" "$pid2" "$pid3"

VARIANTS=(
  kwl_all
  kwl_no_share
  kwl_dit_fused_only
  kwl_vae_compile_only
  kwl_compile_gate_only
  kwl_qknorm_rope_only
  kwl_adaln_group_only
  kwl_ffn_audio_only
)

for variant in "${VARIANTS[@]}"; do
  for required in "$ROOT/$variant/out.mp4" "$ROOT/$variant/perf.json" "$ROOT/$variant/hq_semantics.json"; do
    if [[ ! -s "$required" ]]; then
      echo "[error] missing required output: $required" >&2
      exit 3
    fi
  done
done

OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .conda/ltx23/bin/python scripts/make_multiway_video.py \
  --item "Dense=$ROOT/dense/out.mp4" \
  --item "KWL all=$ROOT/kwl_all/out.mp4" \
  --item "KWL no share=$ROOT/kwl_no_share/out.mp4" \
  --item "DiT fused only=$ROOT/kwl_dit_fused_only/out.mp4" \
  --item "VAE compile only=$ROOT/kwl_vae_compile_only/out.mp4" \
  --item "compile gate only=$ROOT/kwl_compile_gate_only/out.mp4" \
  --item "qknorm rope only=$ROOT/kwl_qknorm_rope_only/out.mp4" \
  --item "adaln group only=$ROOT/kwl_adaln_group_only/out.mp4" \
  --item "ffn audio only=$ROOT/kwl_ffn_audio_only/out.mp4" \
  --cols 3 \
  --tile-width 512 \
  --tile-height 288 \
  --out "$ROOT/kwl-lossless-ablation-9way.mp4"

OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .conda/ltx23/bin/python - "$ROOT" <<'PYSUM'
import json
import math
import sys
from pathlib import Path

import cv2

root = Path(sys.argv[1])
variants = [
    "dense",
    "kwl_all",
    "kwl_no_share",
    "kwl_dit_fused_only",
    "kwl_vae_compile_only",
    "kwl_compile_gate_only",
    "kwl_qknorm_rope_only",
    "kwl_adaln_group_only",
    "kwl_ffn_audio_only",
]

def perf(name: str) -> dict:
    path = root / name / "perf.json"
    data = json.loads(path.read_text())
    steps = data.get("steps", []) or []
    stages = {}
    for item in steps:
        stage = str(item.get("name", "unknown"))
        stages[stage] = stages.get(stage, 0.0) + float(item.get("duration_ms", 0.0)) / 1000.0
    denoise_steps = data.get("denoise_steps_ms", []) or []
    return {
        "perf_json": str(path),
        "output_video": str(root / name / "out.mp4"),
        "total_s": float(data.get("total_duration_ms", 0.0)) / 1000.0,
        "denoise_total_s": sum(float(item.get("duration_ms", 0.0)) for item in denoise_steps) / 1000.0,
        "denoise_step_count": len(denoise_steps),
        "stage_durations_s": stages,
    }

def frame_diff(left: Path, right: Path) -> dict:
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

dense_video = root / "dense/out.mp4"
summary = {
    "root": str(root),
    "multiway_video": str(root / "kwl-lossless-ablation-9way.mp4"),
    "variants": {},
    "diff_vs_dense": {},
}
for name in variants:
    summary["variants"][name] = perf(name)
    sem = root / name / "hq_semantics.json"
    if sem.exists():
        summary["variants"][name]["semantics"] = json.loads(sem.read_text())
    if name != "dense":
        summary["diff_vs_dense"][name] = frame_diff(dense_video, root / name / "out.mp4")
        total = summary["variants"][name]["total_s"]
        denoise = summary["variants"][name]["denoise_total_s"]
        dense_total = summary["variants"]["dense"]["total_s"]
        dense_denoise = summary["variants"]["dense"]["denoise_total_s"]
        summary["variants"][name]["speedup_vs_dense_total"] = dense_total / total if total else None
        summary["variants"][name]["speedup_vs_dense_denoise"] = dense_denoise / denoise if denoise else None

(root / "ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["diff_vs_dense"], indent=2, sort_keys=True), flush=True)
PYSUM

echo "[done] summary: $ROOT/ablation_summary.json"
echo "[done] multiway: $ROOT/kwl-lossless-ablation-9way.mp4"
