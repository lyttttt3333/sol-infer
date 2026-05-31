#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -J ltx23-hq-vs-kwl
#SBATCH -o outputs/slurm/ltx23-hq-vs-kwl-%j.out
#SBATCH -e outputs/slurm/ltx23-hq-vs-kwl-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-sglang-hq-vs-kwl-1080p10s}"
FORCE="${FORCE:-1}"
WARMUP="${WARMUP:-true}"
WARMUP_STEPS="${WARMUP_STEPS:-15}"
COMPARE_MP4="$ROOT/sglang-hq-vs-kwl-side-by-side.mp4"
SUMMARY_JSON="$ROOT/benchmark_summary.json"
mkdir -p "$ROOT"

echo "[start] $(date -Is) root=$ROOT force=$FORCE warmup=$WARMUP warmup_steps=$WARMUP_STEPS"

echo "[launch] dense on GPU0"
(
  export CUDA_VISIBLE_DEVICES=0
  export SGLANG_HQ_VARIANT=dense
  export ROOT="$ROOT"
  export OUT_DIR="$ROOT/dense"
  export FORCE="$FORCE"
  export WARMUP="$WARMUP"
  export WARMUP_STEPS="$WARMUP_STEPS"
  export MASTER_PORT=30005
  bash scripts/run_ltx23_sglang_hq_1080p10s.sh
) > "$ROOT/dense.log" 2>&1 &
DENSE_PID=$!

echo "[launch] kwl on GPU1"
(
  export CUDA_VISIBLE_DEVICES=1
  export SGLANG_HQ_VARIANT=kwl
  export ROOT="$ROOT"
  export OUT_DIR="$ROOT/kwl"
  export FORCE="$FORCE"
  export WARMUP="$WARMUP"
  export WARMUP_STEPS="$WARMUP_STEPS"
  export MASTER_PORT=30015
  bash scripts/run_ltx23_sglang_hq_1080p10s.sh
) > "$ROOT/kwl.log" 2>&1 &
KWL_PID=$!

set +e
wait "$DENSE_PID"
DENSE_STATUS=$?
wait "$KWL_PID"
KWL_STATUS=$?
set -e

echo "[done-runs] dense_status=$DENSE_STATUS kwl_status=$KWL_STATUS"
if [[ "$DENSE_STATUS" != "0" || "$KWL_STATUS" != "0" ]]; then
  echo "[error] at least one variant failed; see $ROOT/dense.log and $ROOT/kwl.log" >&2
  exit 1
fi

for required in "$ROOT/dense/out.mp4" "$ROOT/kwl/out.mp4" "$ROOT/dense/perf.json" "$ROOT/kwl/perf.json"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing required output: $required" >&2
    exit 2
  fi
done

.conda/ltx23/bin/python scripts/make_side_by_side_video.py \
  --left "$ROOT/dense/out.mp4" \
  --right "$ROOT/kwl/out.mp4" \
  --out "$COMPARE_MP4" \
  --left-label "SGLang HQ" \
  --right-label "SGLang HQ + KWL"

.conda/ltx23/bin/python - "$ROOT" "$SUMMARY_JSON" "$COMPARE_MP4" <<'PYSUM'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
compare_mp4 = Path(sys.argv[3])

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

dense = load_perf("dense")
kwl = load_perf("kwl")
summary = {
    "root": str(root),
    "side_by_side_video": str(compare_mp4),
    "dense": dense,
    "kwl": kwl,
    "speedup_kwl_vs_dense_total": (dense["total_s"] / kwl["total_s"]) if kwl["total_s"] else None,
    "speedup_kwl_vs_dense_denoise": (dense["denoise_total_s"] / kwl["denoise_total_s"]) if kwl["denoise_total_s"] else None,
    "semantics": {
        "pipeline_class_name": "LTX2TwoStageHQPipeline",
        "stage1_steps": 15,
        "stage2_sigmas": [0.909375, 0.725, 0.421875, 0.0],
        "stage2_steps": 3,
        "stage1_lora_strength": 0.25,
        "stage2_lora_strength": 0.5,
        "sparse_attention": False,
        "nvfp4_fp4": False,
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PYSUM

echo "[done] side-by-side: $COMPARE_MP4"
echo "[done] summary: $SUMMARY_JSON"
