#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -J ltx23-hq-kwl-sparse-cache
#SBATCH -o outputs/slurm/ltx23-hq-kwl-sparse-cache-%j.out
#SBATCH -e outputs/slurm/ltx23-hq-kwl-sparse-cache-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-sglang-hq-kwl-sparse-cache-matrix-1080p10s}"
FORCE="${FORCE:-1}"
WARMUP="${WARMUP:-true}"
WARMUP_STEPS="${WARMUP_STEPS:-15}"
SUMMARY_JSON="$ROOT/benchmark_summary.json"
COMPARE_MP4="$ROOT/kwl-vs-kwl-sparse-cache-side-by-side.mp4"
mkdir -p "$ROOT"

variants=(kwl kwl_sparse kwl_cache kwl_sparse_cache)
devices=(0 1 2 3)
ports=(30105 30115 30125 30135)
pids=()

log_env() {
  echo "[env] ROOT=$ROOT FORCE=$FORCE WARMUP=$WARMUP WARMUP_STEPS=$WARMUP_STEPS"
  echo "[env] SGLANG_HQ_CACHE_ALGO=${SGLANG_HQ_CACHE_ALGO:-pab}"
  echo "[env] sparse_dense_layers=${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS:-0} sparse_dense_steps=${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS:-3}"
  echo "[env] pab_start=${SGLANG_LTX2_PAB_START_STEP:-6} pab_stage2_enabled=${SGLANG_LTX2_PAB_STAGE2_ENABLED:-0} pab_stage2_start=${SGLANG_LTX2_PAB_STAGE2_START_STEP:-0}"
}

run_variant() {
  local variant="$1"
  local device="$2"
  local port="$3"
  local out_dir="$ROOT/$variant"
  echo "[launch] $variant on GPU${device} port=${port}"
  (
    export CUDA_VISIBLE_DEVICES="$device"
    export SGLANG_HQ_VARIANT="$variant"
    export ROOT="$ROOT"
    export OUT_DIR="$out_dir"
    export FORCE="$FORCE"
    export WARMUP="$WARMUP"
    export WARMUP_STEPS="$WARMUP_STEPS"
    export MASTER_PORT="$port"
    bash scripts/run_ltx23_sglang_hq_1080p10s.sh "$variant"
  ) > "$ROOT/$variant.log" 2>&1 &
  pids+=("$!")
}

log_env
for i in "${!variants[@]}"; do
  run_variant "${variants[$i]}" "${devices[$i]}" "${ports[$i]}"
done

set +e
statuses=()
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  statuses+=("$?")
done
set -e

failed=0
for i in "${!variants[@]}"; do
  echo "[done-runs] ${variants[$i]} status=${statuses[$i]} log=$ROOT/${variants[$i]}.log"
  if [[ "${statuses[$i]}" != "0" ]]; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "[error] at least one variant failed; inspect $ROOT/*.log" >&2
  exit 1
fi

for variant in "${variants[@]}"; do
  for required in "$ROOT/$variant/out.mp4" "$ROOT/$variant/perf.json" "$ROOT/$variant/hq_semantics.json"; do
    if [[ ! -s "$required" ]]; then
      echo "[error] missing required output: $required" >&2
      exit 2
    fi
  done
done

.conda/ltx23/bin/python scripts/make_side_by_side_video.py \
  --left "$ROOT/kwl/out.mp4" \
  --right "$ROOT/kwl_sparse_cache/out.mp4" \
  --out "$COMPARE_MP4" \
  --left-label "HQ + KWL" \
  --right-label "HQ + KWL + sparse + PAB"

.conda/ltx23/bin/python - "$ROOT" "$SUMMARY_JSON" "$COMPARE_MP4" "${variants[@]}" <<'PYSUM'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
compare_mp4 = Path(sys.argv[3])
variants = sys.argv[4:]

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
    semantics_path = root / name / "hq_semantics.json"
    semantics = json.loads(semantics_path.read_text()) if semantics_path.exists() else {}
    return {
        "perf_json": str(path),
        "semantics_json": str(semantics_path),
        "output_video": str(root / name / "out.mp4"),
        "total_s": float(data.get("total_duration_ms", 0.0)) / 1000.0,
        "denoise_total_s": denoise_total_s,
        "denoise_step_count": len(denoise_steps),
        "stage_durations_s": stage_durations_s,
        "semantics": semantics,
    }

results = {name: load_perf(name) for name in variants}
kwl = results.get("kwl", {})
kwl_total = float(kwl.get("total_s") or 0.0)
kwl_denoise = float(kwl.get("denoise_total_s") or 0.0)
for name, item in results.items():
    total = float(item.get("total_s") or 0.0)
    denoise = float(item.get("denoise_total_s") or 0.0)
    item["speedup_vs_kwl_total"] = (kwl_total / total) if total and kwl_total else None
    item["speedup_vs_kwl_denoise"] = (kwl_denoise / denoise) if denoise and kwl_denoise else None

summary = {
    "root": str(root),
    "side_by_side_video": str(compare_mp4),
    "baseline_variant": "kwl",
    "variants": results,
    "notes": {
        "kwl": "kernel-wise lossless fused/operator optimized HQ path",
        "kwl_sparse": "KWL plus piecewise sparse video self-attention; lossy algorithmic approximation",
        "kwl_cache": "KWL plus LTX2 PAB attention-output cache; lossy algorithmic reuse",
        "kwl_sparse_cache": "KWL plus sparse attention plus PAB cache",
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PYSUM

echo "[done] side-by-side: $COMPARE_MP4"
echo "[done] summary: $SUMMARY_JSON"
