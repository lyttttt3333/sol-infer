#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 01:30:00
#SBATCH -J ltx23-hq-kwl-s2sparse
#SBATCH -o outputs/slurm/ltx23-hq-kwl-s2sparse-%j.out
#SBATCH -e outputs/slurm/ltx23-hq-kwl-s2sparse-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-sglang-hq-kwl-stage2-sparse-1080p10s}"
FORCE="${FORCE:-1}"
WARMUP="${WARMUP:-true}"
WARMUP_STEPS="${WARMUP_STEPS:-15}"
SUMMARY_JSON="$ROOT/benchmark_summary.json"
COMPARE_MP4="$ROOT/hq-kwl-vs-stage2-sparse-side-by-side.mp4"
mkdir -p "$ROOT"

variants=(kwl kwl_stage2_sparse)
labels=("SGLang HQ + KWL" "SGLang HQ + KWL + Stage2Sparse")
devices=(0 1)
ports=(30505 30515)
pids=()

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
    if [[ "${SGLANG_DIFFUSION_LTX2_EVENT_PROFILE:-0}" == "1" ]]; then
      export SGLANG_DIFFUSION_LTX2_PROFILE_PATH="$out_dir/ltx2_event_profile.json"
    fi
    if [[ "${SGLANG_PIECEWISE_ATTN_STATS_ENABLE:-0}" == "1" && "$variant" == *sparse* ]]; then
      export SGLANG_PIECEWISE_ATTN_STATS_PATH="$out_dir/piecewise_attn_stats.json"
      export SGLANG_PIECEWISE_ATTN_STATS_FLUSH_EVERY="${SGLANG_PIECEWISE_ATTN_STATS_FLUSH_EVERY:-20}"
    else
      unset SGLANG_PIECEWISE_ATTN_STATS_PATH
    fi
    bash scripts/run_ltx23_sglang_hq_1080p10s.sh "$variant"
  ) > "$ROOT/$variant.log" 2>&1 &
  pids+=("$!")
}

echo "[start] $(date -Is) root=$ROOT force=$FORCE warmup=$WARMUP warmup_steps=$WARMUP_STEPS"
for i in "${!variants[@]}"; do
  run_variant "${variants[$i]}" "${devices[$i]}" "${ports[$i]}"
done

set +e
statuses=()
for pid in "${pids[@]}"; do
  wait "$pid"
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
  --right "$ROOT/kwl_stage2_sparse/out.mp4" \
  --out "$COMPARE_MP4" \
  --left-label "HQ + KWL" \
  --right-label "HQ + KWL + stage2 sparse"

.conda/ltx23/bin/python - "$ROOT" "$SUMMARY_JSON" "$COMPARE_MP4" "${variants[@]}" <<'PYSUM'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
compare_mp4 = Path(sys.argv[3])
variants = sys.argv[4:]
stage_name_re = {
    "stage1": re.compile(r"LTX2AVDenoisingStage"),
    "stage2": re.compile(r"LTX2RefinementStage"),
    "decode": re.compile(r"LTX2AVDecodingStage"),
}

def load_perf(name: str):
    d = root / name
    perf = json.loads((d / "perf.json").read_text())
    semantics = json.loads((d / "hq_semantics.json").read_text())
    denoise_steps = perf.get("denoise_steps_ms", []) or []
    steps = perf.get("steps", []) or []
    stage_times = {}
    for step in steps:
        stage_name = str(step.get("name", ""))
        for key, pattern in stage_name_re.items():
            if pattern.search(stage_name):
                stage_times[key] = stage_times.get(key, 0.0) + float(step.get("duration_ms", 0.0)) / 1000.0
    return {
        "video": str(d / "out.mp4"),
        "perf_json": str(d / "perf.json"),
        "semantics_json": str(d / "hq_semantics.json"),
        "total_s": float(perf.get("total_duration_ms", 0.0)) / 1000.0,
        "denoise_total_s": sum(float(x.get("duration_ms", 0.0)) for x in denoise_steps) / 1000.0,
        "denoise_step_count": len(denoise_steps),
        "stage1_s": stage_times.get("stage1"),
        "stage2_s": stage_times.get("stage2"),
        "decode_s": stage_times.get("decode"),
        "semantics": semantics,
    }

results = {name: load_perf(name) for name in variants}
base = results["kwl"]
for item in results.values():
    item["speedup_vs_kwl_total"] = base["total_s"] / item["total_s"] if item["total_s"] else None
    item["speedup_vs_kwl_denoise"] = base["denoise_total_s"] / item["denoise_total_s"] if item["denoise_total_s"] else None
    item["speedup_vs_kwl_stage1"] = base["stage1_s"] / item["stage1_s"] if base.get("stage1_s") and item.get("stage1_s") else None
    item["speedup_vs_kwl_stage2"] = base["stage2_s"] / item["stage2_s"] if base.get("stage2_s") and item.get("stage2_s") else None

summary = {
    "root": str(root),
    "side_by_side_video": str(compare_mp4),
    "pipeline": "LTX2TwoStageHQPipeline",
    "resolution": "1920x1088",
    "num_frames": 241,
    "fps": 24,
    "stage1_steps": 15,
    "stage2_steps": 3,
    "baseline_variant": "kwl",
    "target_variant": "kwl_stage2_sparse",
    "target_method": {
        "kwl": "kernel-wise lossless fused/operator optimized HQ path",
        "stage2_sparse": "piecewise_attn only on transformer_2, sparsity=0.9, block_size=64, only video-to-video self attention, dense fallback=FA",
    },
    "variants": results,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PYSUM

echo "[done] side-by-side: $COMPARE_MP4"
echo "[done] summary: $SUMMARY_JSON"
