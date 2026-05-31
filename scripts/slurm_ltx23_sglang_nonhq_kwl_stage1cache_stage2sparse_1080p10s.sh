#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -J ltx23-kwl-s1cache-s2sparse
#SBATCH -o outputs/slurm/ltx23-kwl-s1cache-s2sparse-%j.out
#SBATCH -e outputs/slurm/ltx23-kwl-s1cache-s2sparse-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-sglang-nonhq-1080p-kwl-stage1cache-stage2sparse}"
FORCE="${FORCE:-1}"
WARMUP="${WARMUP:-true}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
SEED="${SEED:-42}"
SUMMARY_JSON="$ROOT/benchmark_summary.json"
mkdir -p "$ROOT"

variants=(
  kwl
  kwl_cache_teacache_c04_s6
  kwl_cache_teacache_c04_s6_stage2_sparse_piecewise
)
labels=(
  "SGLang+KWL"
  "SGLang+KWL+Stage1TeaCache"
  "SGLang+KWL+Stage1TeaCache+Stage2Sparse"
)

prompts=(
  "A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail"
)

devices=(0 1 2 3)
ports=(30405 30415 30425 30435)
failed=0

task_prompt_indices=()
task_variants=()
for prompt_idx in "${!prompts[@]}"; do
  for variant in "${variants[@]}"; do
    task_prompt_indices+=("$prompt_idx")
    task_variants+=("$variant")
  done
done

launch_task() {
  local prompt_idx="$1"
  local variant="$2"
  local device="$3"
  local port="$4"
  local out_dir="$ROOT/prompt_${prompt_idx}/$variant"
  local prompt="${prompts[$prompt_idx]}"
  echo "[launch] prompt=$prompt_idx variant=$variant gpu=$device port=$port"
  (
    export CUDA_VISIBLE_DEVICES="$device"
    export SGLANG_NONHQ_VARIANT="$variant"
    export PROMPT_INDEX="$prompt_idx"
    export PROMPT="$prompt"
    export ROOT="$ROOT"
    export OUT_DIR="$out_dir"
    export FORCE="$FORCE"
    export WARMUP="$WARMUP"
    export WARMUP_STEPS="$WARMUP_STEPS"
    export SEED="$SEED"
    export MASTER_PORT="$port"
    bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh "$variant"
  ) > "$ROOT/prompt_${prompt_idx}_${variant}.log" 2>&1 &
}

batch_size=${#devices[@]}
for ((start=0; start<${#task_variants[@]}; start+=batch_size)); do
  pids=()
  for ((slot=0; slot<batch_size; slot++)); do
    idx=$((start + slot))
    if (( idx >= ${#task_variants[@]} )); then
      break
    fi
    launch_task "${task_prompt_indices[$idx]}" "${task_variants[$idx]}" "${devices[$slot]}" "${ports[$slot]}"
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
done

if [[ "$failed" != "0" ]]; then
  echo "[error] at least one variant failed; inspect $ROOT/*.log" >&2
  exit 1
fi

for prompt_idx in "${!prompts[@]}"; do
  args=()
  for i in "${!variants[@]}"; do
    variant="${variants[$i]}"
    label="${labels[$i]}"
    video="$ROOT/prompt_${prompt_idx}/$variant/out.mp4"
    perf="$ROOT/prompt_${prompt_idx}/$variant/perf.json"
    sem="$ROOT/prompt_${prompt_idx}/$variant/nonhq_semantics.json"
    for required in "$video" "$perf" "$sem"; do
      if [[ ! -s "$required" ]]; then
        echo "[error] missing required output: $required" >&2
        exit 2
      fi
    done
    args+=(--item "$label=$video")
  done
  .conda/ltx23/bin/python scripts/make_multiway_video.py "${args[@]}" --cols 3 --tile-width 640 --tile-height 360 --out "$ROOT/prompt_${prompt_idx}/three_way.mp4"
done

.conda/ltx23/bin/python - "$ROOT" "$SUMMARY_JSON" "$SEED" "${variants[@]}" <<'PYSUM'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
seed = int(sys.argv[3])
variants = sys.argv[4:]

stage_name_re = {
    "stage1": re.compile(r"LTX2AVDenoisingStage"),
    "stage2": re.compile(r"LTX2RefinementStage"),
}


def load(prompt_idx: int, variant: str):
    d = root / f"prompt_{prompt_idx}" / variant
    perf = json.loads((d / "perf.json").read_text())
    sem = json.loads((d / "nonhq_semantics.json").read_text())
    denoise_steps = perf.get("denoise_steps_ms", []) or []
    steps = perf.get("steps", []) or []
    stage_times = {}
    for step in steps:
        name = str(step.get("name", ""))
        for key, pattern in stage_name_re.items():
            if pattern.search(name):
                stage_times[key] = float(step.get("duration_ms", 0.0)) / 1000.0
    return {
        "video": str(d / "out.mp4"),
        "perf_json": str(d / "perf.json"),
        "semantics_json": str(d / "nonhq_semantics.json"),
        "total_s": float(perf.get("total_duration_ms", 0.0)) / 1000.0,
        "denoise_total_s": sum(float(x.get("duration_ms", 0.0)) for x in denoise_steps) / 1000.0,
        "denoise_step_count": len(denoise_steps),
        "stage1_s": stage_times.get("stage1"),
        "stage2_s": stage_times.get("stage2"),
        "semantics": sem,
    }


prompts = {}
for prompt_idx in range(1):
    prompt_result = {variant: load(prompt_idx, variant) for variant in variants}
    base_kwl = prompt_result["kwl"]
    base_cache = prompt_result["kwl_cache_teacache_c04_s6"]
    for item in prompt_result.values():
        item["speedup_vs_kwl_total"] = base_kwl["total_s"] / item["total_s"] if item["total_s"] else None
        item["speedup_vs_kwl_denoise"] = base_kwl["denoise_total_s"] / item["denoise_total_s"] if item["denoise_total_s"] else None
        item["speedup_vs_kwl_stage1"] = (
            base_kwl["stage1_s"] / item["stage1_s"] if base_kwl.get("stage1_s") and item.get("stage1_s") else None
        )
        item["speedup_vs_kwl_stage2"] = (
            base_kwl["stage2_s"] / item["stage2_s"] if base_kwl.get("stage2_s") and item.get("stage2_s") else None
        )
        item["speedup_vs_kwl_cache_total"] = base_cache["total_s"] / item["total_s"] if item["total_s"] else None
        item["speedup_vs_kwl_cache_denoise"] = (
            base_cache["denoise_total_s"] / item["denoise_total_s"] if item["denoise_total_s"] else None
        )
        item["speedup_vs_kwl_cache_stage2"] = (
            base_cache["stage2_s"] / item["stage2_s"] if base_cache.get("stage2_s") and item.get("stage2_s") else None
        )
    prompts[f"prompt_{prompt_idx}"] = prompt_result

summary = {
    "root": str(root),
    "seed": seed,
    "pipeline": "LTX2TwoStagePipeline non-HQ 1080p",
    "resolution": "1920x1088",
    "num_frames": 241,
    "fps": 24,
    "stage1_steps": 30,
    "stage2_steps": 3,
    "noise_alignment": "All variants use the same prompt, seed, official negative prompt, non-HQ two-stage request settings, and 1080p frame geometry.",
    "variants": variants,
    "target_comparison": "kwl vs kwl_cache_teacache_c04_s6_stage2_sparse_piecewise",
    "target_method": {
        "kwl": "lossless kernel-wise SGLang DiT/VAE fused ops enabled by the runner",
        "stage1_cache": "TeaCache c=0.04, start step=6, max continuous hits=1, stage2 disabled",
        "stage2_sparse": "piecewise_attn only on transformer_2, sparsity=0.9, block_size=64, only video-to-video self attention, dense fallback=SDPA",
    },
    "per_prompt": prompts,
    "videos": {
        "prompt_0_three_way": str(root / "prompt_0" / "three_way.mp4"),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PYSUM

echo "[done] summary: $SUMMARY_JSON"
echo "[done] video: $ROOT/prompt_0/three_way.mp4"
