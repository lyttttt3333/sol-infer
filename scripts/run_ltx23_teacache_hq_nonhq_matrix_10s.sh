#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ROOT="${ROOT:-outputs/ltx23-teacache-hq-nonhq-matrix-10s}"
FORCE="${FORCE:-1}"
WARMUP="${WARMUP:-false}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.conda/ltx23/bin/python}"
BASE_PORT="${BASE_PORT:-30410}"
DEVICES_CSV="${DEVICES:-0,1,2,3,4,5,6,7}"
HQ_VARIANTS_TEXT="${HQ_VARIANTS:-kwl kwl_teacache_c04_s6 kwl_teacache_c06_s5}"
NONHQ_VARIANTS_TEXT="${NONHQ_VARIANTS:-kwl kwl_cache_teacache_c04_s6 kwl_cache_teacache_c06_s5}"

IFS=',' read -r -a devices <<< "$DEVICES_CSV"
read -r -a hq_variants <<< "$HQ_VARIANTS_TEXT"
read -r -a nonhq_variants <<< "$NONHQ_VARIANTS_TEXT"

prompts=(
  "${PROMPT_0:-A cinematic 10 second close-up of an elderly woman ceramic artist shaping a blue clay vase on a pottery wheel, warm window light, realistic hands, fine clay texture, smooth camera movement, high detail}"
  "${PROMPT_1:-A wildlife documentary tracking shot of a red fox running through tall green grass at sunrise, dew sparkling, natural fur motion, shallow depth of field, realistic movement, high detail}"
)

mkdir -p "$ROOT/logs"

task_pipelines=()
task_prompt_indices=()
task_variants=()
for prompt_idx in "${!prompts[@]}"; do
  for variant in "${hq_variants[@]}"; do
    task_pipelines+=("hq")
    task_prompt_indices+=("$prompt_idx")
    task_variants+=("$variant")
  done
  for variant in "${nonhq_variants[@]}"; do
    task_pipelines+=("nonhq")
    task_prompt_indices+=("$prompt_idx")
    task_variants+=("$variant")
  done
done

variant_label() {
  case "$1" in
    kwl) echo "KWL baseline" ;;
    kwl_teacache_c04_s6|kwl_cache_teacache_c04_s6) echo "TeaCache t=0.04 s=6" ;;
    kwl_teacache_c06_s5|kwl_cache_teacache_c06_s5) echo "TeaCache t=0.06 s=5" ;;
    kwl_teacache_c08_s5|kwl_cache_teacache_c08_s5) echo "TeaCache t=0.08 s=5" ;;
    *) echo "$1" ;;
  esac
}

launch_task() {
  local pipeline="$1"
  local prompt_idx="$2"
  local variant="$3"
  local device="$4"
  local port="$5"
  local prompt="${prompts[$prompt_idx]}"
  local out_dir="$ROOT/$pipeline/prompt_${prompt_idx}/$variant"
  local log_file="$ROOT/logs/${pipeline}_prompt${prompt_idx}_${variant}.log"
  echo "[launch] pipeline=$pipeline prompt=$prompt_idx variant=$variant gpu=$device port=$port"
  (
    export CUDA_VISIBLE_DEVICES="$device"
    export PROMPT_INDEX="$prompt_idx"
    export PROMPT="$prompt"
    export ROOT="$ROOT/$pipeline"
    export OUT_DIR="$out_dir"
    export FORCE="$FORCE"
    export WARMUP="$WARMUP"
    export WARMUP_STEPS="$WARMUP_STEPS"
    export SEED="$SEED"
    export MASTER_PORT="$port"
    export PYTHON_BIN="$PYTHON_BIN"
    if [[ "$pipeline" == "hq" ]]; then
      export SGLANG_HQ_VARIANT="$variant"
      bash scripts/run_ltx23_sglang_hq_1080p10s.sh "$variant"
    else
      export SGLANG_NONHQ_VARIANT="$variant"
      bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh "$variant"
    fi
  ) > "$log_file" 2>&1 &
}

failed=0
batch_size=${#devices[@]}
for ((start=0; start<${#task_variants[@]}; start+=batch_size)); do
  pids=()
  for ((slot=0; slot<batch_size; slot++)); do
    idx=$((start + slot))
    if (( idx >= ${#task_variants[@]} )); then
      break
    fi
    port=$((BASE_PORT + idx))
    launch_task "${task_pipelines[$idx]}" "${task_prompt_indices[$idx]}" "${task_variants[$idx]}" "${devices[$slot]}" "$port"
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
done

if [[ "$failed" != "0" ]]; then
  echo "[error] at least one TeaCache matrix task failed; inspect $ROOT/logs" >&2
  exit 1
fi

for prompt_idx in "${!prompts[@]}"; do
  hq_args=()
  for variant in "${hq_variants[@]}"; do
    video="$ROOT/hq/prompt_${prompt_idx}/$variant/out.mp4"
    [[ -s "$video" ]] || { echo "[error] missing $video" >&2; exit 2; }
    hq_args+=(--item "$(variant_label "$variant")=$video")
  done
  "$PYTHON_BIN" scripts/make_multiway_video.py "${hq_args[@]}" \
    --cols "${HQ_COMPARE_COLS:-3}" --tile-width 640 --tile-height 360 \
    --out "$ROOT/hq/prompt_${prompt_idx}/compare.mp4"

  nonhq_args=()
  for variant in "${nonhq_variants[@]}"; do
    video="$ROOT/nonhq/prompt_${prompt_idx}/$variant/out.mp4"
    [[ -s "$video" ]] || { echo "[error] missing $video" >&2; exit 2; }
    nonhq_args+=(--item "$(variant_label "$variant")=$video")
  done
  "$PYTHON_BIN" scripts/make_multiway_video.py "${nonhq_args[@]}" \
    --cols "${NONHQ_COMPARE_COLS:-3}" --tile-width 640 --tile-height 360 \
    --out "$ROOT/nonhq/prompt_${prompt_idx}/compare.mp4"
done

"$PYTHON_BIN" scripts/make_ltx23_cache_report.py \
  --root "$ROOT" \
  --prompt-count "${#prompts[@]}" \
  --hq-variants "$HQ_VARIANTS_TEXT" \
  --nonhq-variants "$NONHQ_VARIANTS_TEXT" \
  > "$ROOT/benchmark_summary.stdout.json"

echo "[done] summary: $ROOT/benchmark_summary.json"
echo "[done] markdown: $ROOT/benchmark_summary.md"
echo "[done] html: $ROOT/benchmark_report.html"
