#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ROOT="${ROOT:-outputs/cosmos3-cache-matrix}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SIZES_TEXT="${MODEL_SIZES:-16b 64b}"
VARIANTS_TEXT="${VARIANTS:-baseline teacache_c04_s5 pab_cross2 dbcache_mild}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-35}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
FLOW_SHIFT="${FLOW_SHIFT:-10.0}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
WARMUP="${WARMUP:-false}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
SCHEDULER_PORT_BASE="${SCHEDULER_PORT_BASE:-5600}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-31000}"
PROMPT_COUNT="${PROMPT_COUNT:-2}"
PROMPT_START_INDEX="${PROMPT_START_INDEX:-0}"
PROMPT_END_INDEX="${PROMPT_END_INDEX:-$((PROMPT_COUNT - 1))}"
MAKE_COMPARE="${MAKE_COMPARE:-1}"
MAKE_REPORT="${MAKE_REPORT:-1}"
COSMOS3_16B_MODEL_PATH="${COSMOS3_16B_MODEL_PATH:-nvidia/Cosmos3-Nano}"
COSMOS3_64B_MODEL_PATH="${COSMOS3_64B_MODEL_PATH:-nvidia/Cosmos3-Super}"
COSMOS3_16B_NUM_GPUS="${COSMOS3_16B_NUM_GPUS:-1}"
COSMOS3_64B_NUM_GPUS="${COSMOS3_64B_NUM_GPUS:-4}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, distorted, low quality, flickering, overexposed, underexposed, low contrast, text artifacts, unstable motion}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PWD/outputs/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/outputs/.cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-$PWD/outputs/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/outputs/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/outputs/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/outputs/.cache/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$PWD/outputs/.cache/cuda}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export SGLANG_DIFFUSION_CACHE_ROOT="${SGLANG_DIFFUSION_CACHE_ROOT:-$PWD/outputs/.cache/sgl_diffusion}"
export TMPDIR="${TMPDIR:-$PWD/outputs/.tmp}"
export SGLANG_DISABLE_COSMOS3_GUARDRAILS="${SGLANG_DISABLE_COSMOS3_GUARDRAILS:-1}"
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING="${SGLANG_DIFFUSION_SYNC_STAGE_PROFILING:-1}"

mkdir -p \
  "$ROOT/logs" \
  outputs/.cache/huggingface \
  outputs/.cache/xdg \
  outputs/.cache/torch \
  outputs/.cache/triton \
  outputs/.cache/torchinductor \
  outputs/.cache/torch_extensions \
  outputs/.cache/cuda \
  outputs/.cache/sgl_diffusion \
  outputs/.tmp

read -r -a model_sizes <<< "$MODEL_SIZES_TEXT"
read -r -a variants <<< "$VARIANTS_TEXT"

prompts=(
  "${PROMPT_0:-A documentary video of an elderly botanist carefully watering orchids inside a glass greenhouse, morning sunlight, realistic hands, gentle camera movement, natural colors}"
  "${PROMPT_1:-A red fox running across a snowy forest trail at sunrise, powder snow kicked up by its paws, realistic fur motion, smooth tracking shot, cinematic natural light}"
  "${PROMPT_2:-A street food vendor flipping scallion pancakes on a busy night market grill, steam rising, realistic hands, handheld documentary camera, neon signs}"
  "${PROMPT_3:-A golden retriever jumping into a clear mountain lake, water splashing in slow motion, wet fur detail, bright natural daylight}"
  "${PROMPT_4:-A young violinist practicing alone in a sunlit apartment, close-up bow movement, dust floating in the light, realistic fingers}"
  "${PROMPT_5:-A hummingbird hovering beside red flowers in a backyard garden, fast wing motion, shallow depth of field, natural colors}"
  "${PROMPT_6:-A chef slicing tomatoes and herbs on a wooden cutting board, sharp knife movement, realistic kitchen lighting, close-up food texture}"
  "${PROMPT_7:-A horse galloping along a beach at sunset, wet sand reflections, flowing mane, smooth tracking shot, cinematic realism}"
  "${PROMPT_8:-A child in a yellow raincoat walking through shallow puddles after rain, city sidewalk reflections, gentle camera movement, realistic motion}"
  "${PROMPT_9:-A tabby cat stretching on a windowsill beside houseplants, afternoon sunlight, soft fur detail, calm indoor camera movement}"
)

model_path_for_size() {
  case "$1" in
    16b|nano|Nano) echo "$COSMOS3_16B_MODEL_PATH" ;;
    64b|super|Super) echo "$COSMOS3_64B_MODEL_PATH" ;;
    *) echo "[error] unknown MODEL_SIZES entry: $1" >&2; exit 2 ;;
  esac
}

num_gpus_for_size() {
  case "$1" in
    16b|nano|Nano) echo "$COSMOS3_16B_NUM_GPUS" ;;
    64b|super|Super) echo "$COSMOS3_64B_NUM_GPUS" ;;
    *) echo "[error] unknown MODEL_SIZES entry: $1" >&2; exit 2 ;;
  esac
}

label_for_variant() {
  if [[ "$1" =~ ^teacache_c([0-9]+)_s([0-9]+)(_m([0-9]+))?$ ]]; then
    local threshold_code="${BASH_REMATCH[1]}"
    local start_step="${BASH_REMATCH[2]}"
    local max_hits="${BASH_REMATCH[4]:-1}"
    local threshold
    threshold="$(awk -v code="$threshold_code" 'BEGIN { printf "%.2f", code / 100 }')"
    echo "TeaCache t${threshold} start${start_step} max${max_hits}"
    return
  fi
  case "$1" in
    baseline) echo "Baseline" ;;
    teacache_c04_s5) echo "TeaCache t0.04 start5" ;;
    teacache_c06_s5) echo "TeaCache t0.06 start5" ;;
    teacache_c08_s5) echo "TeaCache t0.08 start5" ;;
    teacache_c12_s5) echo "TeaCache t0.12 start5" ;;
    teacache_c16_s5) echo "TeaCache t0.16 start5" ;;
    teacache_c20_s5) echo "TeaCache t0.20 start5" ;;
    teacache_c30_s5) echo "TeaCache t0.30 start5" ;;
    teacache_c105_s5) echo "TeaCache t1.05 start5" ;;
    teacache_c110_s5) echo "TeaCache t1.10 start5" ;;
    teacache_c115_s5) echo "TeaCache t1.15 start5" ;;
    teacache_c120_s5) echo "TeaCache t1.20 start5" ;;
    pab_cross2) echo "PAB cross window2" ;;
    pab_cross3) echo "PAB cross window3" ;;
    dbcache_mild) echo "DBCache mild" ;;
    dbcache_target15) echo "DBCache target1.5x" ;;
    *) echo "$1" ;;
  esac
}

clear_cache_env() {
  export SGLANG_COSMOS3_TEACACHE_ENABLED=0
  unset SGLANG_COSMOS3_TEACACHE_THRESH
  unset SGLANG_COSMOS3_TEACACHE_START
  unset SGLANG_COSMOS3_TEACACHE_END
  unset SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS
  unset SGLANG_COSMOS3_TEACACHE_PERIODIC_RECOMPUTE_STEPS
  unset SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS

  export SGLANG_COSMOS3_PAB_ENABLED=0
  unset SGLANG_COSMOS3_PAB_CROSS_WINDOW
  unset SGLANG_COSMOS3_PAB_WARMUP
  unset SGLANG_COSMOS3_PAB_START
  unset SGLANG_COSMOS3_PAB_END

  export SGLANG_CACHE_DIT_ENABLED=0
  export SGLANG_CACHE_DIT_TAYLORSEER=0
  export SGLANG_CACHE_DIT_FN=1
  export SGLANG_CACHE_DIT_BN=0
  export SGLANG_CACHE_DIT_WARMUP=4
  export SGLANG_CACHE_DIT_RDT=0.24
  export SGLANG_CACHE_DIT_MC=3
  export SGLANG_CACHE_DIT_SCM_PRESET=none
  unset SGLANG_CACHE_DIT_SCM_COMPUTE_BINS
  unset SGLANG_CACHE_DIT_SCM_CACHE_BINS
}

configure_dynamic_teacache_variant_env() {
  if [[ "$1" =~ ^teacache_c([0-9]+)_s([0-9]+)(_m([0-9]+))?$ ]]; then
    local threshold_code="${BASH_REMATCH[1]}"
    local start_step="${BASH_REMATCH[2]}"
    local max_hits="${BASH_REMATCH[4]:-1}"
    local threshold
    threshold="$(awk -v code="$threshold_code" 'BEGIN { printf "%.2f", code / 100 }')"
    export SGLANG_COSMOS3_TEACACHE_ENABLED=1
    export SGLANG_COSMOS3_TEACACHE_THRESH="$threshold"
    export SGLANG_COSMOS3_TEACACHE_START="$start_step"
    export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS="$max_hits"
    export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
    return 0
  fi
  return 1
}

configure_variant_env() {
  clear_cache_env
  case "$1" in
    baseline)
      ;;
    teacache_c04_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=0.04
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      ;;
    teacache_c06_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=0.06
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      ;;
    teacache_c08_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=0.08
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      ;;
    teacache_c12_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=0.12
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
      ;;
    teacache_c16_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=0.16
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
      ;;
    teacache_c20_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=0.20
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
      ;;
    teacache_c30_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=0.30
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
      ;;
    teacache_c105_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=1.05
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
      ;;
    teacache_c110_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=1.10
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
      ;;
    teacache_c115_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=1.15
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
      ;;
    teacache_c120_s5)
      export SGLANG_COSMOS3_TEACACHE_ENABLED=1
      export SGLANG_COSMOS3_TEACACHE_THRESH=1.20
      export SGLANG_COSMOS3_TEACACHE_START=5
      export SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
      export SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1
      ;;
    pab_cross2)
      export SGLANG_COSMOS3_PAB_ENABLED=1
      export SGLANG_COSMOS3_PAB_CROSS_WINDOW=2
      export SGLANG_COSMOS3_PAB_WARMUP=5
      ;;
    pab_cross3)
      export SGLANG_COSMOS3_PAB_ENABLED=1
      export SGLANG_COSMOS3_PAB_CROSS_WINDOW=3
      export SGLANG_COSMOS3_PAB_WARMUP=5
      ;;
    dbcache_mild)
      export SGLANG_CACHE_DIT_ENABLED=1
      export SGLANG_CACHE_DIT_FN=2
      export SGLANG_CACHE_DIT_BN=2
      export SGLANG_CACHE_DIT_WARMUP=5
      export SGLANG_CACHE_DIT_RDT=0.12
      export SGLANG_CACHE_DIT_MC=1
      ;;
    dbcache_target15)
      export SGLANG_CACHE_DIT_ENABLED=1
      export SGLANG_CACHE_DIT_FN=1
      export SGLANG_CACHE_DIT_BN=1
      export SGLANG_CACHE_DIT_WARMUP=4
      export SGLANG_CACHE_DIT_RDT=0.18
      export SGLANG_CACHE_DIT_MC=2
      ;;
    *)
      if configure_dynamic_teacache_variant_env "$1"; then
        return
      fi
      echo "[error] unknown variant: $1" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local model_size="$1"
  local prompt_idx="$2"
  local variant="$3"
  local model_path
  local num_gpus
  model_path="$(model_path_for_size "$model_size")"
  num_gpus="$(num_gpus_for_size "$model_size")"

  local out_dir="$ROOT/$model_size/prompt_${prompt_idx}/$variant"
  local out_video="$out_dir/out.mp4"
  local perf_json="$out_dir/perf.json"
  local semantics_json="$out_dir/semantics.json"
  local log_file="$ROOT/logs/${model_size}_prompt${prompt_idx}_${variant}.log"
  local prompt="${prompts[$prompt_idx]}"
  local model_slot=0
  local variant_slot=0
  local i
  for i in "${!model_sizes[@]}"; do
    if [[ "${model_sizes[$i]}" == "$model_size" ]]; then
      model_slot="$i"
      break
    fi
  done
  for i in "${!variants[@]}"; do
    if [[ "${variants[$i]}" == "$variant" ]]; then
      variant_slot="$i"
      break
    fi
  done
  local port_offset=$(( model_slot * 10000 + prompt_idx * 100 + variant_slot * 10 ))
  local scheduler_port=$(( SCHEDULER_PORT_BASE + port_offset ))
  local master_port=$(( MASTER_PORT_BASE + port_offset ))

  if [[ "$FORCE" != "1" && -s "$out_video" && -s "$perf_json" ]]; then
    echo "[skip] existing $model_size prompt=$prompt_idx variant=$variant"
    return 0
  fi

  mkdir -p "$out_dir"
  configure_variant_env "$variant"

  cat > "$semantics_json" <<EOF
{
  "model_size": "$model_size",
  "model_path": "$model_path",
  "num_gpus": $num_gpus,
  "variant": "$variant",
  "variant_label": "$(label_for_variant "$variant")",
  "prompt_index": $prompt_idx,
  "prompt": $(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$prompt"),
  "negative_prompt": $(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$NEGATIVE_PROMPT"),
  "height": $HEIGHT,
  "width": $WIDTH,
  "num_frames": $NUM_FRAMES,
  "fps": $FPS,
  "num_inference_steps": $NUM_INFERENCE_STEPS,
  "guidance_scale": $GUIDANCE_SCALE,
  "flow_shift": $FLOW_SHIFT,
  "max_sequence_length": $MAX_SEQUENCE_LENGTH,
  "seed": $SEED,
  "scheduler_port": $scheduler_port,
  "master_port": $master_port
}
EOF

  cmd=(
    "$PYTHON_BIN" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate
    --model-path "$model_path"
    --num-gpus "$num_gpus"
    --prompt "$prompt"
    --negative-prompt "$NEGATIVE_PROMPT"
    --height "$HEIGHT"
    --width "$WIDTH"
    --num-frames "$NUM_FRAMES"
    --fps "$FPS"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --guidance-scale "$GUIDANCE_SCALE"
    --flow-shift "$FLOW_SHIFT"
    --max-sequence-length "$MAX_SEQUENCE_LENGTH"
    --use-guardrails false
    --seed "$SEED"
    --warmup "$WARMUP"
    --warmup-steps "$WARMUP_STEPS"
    --scheduler-port "$scheduler_port"
    --master-port "$master_port"
    --enable-sequence-shard true
    --output-file-path "$out_video"
    --perf-dump-path "$perf_json"
  )

  echo "[run] model=$model_size prompt=$prompt_idx variant=$variant gpus=$num_gpus scheduler_port=$scheduler_port master_port=$master_port"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  if ! "${cmd[@]}" > "$log_file" 2>&1; then
    echo "[error] failed: model=$model_size prompt=$prompt_idx variant=$variant log=$log_file" >&2
    return 1
  fi
}

failed=0
for model_size in "${model_sizes[@]}"; do
  for prompt_idx in "${!prompts[@]}"; do
    if (( prompt_idx >= PROMPT_COUNT )); then
      continue
    fi
    if (( prompt_idx < PROMPT_START_INDEX || prompt_idx > PROMPT_END_INDEX )); then
      continue
    fi
    for variant in "${variants[@]}"; do
      if ! run_one "$model_size" "$prompt_idx" "$variant"; then
        failed=1
        if [[ "$ALLOW_PARTIAL" != "1" ]]; then
          exit 1
        fi
      fi
    done

    compare_args=()
    for variant in "${variants[@]}"; do
      video="$ROOT/$model_size/prompt_${prompt_idx}/$variant/out.mp4"
      if [[ -s "$video" ]]; then
        compare_args+=(--item "$(label_for_variant "$variant")=$video")
      fi
    done
    if [[ "$DRY_RUN" != "1" && "$MAKE_COMPARE" == "1" && ${#compare_args[@]} -ge 4 ]]; then
      "$PYTHON_BIN" scripts/make_multiway_video.py "${compare_args[@]}" \
        --cols "${COMPARE_COLS:-3}" --tile-width 640 --tile-height 360 \
        --out "$ROOT/$model_size/prompt_${prompt_idx}/compare.mp4"
    fi
  done
done

if [[ "$DRY_RUN" != "1" && "$MAKE_REPORT" == "1" ]]; then
  "$PYTHON_BIN" scripts/make_cosmos3_cache_report.py \
    --root "$ROOT" \
    --model-sizes "$MODEL_SIZES_TEXT" \
    --variants "$VARIANTS_TEXT" \
    --prompt-count "$PROMPT_COUNT"
fi

if [[ "$failed" != "0" ]]; then
  exit 1
fi

echo "[done] root: $ROOT"
echo "[done] report: $ROOT/benchmark_report.html"
