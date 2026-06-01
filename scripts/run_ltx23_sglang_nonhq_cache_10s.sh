#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${SGLANG_NONHQ_VARIANT:-${1:-dense}}"
case "$VARIANT" in
  dense|kwl|cache_pab_late12_w3|cache_teacache_c04_s6|cache_teacache_c06_s5|cache_teacache_c08_s5|cache_dbcache_aggressive|kwl_cache_teacache_c04_s6|kwl_cache_teacache_c06_s5|kwl_cache_teacache_c08_s5|kwl_cache_teacache_c04_s6_sparse_piecewise|kwl_cache_teacache_c04_s6_stage2_sparse_piecewise) ;;
  *)
    echo "Usage: SGLANG_NONHQ_VARIANT=dense|kwl|cache_pab_late12_w3|cache_teacache_c04_s6|cache_teacache_c06_s5|cache_teacache_c08_s5|cache_dbcache_aggressive|kwl_cache_teacache_c04_s6|kwl_cache_teacache_c06_s5|kwl_cache_teacache_c08_s5|kwl_cache_teacache_c04_s6_sparse_piecewise|kwl_cache_teacache_c04_s6_stage2_sparse_piecewise $0 [variant]" >&2
    exit 2
    ;;
esac

mkdir -p outputs/slurm
source "$REPO_ROOT/scripts/env_ltx23_persistent_cache.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"

if [[ -d "$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13" ]]; then
  export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
  export CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-$PWD/.conda/ltx23/bin/python}"
MODEL_PATH="${MODEL_PATH:-outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493}"
OFFICIAL_MODEL_DIR="${OFFICIAL_MODEL_DIR:-outputs/LTX-2.3-official-files}"
DISTILLED_LORA="${DISTILLED_LORA:-$OFFICIAL_MODEL_DIR/ltx-2.3-22b-distilled-lora-384-1.1.safetensors}"
SPATIAL_UPSAMPLER="${SPATIAL_UPSAMPLER:-$MODEL_PATH/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
ROOT="${ROOT:-outputs/ltx23-sglang-nonhq-cache-10s}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/prompt_${PROMPT_INDEX}/$VARIANT}"
OUT_VIDEO="$OUT_DIR/out.mp4"
PERF_JSON="$OUT_DIR/perf.json"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1088}"
NUM_FRAMES="${NUM_FRAMES:-241}"
FPS="${FPS:-24}"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
WARMUP="${WARMUP:-false}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"
MASTER_PORT="${MASTER_PORT:-30005}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-${SGLANG_LTX2_PERFORMANCE_MODE:-speed}}"
TWO_STAGE_DEVICE_MODE="${TWO_STAGE_DEVICE_MODE:-${SGLANG_LTX2_TWO_STAGE_DEVICE_MODE:-resident}}"
export PERFORMANCE_MODE TWO_STAGE_DEVICE_MODE

for required in "$PYTHON_BIN" "$MODEL_PATH/model_index.json" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing required SGLang non-HQ asset: $required" >&2
    exit 1
  fi
done

clear_cache_env() {
  export SGLANG_LTX2_PAB_ENABLED=0
  export SGLANG_CACHE_DIT_ENABLED=0
  export SGLANG_LTX2_TEACACHE_ENABLED=0
  unset SGLANG_LTX2_PAB_SPATIAL_WINDOW SGLANG_LTX2_PAB_TEMPORAL_WINDOW SGLANG_LTX2_PAB_CROSS_WINDOW
  unset SGLANG_LTX2_PAB_START_STEP SGLANG_LTX2_PAB_END_STEP SGLANG_LTX2_PAB_DISABLE_AUDIO_VIDEO_CROSS
  unset SGLANG_LTX2_PAB_A2V_WINDOW SGLANG_LTX2_PAB_V2A_WINDOW SGLANG_LTX2_PAB_STAGE2_ENABLED
  unset SGLANG_CACHE_DIT_WARMUP SGLANG_CACHE_DIT_RDT SGLANG_CACHE_DIT_MC SGLANG_CACHE_DIT_FN SGLANG_CACHE_DIT_BN
  unset SGLANG_LTX2_TEACACHE_THRESH SGLANG_LTX2_TEACACHE_START SGLANG_LTX2_TEACACHE_END
  unset SGLANG_LTX2_TEACACHE_STAGE2_DISABLE SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS
  unset SGLANG_LTX2_TEACACHE_STAGE1_ENABLED SGLANG_LTX2_TEACACHE_PERIODIC_RECOMPUTE_STEPS
}

clear_sparse_env() {
  export SGLANG_NONHQ_SPARSE_ENABLED=0
  unset SGLANG_PIECEWISE_ATTN_SPARSITY SGLANG_PIECEWISE_ATTN_BLOCK_SIZE SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF
  unset SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS
  unset SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY
  unset SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK
  COMPONENT_ATTENTION_BACKENDS="${COMPONENT_ATTENTION_BACKENDS:-transformer=fa,transformer_2=fa}"
  ATTENTION_BACKEND_CONFIG="${ATTENTION_BACKEND_CONFIG:-}"
  SPARSE_ALGO="none"
}

disable_kwl_env() {
  export SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=0
  export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=0
  export SGLANG_LTX2_COMPILE_MARK_STEP_BEGIN=0
  export SGLANG_LTX2_COMPILE_PREWARM_PERTURBATION_MASKS=0
  export SGLANG_LTX2_FUSED_QK_ROPE=0
  export SGLANG_LTX2_FUSED_RMS_ADALN=0
  export SGLANG_LTX2_FUSED_ADALN=0
  export SGLANG_LTX2_FUSED_MODULATE=0
  export SGLANG_LTX2_FUSED_RESIDUAL_GATE=0
  export SGLANG_LTX2_FUSED_QKNORM=0
  export SGLANG_LTX2_FUSED_QKNORM_ROPE=0
  export SGLANG_LTX2_FUSED_DUAL_MODULATE=0
  export SGLANG_LTX2_FUSED_CA_DUAL_MODULATE=0
  export SGLANG_LTX2_FUSED_ADA_VALUES=0
  export SGLANG_LTX2_FUSED_ADA_VALUES_ALL=0
  export SGLANG_LTX2_FUSED_ADA_DIRECT=0
  export SGLANG_LTX2_FUSED_Q_GATE=0
  export SGLANG_LTX2_FUSED_QKV=0
  export SGLANG_LTX2_FUSED_AUDIO_QKVG=0
  export SGLANG_LTX2_FUSED_KV=0
  export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=0
  export SGLANG_LTX2_FUSED_GELU_INPLACE=0
  export SGLANG_LTX2_COMPILE_GATE_TO_OUT=0
  export SGLANG_LTX2_COMPILE_A2V_GATE_TO_OUT=0
  export SGLANG_LTX2_COMPILE_VAE_DECODER=0
  export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=0
  export SGLANG_ENABLE_FUSED_QKNORM_ROPE=0
}

enable_kwl_env() {
  disable_kwl_env
  export SGLANG_LTX2_FUSED_QK_ROPE=1
  export SGLANG_LTX2_FUSED_RMS_ADALN=1
  export SGLANG_LTX2_FUSED_ADALN=1
  export SGLANG_LTX2_FUSED_QKNORM_ROPE=1
  export SGLANG_LTX2_FUSED_DUAL_MODULATE=1
  export SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1
  export SGLANG_LTX2_FUSED_RESIDUAL_GATE=1
  export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1
  export SGLANG_LTX2_COMPILE_GATE_TO_OUT=1
  export SGLANG_LTX2_FUSED_AUDIO_QKVG=1
  export SGLANG_ENABLE_FUSED_QKNORM_ROPE=1
  export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER="${SGLANG_NONHQ_KWL_COMPILE_TILED_VAE:-0}"
  export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
}

enable_cache_env() {
  CACHE_ALGO="none"
  case "$VARIANT" in
    cache_pab_late12_w3)
      CACHE_ALGO="pab_attn_late12_w3"
      export SGLANG_LTX2_PAB_ENABLED=1
      export SGLANG_LTX2_PAB_SPATIAL_WINDOW=3
      export SGLANG_LTX2_PAB_TEMPORAL_WINDOW=3
      export SGLANG_LTX2_PAB_CROSS_WINDOW=3
      export SGLANG_LTX2_PAB_START_STEP=12
      export SGLANG_LTX2_PAB_END_STEP=-1
      export SGLANG_LTX2_PAB_DISABLE_AUDIO_VIDEO_CROSS=1
      export SGLANG_LTX2_PAB_A2V_WINDOW=1
      export SGLANG_LTX2_PAB_V2A_WINDOW=1
      export SGLANG_LTX2_PAB_STAGE2_ENABLED=0
      ;;
    cache_teacache_c04_s6|cache_teacache_c06_s5|cache_teacache_c08_s5|kwl_cache_teacache_c04_s6|kwl_cache_teacache_c06_s5|kwl_cache_teacache_c08_s5|kwl_cache_teacache_c04_s6_sparse_piecewise|kwl_cache_teacache_c04_s6_stage2_sparse_piecewise)
      CACHE_ALGO="teacache"
      export SGLANG_LTX2_TEACACHE_ENABLED=1
      export SGLANG_LTX2_TEACACHE_STAGE1_ENABLED="${SGLANG_LTX2_TEACACHE_STAGE1_ENABLED:-1}"
      export SGLANG_LTX2_TEACACHE_END="${SGLANG_LTX2_TEACACHE_END:--1}"
      export SGLANG_LTX2_TEACACHE_STAGE2_DISABLE="${SGLANG_LTX2_TEACACHE_STAGE2_DISABLE:-1}"
      export SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS="${SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS:-1}"
      export SGLANG_LTX2_TEACACHE_PERIODIC_RECOMPUTE_STEPS="${SGLANG_LTX2_TEACACHE_PERIODIC_RECOMPUTE_STEPS:-0}"
      case "$VARIANT" in
        cache_teacache_c04_s6|kwl_cache_teacache_c04_s6|kwl_cache_teacache_c04_s6_sparse_piecewise|kwl_cache_teacache_c04_s6_stage2_sparse_piecewise)
          CACHE_ALGO="teacache_c04_s6"
          export SGLANG_LTX2_TEACACHE_THRESH="${SGLANG_LTX2_TEACACHE_THRESH:-0.04}"
          export SGLANG_LTX2_TEACACHE_START="${SGLANG_LTX2_TEACACHE_START:-6}"
          ;;
        cache_teacache_c06_s5|kwl_cache_teacache_c06_s5)
          CACHE_ALGO="teacache_c06_s5"
          export SGLANG_LTX2_TEACACHE_THRESH="${SGLANG_LTX2_TEACACHE_THRESH:-0.06}"
          export SGLANG_LTX2_TEACACHE_START="${SGLANG_LTX2_TEACACHE_START:-5}"
          ;;
        cache_teacache_c08_s5|kwl_cache_teacache_c08_s5)
          CACHE_ALGO="teacache_c08_s5"
          export SGLANG_LTX2_TEACACHE_THRESH="${SGLANG_LTX2_TEACACHE_THRESH:-0.08}"
          export SGLANG_LTX2_TEACACHE_START="${SGLANG_LTX2_TEACACHE_START:-5}"
          ;;
      esac
      ;;
    cache_dbcache_aggressive)
      CACHE_ALGO="dbcache_aggressive"
      export SGLANG_CACHE_DIT_ENABLED=1
      export SGLANG_CACHE_DIT_WARMUP=4
      export SGLANG_CACHE_DIT_RDT=0.24
      export SGLANG_CACHE_DIT_MC=3
      export SGLANG_CACHE_DIT_FN=1
      export SGLANG_CACHE_DIT_BN=0
      ;;
    *)
      ;;
  esac
}

enable_sparse_env() {
  case "$VARIANT" in
    kwl_cache_teacache_c04_s6_sparse_piecewise)
      SPARSE_ALGO="piecewise_stage1_dense5_ramp80to90_stage2_90_v2v_only"
      export SGLANG_NONHQ_SPARSE_ENABLED=1
      export SGLANG_PIECEWISE_ATTN_SPARSITY=0.9
      export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE=64
      export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF=true
      export SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE=true
      export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS=5
      export SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY=0.8
      export SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY=0.9
      export SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK=sdpa
      COMPONENT_ATTENTION_BACKENDS="transformer=piecewise_attn,transformer_2=piecewise_attn"
      ATTENTION_BACKEND_CONFIG="piecewise_sparsity=0.9,piecewise_block_size=64,piecewise_only_video_self_attention=true,piecewise_stage1_schedule=true,piecewise_stage1_dense_steps=5,piecewise_stage1_start_sparsity=0.8,piecewise_stage1_end_sparsity=0.9,piecewise_dense_fallback=sdpa"
      ;;
    kwl_cache_teacache_c04_s6_stage2_sparse_piecewise)
      SPARSE_ALGO="piecewise_stage2_90_v2v_only"
      export SGLANG_NONHQ_SPARSE_ENABLED=1
      export SGLANG_PIECEWISE_ATTN_SPARSITY=0.9
      export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE=64
      export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF=true
      export SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE=false
      export SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK=sdpa
      COMPONENT_ATTENTION_BACKENDS="transformer=fa,transformer_2=piecewise_attn"
      ATTENTION_BACKEND_CONFIG="piecewise_sparsity=0.9,piecewise_block_size=64,piecewise_only_video_self_attention=true,piecewise_stage1_schedule=false,piecewise_dense_fallback=sdpa"
      ;;
    *)
      COMPONENT_ATTENTION_BACKENDS="${COMPONENT_ATTENTION_BACKENDS:-transformer=fa,transformer_2=fa}"
      ATTENTION_BACKEND_CONFIG="${ATTENTION_BACKEND_CONFIG:-}"
      ;;
  esac
}

clear_cache_env
clear_sparse_env
CACHE_ALGO="none"
if [[ "$VARIANT" == "kwl" || "$VARIANT" == kwl_* ]]; then
  enable_kwl_env
else
  disable_kwl_env
fi
enable_cache_env
enable_sparse_env
ATTENTION_ARGS=(--component-attention-backends "$COMPONENT_ATTENTION_BACKENDS")
if [[ -n "$ATTENTION_BACKEND_CONFIG" ]]; then
  ATTENTION_ARGS+=(--attention-backend-config "$ATTENTION_BACKEND_CONFIG")
fi
OFFLOAD_ARGS=()
if [[ "${SGLANG_LTX2_DIT_CPU_OFFLOAD:-0}" =~ ^(1|true|yes|on)$ ]]; then
  OFFLOAD_ARGS+=(--dit-cpu-offload true)
fi
if [[ "${SGLANG_LTX2_TEXT_ENCODER_CPU_OFFLOAD:-0}" =~ ^(1|true|yes|on)$ ]]; then
  OFFLOAD_ARGS+=(--text-encoder-cpu-offload true)
fi
if [[ "${SGLANG_LTX2_VAE_CPU_OFFLOAD:-0}" =~ ^(1|true|yes|on)$ ]]; then
  OFFLOAD_ARGS+=(--vae-cpu-offload true)
fi
if [[ "${SGLANG_LTX2_DIT_LAYERWISE_OFFLOAD:-0}" =~ ^(1|true|yes|on)$ ]]; then
  OFFLOAD_ARGS+=(--dit-layerwise-offload true)
fi
if [[ -n "${SGLANG_LTX2_LAYERWISE_OFFLOAD_COMPONENTS:-}" ]]; then
  read -r -a layerwise_components <<< "$SGLANG_LTX2_LAYERWISE_OFFLOAD_COMPONENTS"
  OFFLOAD_ARGS+=(--layerwise-offload-components "${layerwise_components[@]}")
fi
if [[ -n "${SGLANG_LTX2_DIT_OFFLOAD_PREFETCH_SIZE:-}" ]]; then
  OFFLOAD_ARGS+=(--dit-offload-prefetch-size "$SGLANG_LTX2_DIT_OFFLOAD_PREFETCH_SIZE")
fi
if [[ -n "${SGLANG_LTX2_PIN_CPU_MEMORY:-}" ]]; then
  OFFLOAD_ARGS+=(--pin-cpu-memory "$SGLANG_LTX2_PIN_CPU_MEMORY")
fi
OFFLOAD_ARG_TEXT="${OFFLOAD_ARGS[*]:-}"
export SGLANG_NONHQ_COMPONENT_ATTENTION_BACKENDS="$COMPONENT_ATTENTION_BACKENDS"
export SGLANG_NONHQ_ATTENTION_BACKEND_CONFIG="$ATTENTION_BACKEND_CONFIG"
export SGLANG_NONHQ_SPARSE_ALGO="$SPARSE_ALGO"

mkdir -p "$OUT_DIR"
if [[ "$FORCE" != "1" && -s "$OUT_VIDEO" && -s "$PERF_JSON" ]]; then
  echo "[skip] $VARIANT prompt=$PROMPT_INDEX already exists: $OUT_VIDEO"
  exit 0
fi

cat > "$OUT_DIR/run_command.txt" <<EOF
SGLANG_NONHQ_VARIANT=$VARIANT PROMPT_INDEX=$PROMPT_INDEX CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
$PYTHON_BIN -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path "$MODEL_PATH" \
  --backend auto \
  --pipeline-class-name LTX2TwoStagePipeline \
  --component-paths.spatial_upsampler "$SPATIAL_UPSAMPLER" \
  --component-paths.distilled_lora "$DISTILLED_LORA" \
  --component-attention-backends "$COMPONENT_ATTENTION_BACKENDS" \
  ${ATTENTION_BACKEND_CONFIG:+--attention-backend-config "$ATTENTION_BACKEND_CONFIG" \
  }--num-gpus 1 \
  --master-port "$MASTER_PORT" \
  --performance-mode "$PERFORMANCE_MODE" \
  --ltx2-two-stage-device-mode "$TWO_STAGE_DEVICE_MODE" \
  $OFFLOAD_ARG_TEXT \
  --warmup "$WARMUP" --warmup-steps "$WARMUP_STEPS" \
  --height "$HEIGHT" --width "$WIDTH" --num-frames "$NUM_FRAMES" --fps "$FPS" --seed "$SEED" \
  --num-inference-steps 30 --guidance-scale 3.0 \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --prompt "$PROMPT" \
  --output-file-path "$OUT_VIDEO" \
  --perf-dump-path "$PERF_JSON" \
  --return-file-paths-only true
EOF

"$PYTHON_BIN" - <<'PYINFO' "$OUT_DIR" "$VARIANT" "$PROMPT_INDEX" "$PROMPT" "$MODEL_PATH" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER" "$CACHE_ALGO" "$SEED"
import json, os, sys
from pathlib import Path
out_dir = Path(sys.argv[1])
summary = {
    "variant": f"sglang_nonhq_{sys.argv[2]}",
    "prompt_index": int(sys.argv[3]),
    "prompt": sys.argv[4],
    "pipeline_class_name": "LTX2TwoStagePipeline",
    "model_path": sys.argv[5],
    "distilled_lora": sys.argv[6],
    "spatial_upsampler": sys.argv[7],
    "seed": int(sys.argv[9]),
    "height": int(os.environ.get("HEIGHT", "1088")),
    "width": int(os.environ.get("WIDTH", "1920")),
    "num_frames": int(os.environ.get("NUM_FRAMES", "241")),
    "fps": int(os.environ.get("FPS", "24")),
    "stage1_steps": 30,
    "stage1_sampler": "euler",
    "performance_mode": os.environ.get("PERFORMANCE_MODE", "speed"),
    "two_stage_device_mode": os.environ.get("TWO_STAGE_DEVICE_MODE", "resident"),
    "dit_cpu_offload": os.environ.get("SGLANG_LTX2_DIT_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes", "on"},
    "text_encoder_cpu_offload": os.environ.get("SGLANG_LTX2_TEXT_ENCODER_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes", "on"},
    "vae_cpu_offload": os.environ.get("SGLANG_LTX2_VAE_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes", "on"},
    "dit_layerwise_offload": os.environ.get("SGLANG_LTX2_DIT_LAYERWISE_OFFLOAD", "0").lower() in {"1", "true", "yes", "on"},
    "pin_cpu_memory": os.environ.get("SGLANG_LTX2_PIN_CPU_MEMORY", "true").lower() not in {"0", "false", "no", "off"},
    "stage2_sigmas": [0.909375, 0.725, 0.421875, 0.0],
    "stage2_steps": 3,
    "stage2_sampler": "euler",
    "stage1_lora_strength": 0.0,
    "stage2_lora_strength": 1.0,
    "video_cfg_scale": 3.0,
    "video_stg_scale": 1.0,
    "video_rescale_scale": 0.7,
    "audio_cfg_scale": 7.0,
    "audio_stg_scale": 1.0,
    "audio_rescale_scale": 0.7,
    "kwl_enabled": sys.argv[2] == "kwl" or sys.argv[2].startswith("kwl_"),
    "cache_enabled": sys.argv[2].startswith("cache_") or "_cache_" in sys.argv[2],
    "cache_algo": sys.argv[8],
    "sparse_attention_enabled": os.environ.get("SGLANG_NONHQ_SPARSE_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
    "sparse_algo": os.environ.get("SGLANG_NONHQ_SPARSE_ALGO", "none"),
    "component_attention_backends": os.environ.get("SGLANG_NONHQ_COMPONENT_ATTENTION_BACKENDS", ""),
    "attention_backend_config": os.environ.get("SGLANG_NONHQ_ATTENTION_BACKEND_CONFIG", ""),
    "piecewise_sparsity": float(os.environ.get("SGLANG_PIECEWISE_ATTN_SPARSITY", "0") or 0),
    "piecewise_block_size": int(os.environ.get("SGLANG_PIECEWISE_ATTN_BLOCK_SIZE", "0") or 0),
    "piecewise_stage1_dense_steps": int(os.environ.get("SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS", "0") or 0),
    "piecewise_stage1_start_sparsity": float(os.environ.get("SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY", "0") or 0),
    "piecewise_stage1_end_sparsity": float(os.environ.get("SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY", "0") or 0),
    "pab_start_step": int(os.environ.get("SGLANG_LTX2_PAB_START_STEP", "-1") or -1),
    "pab_window": int(os.environ.get("SGLANG_LTX2_PAB_SPATIAL_WINDOW", "0") or 0),
    "teacache_thresh": float(os.environ.get("SGLANG_LTX2_TEACACHE_THRESH", "0") or 0),
    "teacache_start": int(os.environ.get("SGLANG_LTX2_TEACACHE_START", "-1") or -1),
    "teacache_end": os.environ.get("SGLANG_LTX2_TEACACHE_END", ""),
    "teacache_stage1_enabled": os.environ.get("SGLANG_LTX2_TEACACHE_STAGE1_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
    "teacache_stage2_enabled": os.environ.get("SGLANG_LTX2_TEACACHE_STAGE2_DISABLE", "1").lower() not in {"1", "true", "yes", "on"},
    "teacache_max_continuous_hits": int(os.environ.get("SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS", "-1") or -1),
    "dbcache_warmup": int(os.environ.get("SGLANG_CACHE_DIT_WARMUP", "0") or 0),
    "dbcache_rdt": float(os.environ.get("SGLANG_CACHE_DIT_RDT", "0") or 0),
    "dbcache_mc": int(os.environ.get("SGLANG_CACHE_DIT_MC", "0") or 0),
}
out_dir.joinpath("nonhq_semantics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PYINFO

echo "[run] SGLang LTX-2.3 non-HQ variant=$VARIANT prompt=$PROMPT_INDEX -> $OUT_VIDEO"
echo "[run] cache_algo=$CACHE_ALGO seed=$SEED"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] command written to $OUT_DIR/run_command.txt"
  exit 0
fi
"$PYTHON_BIN" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path "$MODEL_PATH" \
  --backend auto \
  --pipeline-class-name LTX2TwoStagePipeline \
  --component-paths.spatial_upsampler "$SPATIAL_UPSAMPLER" \
  --component-paths.distilled_lora "$DISTILLED_LORA" \
  "${ATTENTION_ARGS[@]}" \
  --num-gpus 1 \
  --master-port "$MASTER_PORT" \
  --performance-mode "$PERFORMANCE_MODE" \
  --ltx2-two-stage-device-mode "$TWO_STAGE_DEVICE_MODE" \
  "${OFFLOAD_ARGS[@]}" \
  --warmup "$WARMUP" \
  --warmup-steps "$WARMUP_STEPS" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --fps "$FPS" \
  --seed "$SEED" \
  --num-inference-steps 30 \
  --guidance-scale 3.0 \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --prompt "$PROMPT" \
  --output-file-path "$OUT_VIDEO" \
  --perf-dump-path "$PERF_JSON" \
  --return-file-paths-only true
