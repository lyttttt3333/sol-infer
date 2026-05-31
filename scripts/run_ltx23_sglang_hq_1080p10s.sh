#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${SGLANG_HQ_VARIANT:-${1:-dense}}"
case "$VARIANT" in
  dense|kwl|kwl_experimental|kwl_sparse|kwl_stage2_sparse|kwl_cache|kwl_sparse_cache|kwl_stage1_cache_core|kwl_stage1_cache_core_stage2_sparse) ;;
  *)
    echo "Usage: SGLANG_HQ_VARIANT=dense|kwl|kwl_experimental|kwl_sparse|kwl_stage2_sparse|kwl_cache|kwl_sparse_cache|kwl_stage1_cache_core|kwl_stage1_cache_core_stage2_sparse $0 [variant]" >&2
    exit 2
    ;;
esac

mkdir -p outputs/slurm outputs/.cache/huggingface outputs/.cache/xdg outputs/.cache/torch outputs/.cache/triton outputs/.cache/torchinductor outputs/.cache/torch_extensions outputs/.cache/cuda outputs/.cache/sgl_diffusion outputs/.tmp

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$PWD/outputs/.cache/huggingface"
export HF_HUB_CACHE="$PWD/outputs/.cache/huggingface/hub"
export XDG_CACHE_HOME="$PWD/outputs/.cache/xdg"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export TRITON_CACHE_DIR="$PWD/outputs/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$PWD/outputs/.cache/torch_extensions"
export CUDA_CACHE_PATH="$PWD/outputs/.cache/cuda"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export SGLANG_DIFFUSION_CACHE_ROOT="$PWD/outputs/.cache/sgl_diffusion"
export TMPDIR="$PWD/outputs/.tmp"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
# LTX-2.3 official selects FlashAttention4 for unmasked DiT attention on B200.
# Keep SGLang's dense baseline on the same attention math path; otherwise
# video-to-audio cross attention falls back to SDPA and drifts from official.
export SGLANG_LTX2_OFFICIAL_FA4_ATTENTION="${SGLANG_LTX2_OFFICIAL_FA4_ATTENTION:-1}"

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
ROOT="${ROOT:-outputs/ltx23-sglang-hq-1080p10s}"
OUT_DIR="${OUT_DIR:-$ROOT/$VARIANT}"
OUT_VIDEO="$OUT_DIR/out.mp4"
PERF_JSON="$OUT_DIR/perf.json"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"
FORCE="${FORCE:-0}"
WARMUP="${WARMUP:-false}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"
MASTER_PORT="${MASTER_PORT:-30005}"

for required in "$PYTHON_BIN" "$MODEL_PATH/model_index.json" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing required SGLang HQ asset: $required" >&2
    exit 1
  fi
done

clear_lossy_env() {
  unset SGLANG_PIECEWISE_ATTN_SPARSITY
  unset SGLANG_PIECEWISE_ATTN_DENSITY
  unset SGLANG_PIECEWISE_ATTN_BLOCK_SIZE
  unset SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF
  unset SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE
  unset SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS
  unset SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY
  unset SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY
  unset SGLANG_PIECEWISE_ATTN_DENSE_LAYERS
  unset SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS
  unset SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS
  unset SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER
  unset SGLANG_PIECEWISE_ATTN_ROUTE_MODE
  unset SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK
  unset SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND
  unset SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND
  export SGLANG_LTX2_PAB_ENABLED=0
  export SGLANG_LTX2_STAGE1_CACHE_CORE_ENABLED=0
  export SGLANG_CACHE_DIT_ENABLED=0
  export SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU=0
  export SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE=0
  export SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE=0
  export SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=0
}

enable_te_nvfp4_video_ffn_env() {
  export SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=1
  export SGLANG_LTX2_TE_NVFP4_DISABLE_RHT="${SGLANG_LTX2_TE_NVFP4_DISABLE_RHT:-1}"
  export SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING="${SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING:-1}"
  export SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION="${SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION:-1}"
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
  # Strict KWL mode: only enable paths that have been proven not to change
  # the generated frames for this HQ setup. Same-noise ablations showed that
  # fused RMS/AdaLN and fused QKNorm+RoPE are not lossless here; keep those in
  # kwl_experimental only.
  export SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN="${SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN:-0}"
  export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX="${SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX:-0}"
  export SGLANG_LTX2_FUSED_QK_ROPE="${SGLANG_HQ_KWL_FUSED_QK_ROPE:-0}"
  export SGLANG_LTX2_FUSED_RMS_ADALN="${SGLANG_HQ_KWL_FUSED_RMS_ADALN:-0}"
  export SGLANG_LTX2_FUSED_ADALN="${SGLANG_HQ_KWL_FUSED_ADALN:-0}"
  export SGLANG_LTX2_FUSED_QKNORM_ROPE="${SGLANG_HQ_KWL_FUSED_QKNORM_ROPE:-0}"
  export SGLANG_LTX2_FUSED_DUAL_MODULATE="${SGLANG_HQ_KWL_FUSED_DUAL_MODULATE:-0}"
  export SGLANG_LTX2_FUSED_ADA_VALUES_ALL="${SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL:-1}"
  export SGLANG_LTX2_FUSED_RESIDUAL_GATE="${SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE:-0}"
  export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU="${SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU:-0}"
  export SGLANG_LTX2_COMPILE_GATE_TO_OUT="${SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT:-0}"
  export SGLANG_LTX2_FUSED_AUDIO_QKVG="${SGLANG_HQ_KWL_FUSED_AUDIO_QKVG:-0}"
  export SGLANG_ENABLE_FUSED_QKNORM_ROPE="${SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE:-0}"
  export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER="${SGLANG_HQ_KWL_COMPILE_TILED_VAE:-0}"
  export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
}

enable_kwl_experimental_env() {
  disable_kwl_env
  export SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN="${SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN:-1}"
  export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX="${SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX:-1}"
  export SGLANG_LTX2_FUSED_QK_ROPE="${SGLANG_HQ_KWL_FUSED_QK_ROPE:-1}"
  export SGLANG_LTX2_FUSED_RMS_ADALN="${SGLANG_HQ_KWL_FUSED_RMS_ADALN:-1}"
  export SGLANG_LTX2_FUSED_ADALN="${SGLANG_HQ_KWL_FUSED_ADALN:-1}"
  export SGLANG_LTX2_FUSED_QKNORM_ROPE="${SGLANG_HQ_KWL_FUSED_QKNORM_ROPE:-1}"
  export SGLANG_LTX2_FUSED_DUAL_MODULATE="${SGLANG_HQ_KWL_FUSED_DUAL_MODULATE:-1}"
  export SGLANG_LTX2_FUSED_ADA_VALUES_ALL="${SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL:-1}"
  export SGLANG_LTX2_FUSED_RESIDUAL_GATE="${SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE:-1}"
  export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU="${SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU:-1}"
  export SGLANG_LTX2_COMPILE_GATE_TO_OUT="${SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT:-1}"
  export SGLANG_LTX2_FUSED_AUDIO_QKVG="${SGLANG_HQ_KWL_FUSED_AUDIO_QKVG:-1}"
  export SGLANG_ENABLE_FUSED_QKNORM_ROPE="${SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE:-1}"
  export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER="${SGLANG_HQ_KWL_COMPILE_TILED_VAE:-1}"
  export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
}


enable_sparse_env() {
  export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
  export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE="${SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE:-true}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS="${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS:-3}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY:-0.8}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS:-0}"
  export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
  export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"
  COMPONENT_ATTENTION_BACKENDS="${SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS:-transformer=piecewise_attn,transformer_2=piecewise_attn}"
  ATTENTION_BACKEND_CONFIG="piecewise_sparsity=${SGLANG_PIECEWISE_ATTN_SPARSITY},piecewise_block_size=${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE},piecewise_only_video_self_attention=${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF},piecewise_stage1_schedule=${SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE},piecewise_stage1_dense_steps=${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS},piecewise_stage1_start_sparsity=${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY},piecewise_stage1_end_sparsity=${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY},piecewise_dense_layers=${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS},piecewise_approx_remainder=${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER},piecewise_route_mode=${SGLANG_PIECEWISE_ATTN_ROUTE_MODE}"
}

enable_stage2_sparse_env() {
  export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
  export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE=false
  export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS=0
  export SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS:-none}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS:-none}"
  export SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS:-none}"
  export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
  export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"
  export SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK="${SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK:-fa}"
  COMPONENT_ATTENTION_BACKENDS="${SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS:-transformer=fa,transformer_2=piecewise_attn}"
  ATTENTION_BACKEND_CONFIG="piecewise_sparsity=${SGLANG_PIECEWISE_ATTN_SPARSITY},piecewise_block_size=${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE},piecewise_only_video_self_attention=${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF},piecewise_stage1_schedule=false,piecewise_stage1_dense_steps=0,piecewise_stage1_start_sparsity=${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY},piecewise_stage1_end_sparsity=${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY},piecewise_dense_layers=${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS},piecewise_stage1_dense_layers=${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS},piecewise_stage2_dense_layers=${SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS},piecewise_approx_remainder=${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER},piecewise_route_mode=${SGLANG_PIECEWISE_ATTN_ROUTE_MODE},piecewise_dense_fallback=${SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK}"
}

enable_stage1_cache_core_env() {
  CACHE_ALGO="stage1_cache_core"
  export SGLANG_LTX2_STAGE1_CACHE_CORE_ENABLED=1
  export SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET="${SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET:-12of15_delta05_29calls}"
  export SGLANG_LTX2_STAGE1_CACHE_CORE_CACHE_DEVICE="${SGLANG_LTX2_STAGE1_CACHE_CORE_CACHE_DEVICE:-default}"
}


enable_cache_env() {
  CACHE_ALGO="${SGLANG_HQ_CACHE_ALGO:-pab}"
  case "$CACHE_ALGO" in
    pab)
      export SGLANG_LTX2_PAB_ENABLED=1
      export SGLANG_LTX2_PAB_SPATIAL_WINDOW="${SGLANG_LTX2_PAB_SPATIAL_WINDOW:-3}"
      export SGLANG_LTX2_PAB_TEMPORAL_WINDOW="${SGLANG_LTX2_PAB_TEMPORAL_WINDOW:-3}"
      export SGLANG_LTX2_PAB_CROSS_WINDOW="${SGLANG_LTX2_PAB_CROSS_WINDOW:-3}"
      export SGLANG_LTX2_PAB_START_STEP="${SGLANG_LTX2_PAB_START_STEP:-6}"
      export SGLANG_LTX2_PAB_END_STEP="${SGLANG_LTX2_PAB_END_STEP:--1}"
      export SGLANG_LTX2_PAB_DISABLE_AUDIO_VIDEO_CROSS="${SGLANG_LTX2_PAB_DISABLE_AUDIO_VIDEO_CROSS:-1}"
      export SGLANG_LTX2_PAB_A2V_WINDOW="${SGLANG_LTX2_PAB_A2V_WINDOW:-1}"
      export SGLANG_LTX2_PAB_V2A_WINDOW="${SGLANG_LTX2_PAB_V2A_WINDOW:-1}"
      export SGLANG_LTX2_PAB_STAGE2_ENABLED="${SGLANG_LTX2_PAB_STAGE2_ENABLED:-0}"
      export SGLANG_LTX2_PAB_STAGE2_SPATIAL_WINDOW="${SGLANG_LTX2_PAB_STAGE2_SPATIAL_WINDOW:-3}"
      export SGLANG_LTX2_PAB_STAGE2_TEMPORAL_WINDOW="${SGLANG_LTX2_PAB_STAGE2_TEMPORAL_WINDOW:-3}"
      export SGLANG_LTX2_PAB_STAGE2_CROSS_WINDOW="${SGLANG_LTX2_PAB_STAGE2_CROSS_WINDOW:-3}"
      export SGLANG_LTX2_PAB_STAGE2_START_STEP="${SGLANG_LTX2_PAB_STAGE2_START_STEP:-0}"
      export SGLANG_LTX2_PAB_STAGE2_END_STEP="${SGLANG_LTX2_PAB_STAGE2_END_STEP:--1}"
      ;;
    dbcache)
      export SGLANG_CACHE_DIT_ENABLED=1
      export SGLANG_CACHE_DIT_WARMUP="${SGLANG_CACHE_DIT_WARMUP:-4}"
      export SGLANG_CACHE_DIT_RDT="${SGLANG_CACHE_DIT_RDT:-0.24}"
      export SGLANG_CACHE_DIT_MC="${SGLANG_CACHE_DIT_MC:-3}"
      export SGLANG_CACHE_DIT_FN="${SGLANG_CACHE_DIT_FN:-1}"
      export SGLANG_CACHE_DIT_BN="${SGLANG_CACHE_DIT_BN:-0}"
      ;;
    none)
      ;;
    *)
      echo "[error] unsupported SGLANG_HQ_CACHE_ALGO=$CACHE_ALGO; use pab|dbcache|none" >&2
      exit 2
      ;;
  esac
}

clear_lossy_env
COMPONENT_ATTENTION_BACKENDS=""
ATTENTION_BACKEND_CONFIG=""
CACHE_ALGO="none"
if [[ "$VARIANT" == kwl_experimental* ]]; then
  enable_kwl_experimental_env
elif [[ "$VARIANT" == kwl* ]]; then
  enable_kwl_env
else
  disable_kwl_env
fi
if [[ "$VARIANT" == "kwl_stage2_sparse" || "$VARIANT" == "kwl_stage1_cache_core_stage2_sparse" ]]; then
  enable_stage2_sparse_env
elif [[ "$VARIANT" == *sparse* ]]; then
  enable_sparse_env
fi
if [[ "$VARIANT" == "kwl_cache" || "$VARIANT" == "kwl_sparse_cache" ]]; then
  enable_cache_env
fi
if [[ "$VARIANT" == "kwl_stage1_cache_core" || "$VARIANT" == "kwl_stage1_cache_core_stage2_sparse" ]]; then
  enable_stage1_cache_core_env
fi
if [[ "${SGLANG_HQ_ENABLE_TE_NVFP4_FFN:-0}" == "1" ]]; then
  enable_te_nvfp4_video_ffn_env
fi
if [[ -z "$COMPONENT_ATTENTION_BACKENDS" && -n "${SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS:-}" ]]; then
  COMPONENT_ATTENTION_BACKENDS="$SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS"
fi
if [[ -z "$ATTENTION_BACKEND_CONFIG" && -n "${SGLANG_HQ_ATTENTION_BACKEND_CONFIG:-}" ]]; then
  ATTENTION_BACKEND_CONFIG="$SGLANG_HQ_ATTENTION_BACKEND_CONFIG"
fi
EXTRA_GENERATE_ARGS=()
if [[ -n "$COMPONENT_ATTENTION_BACKENDS" ]]; then
  EXTRA_GENERATE_ARGS+=(--component-attention-backends "$COMPONENT_ATTENTION_BACKENDS")
fi
if [[ -n "$ATTENTION_BACKEND_CONFIG" ]]; then
  EXTRA_GENERATE_ARGS+=(--attention-backend-config "$ATTENTION_BACKEND_CONFIG")
fi
EXTRA_ARG_TEXT="${EXTRA_GENERATE_ARGS[*]:-}"

mkdir -p "$OUT_DIR"
if [[ "$FORCE" != "1" && -s "$OUT_VIDEO" && -s "$PERF_JSON" ]]; then
  echo "[skip] $VARIANT already exists: $OUT_VIDEO"
  exit 0
fi

cat > "$OUT_DIR/run_command.txt" <<EOF
SGLANG_HQ_VARIANT=$VARIANT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \\
$PYTHON_BIN -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \\
  --model-path "$MODEL_PATH" \\
  --backend auto \\
  $EXTRA_ARG_TEXT \\
  --pipeline-class-name LTX2TwoStageHQPipeline \\
  --component-paths.spatial_upsampler "$SPATIAL_UPSAMPLER" \\
  --component-paths.distilled_lora "$DISTILLED_LORA" \\
  --num-gpus 1 \\
  --master-port "$MASTER_PORT" \\
  --performance-mode speed \\
  --ltx2-two-stage-device-mode resident \\
  --warmup "$WARMUP" --warmup-steps "$WARMUP_STEPS" \\
  --height 1088 --width 1920 --num-frames 241 --fps 24 --seed 42 \\
  --num-inference-steps 15 --guidance-scale 3.0 \\
  --negative-prompt "$NEGATIVE_PROMPT" \\
  --prompt "$PROMPT" \\
  --output-file-path "$OUT_VIDEO" \\
  --perf-dump-path "$PERF_JSON" \\
  --return-file-paths-only true
EOF

"$PYTHON_BIN" - <<'PYINFO' "$OUT_DIR" "$VARIANT" "$MODEL_PATH" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER" "$CACHE_ALGO" "$COMPONENT_ATTENTION_BACKENDS" "$ATTENTION_BACKEND_CONFIG"
import json
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])
summary = {
    "variant": f"sglang_hq_{sys.argv[2]}",
    "pipeline_class_name": "LTX2TwoStageHQPipeline",
    "model_path": sys.argv[3],
    "distilled_lora": sys.argv[4],
    "spatial_upsampler": sys.argv[5],
    "stage1_steps": 15,
    "stage1_sampler": "res2s",
    "stage2_sigmas": [0.909375, 0.725, 0.421875, 0.0],
    "stage2_steps": 3,
    "stage2_sampler": "res2s",
    "stage1_lora_strength": 0.25,
    "stage2_lora_strength": 0.5,
    "video_cfg_scale": 3.0,
    "video_stg_scale": 0.0,
    "video_rescale_scale": 0.45,
    "audio_cfg_scale": 7.0,
    "audio_stg_scale": 0.0,
    "audio_rescale_scale": 1.0,
    "lossy_sparse_attention": "sparse" in sys.argv[2],
    "cache_enabled": "cache" in sys.argv[2],
    "cache_algo": sys.argv[6] if "cache" in sys.argv[2] else "none",
    "component_attention_backends": sys.argv[7],
    "attention_backend_config": sys.argv[8],
    "sparse_stage1_dense_steps": int(__import__("os").environ.get("SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS", "0") or 0),
    "sparse_dense_layers": __import__("os").environ.get("SGLANG_PIECEWISE_ATTN_DENSE_LAYERS", ""),
    "pab_start_step": int(__import__("os").environ.get("SGLANG_LTX2_PAB_START_STEP", "-1") or -1),
    "pab_end_step": __import__("os").environ.get("SGLANG_LTX2_PAB_END_STEP", ""),
    "pab_spatial_window": int(__import__("os").environ.get("SGLANG_LTX2_PAB_SPATIAL_WINDOW", "0") or 0),
    "pab_temporal_window": int(__import__("os").environ.get("SGLANG_LTX2_PAB_TEMPORAL_WINDOW", "0") or 0),
    "pab_cross_window": int(__import__("os").environ.get("SGLANG_LTX2_PAB_CROSS_WINDOW", "0") or 0),
    "pab_stage2_enabled": __import__("os").environ.get("SGLANG_LTX2_PAB_STAGE2_ENABLED", "0") in ("1", "true", "yes", "on"),
    "pab_stage2_start_step": int(__import__("os").environ.get("SGLANG_LTX2_PAB_STAGE2_START_STEP", "-1") or -1),
    "pab_stage2_end_step": __import__("os").environ.get("SGLANG_LTX2_PAB_STAGE2_END_STEP", ""),
    "stage1_cache_core_enabled": __import__("os").environ.get("SGLANG_LTX2_STAGE1_CACHE_CORE_ENABLED", "0") in ("1", "true", "yes", "on"),
    "stage1_cache_core_preset": __import__("os").environ.get("SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET", ""),
    "stage1_cache_core_expected_calls": __import__("os").environ.get("SGLANG_LTX2_STAGE1_CACHE_CORE_EXPECTED_CALLS", ""),
    "stage1_cache_core_skip_indices": __import__("os").environ.get("SGLANG_LTX2_STAGE1_CACHE_CORE_SKIP_INDICES", ""),
    "share_block0_self_attn": __import__("os").environ.get("SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN", "0") in ("1", "true", "yes", "on"),
    "share_guidance_prefix": __import__("os").environ.get("SGLANG_LTX2_SHARE_GUIDANCE_PREFIX", "0") in ("1", "true", "yes", "on"),
    "kwl_flags": {
        "fused_qk_rope": __import__("os").environ.get("SGLANG_LTX2_FUSED_QK_ROPE", "0"),
        "fused_rms_adaln": __import__("os").environ.get("SGLANG_LTX2_FUSED_RMS_ADALN", "0"),
        "fused_adaln": __import__("os").environ.get("SGLANG_LTX2_FUSED_ADALN", "0"),
        "fused_qknorm_rope": __import__("os").environ.get("SGLANG_LTX2_FUSED_QKNORM_ROPE", "0"),
        "fused_dual_modulate": __import__("os").environ.get("SGLANG_LTX2_FUSED_DUAL_MODULATE", "0"),
        "fused_ada_values_all": __import__("os").environ.get("SGLANG_LTX2_FUSED_ADA_VALUES_ALL", "0"),
        "fused_residual_gate": __import__("os").environ.get("SGLANG_LTX2_FUSED_RESIDUAL_GATE", "0"),
        "fused_ffn_proj_in_gelu": __import__("os").environ.get("SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU", "0"),
        "compile_gate_to_out": __import__("os").environ.get("SGLANG_LTX2_COMPILE_GATE_TO_OUT", "0"),
        "fused_audio_qkvg": __import__("os").environ.get("SGLANG_LTX2_FUSED_AUDIO_QKVG", "0"),
        "enable_fused_qknorm_rope": __import__("os").environ.get("SGLANG_ENABLE_FUSED_QKNORM_ROPE", "0"),
        "compile_tiled_vae_decoder": __import__("os").environ.get("SGLANG_LTX2_COMPILE_TILED_VAE_DECODER", "0"),
    },
    "lossy_nvfp4_fp4": __import__("os").environ.get("SGLANG_LTX2_TE_NVFP4_VIDEO_FFN", "0") in ("1", "true", "yes", "on"),
    "te_nvfp4_video_ffn_enabled": __import__("os").environ.get("SGLANG_LTX2_TE_NVFP4_VIDEO_FFN", "0") in ("1", "true", "yes", "on"),
    "te_nvfp4_video_ffn_layers": ["transformer_blocks.*.ff.proj_in", "transformer_blocks.*.ff.proj_out"],
    "te_nvfp4_recipe": {
        "disable_rht": __import__("os").environ.get("SGLANG_LTX2_TE_NVFP4_DISABLE_RHT", ""),
        "disable_stochastic_rounding": __import__("os").environ.get("SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", ""),
        "disable_2d_quantization": __import__("os").environ.get("SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION", ""),
    },
}
out_dir.joinpath("hq_semantics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PYINFO

echo "[run] SGLang LTX-2.3 HQ variant=$VARIANT -> $OUT_VIDEO"
echo "[run] model=$MODEL_PATH"
echo "[run] lora=$DISTILLED_LORA"
echo "[run] upsampler=$SPATIAL_UPSAMPLER"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] command written to $OUT_DIR/run_command.txt"
  exit 0
fi
"$PYTHON_BIN" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path "$MODEL_PATH" \
  --backend auto \
  $EXTRA_ARG_TEXT \
  --pipeline-class-name LTX2TwoStageHQPipeline \
  --component-paths.spatial_upsampler "$SPATIAL_UPSAMPLER" \
  --component-paths.distilled_lora "$DISTILLED_LORA" \
  --num-gpus 1 \
  --master-port "$MASTER_PORT" \
  --performance-mode speed \
  --ltx2-two-stage-device-mode resident \
  --warmup "$WARMUP" \
  --warmup-steps "$WARMUP_STEPS" \
  --height 1088 \
  --width 1920 \
  --num-frames 241 \
  --fps 24 \
  --seed 42 \
  --num-inference-steps 15 \
  --guidance-scale 3.0 \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --prompt "$PROMPT" \
  --output-file-path "$OUT_VIDEO" \
  --perf-dump-path "$PERF_JSON" \
  --return-file-paths-only true
