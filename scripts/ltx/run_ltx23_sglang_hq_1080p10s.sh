#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Two launch modes only — keep it unambiguous:
#   baseline : official two-stage, no acceleration (dense reference)
#   fullopt  : the full acceleration stack (2.47x; see enable_fullopt_env below)
# `dense` is accepted as a silent alias of `baseline` for back-compat. Advanced
# users who want to ablate individual techniques can set the underlying
# SGLANG_LTX2_* / SGLANG_PIECEWISE_ATTN_* env vars directly on top of baseline.
VARIANT="${SGLANG_HQ_VARIANT:-${1:-baseline}}"
case "$VARIANT" in
  baseline|fullopt) ;;
  dense) VARIANT="baseline" ;;
  *)
    echo "Usage: $0 [baseline|fullopt]   (or SGLANG_HQ_VARIANT=baseline|fullopt)" >&2
    exit 2
    ;;
esac

mkdir -p outputs/slurm outputs/.cache/huggingface outputs/.cache/xdg outputs/.cache/torch outputs/.cache/triton outputs/.cache/torchinductor outputs/.cache/torch_extensions outputs/.cache/cuda outputs/.cache/sgl_diffusion outputs/.tmp

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Honor a pre-set HF_HOME/HF_HUB_CACHE so a deployer can point at an existing
# weight cache; otherwise default to a repo-local cache.
export HF_HOME="${HF_HOME:-$PWD/outputs/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/outputs/.cache/huggingface/hub}"
export XDG_CACHE_HOME="$PWD/outputs/.cache/xdg"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export TRITON_CACHE_DIR="$PWD/outputs/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$PWD/outputs/.cache/torch_extensions"
export CUDA_CACHE_PATH="$PWD/outputs/.cache/cuda"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export SGLANG_DIFFUSION_CACHE_ROOT="$PWD/outputs/.cache/sgl_diffusion"
export TMPDIR="${SGLANG_HQ_TMPDIR:-$PWD/outputs/.tmp}"
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
# distilled LoRA: prefer the -1.1 filename, fall back to the un-suffixed one
# (different HF snapshots ship one or the other).
if [[ -z "${DISTILLED_LORA:-}" ]]; then
  DISTILLED_LORA="$OFFICIAL_MODEL_DIR/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
  [[ -e "$DISTILLED_LORA" ]] || DISTILLED_LORA="$OFFICIAL_MODEL_DIR/ltx-2.3-22b-distilled-lora-384.safetensors"
fi
SPATIAL_UPSAMPLER="${SPATIAL_UPSAMPLER:-$MODEL_PATH/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
ROOT="${ROOT:-outputs/ltx23-sglang-hq-1080p10s}"
OUT_DIR="${OUT_DIR:-$ROOT/$VARIANT}"
OUT_VIDEO="$OUT_DIR/out.mp4"
STAGE1_VIDEO="$OUT_DIR/stage1_out.mp4"
PERF_JSON="$OUT_DIR/perf.json"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
SEED="${SEED:-42}"
SAVE_STAGE1_OUTPUT="${SAVE_STAGE1_OUTPUT:-0}"
STAGE1_ONLY_OUTPUT="${STAGE1_ONLY_OUTPUT:-0}"
FORCE="${FORCE:-0}"
WARMUP="${WARMUP:-false}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"
MASTER_PORT="${MASTER_PORT:-30005}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-${SGLANG_LTX2_PERFORMANCE_MODE:-speed}}"
TWO_STAGE_DEVICE_MODE="${TWO_STAGE_DEVICE_MODE:-${SGLANG_LTX2_TWO_STAGE_DEVICE_MODE:-resident}}"
export PERFORMANCE_MODE TWO_STAGE_DEVICE_MODE
export SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_1="${SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_1:-${SGLANG_LTX2_STAGE1_DISTILLED_LORA_STRENGTH:-0.25}}"
export SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_2="${SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_2:-${SGLANG_LTX2_STAGE2_DISTILLED_LORA_STRENGTH:-0.5}}"

if [[ "$STAGE1_ONLY_OUTPUT" =~ ^(1|true|yes|on)$ ]]; then
  OUT_VIDEO="$STAGE1_VIDEO"
  PERF_JSON="$OUT_DIR/stage1_perf.json"
fi

# When MODEL_PATH is an HF repo id (overlay/native flow) rather than a local
# materialized dir, skip the local model_index.json check and let the runtime
# resolve + materialize it.
required_assets=("$PYTHON_BIN" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER")
[[ -d "$MODEL_PATH" ]] && required_assets+=("$MODEL_PATH/model_index.json")
for required in "${required_assets[@]}"; do
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
  export SGLANG_LTX2_TEACACHE_ENABLED=0
  unset SGLANG_LTX2_TEACACHE_THRESH
  unset SGLANG_LTX2_TEACACHE_START
  unset SGLANG_LTX2_TEACACHE_END
  unset SGLANG_LTX2_TEACACHE_STAGE1_ENABLED
  unset SGLANG_LTX2_TEACACHE_STAGE2_DISABLE
  unset SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS
  unset SGLANG_LTX2_TEACACHE_PERIODIC_RECOMPUTE_STEPS
  export SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU=0
  export SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE=0
  export SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE=0
  export SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=0
  export SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU=0
  export SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE=0
  unset SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_RATIO
  unset SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_METHOD
  unset SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_STEPS
}

enable_te_nvfp4_video_ffn_env() {
  export SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=1
  export SGLANG_LTX2_TE_NVFP4_DISABLE_RHT="${SGLANG_LTX2_TE_NVFP4_DISABLE_RHT:-1}"
  export SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING="${SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING:-1}"
  export SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION="${SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION:-1}"
  export SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU="${SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU:-0}"
  export SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE="${SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE:-0}"
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
  export SGLANG_LTX2_COMPILE_GATE_TO_OUT_RESIDUAL=0
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
  export SGLANG_LTX2_FUSED_CA_DUAL_MODULATE="${SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE:-0}"
  export SGLANG_LTX2_FUSED_ADA_VALUES_ALL="${SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL:-1}"
  export SGLANG_LTX2_FUSED_RESIDUAL_GATE="${SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE:-0}"
  export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU="${SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU:-0}"
  export SGLANG_LTX2_COMPILE_GATE_TO_OUT="${SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT:-0}"
  export SGLANG_LTX2_COMPILE_GATE_TO_OUT_RESIDUAL="${SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT_RESIDUAL:-0}"
  export SGLANG_LTX2_FUSED_AUDIO_QKVG="${SGLANG_HQ_KWL_FUSED_AUDIO_QKVG:-0}"
  export SGLANG_ENABLE_FUSED_QKNORM_ROPE="${SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE:-0}"
  export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER="${SGLANG_HQ_KWL_COMPILE_TILED_VAE:-0}"
  export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
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

# The single full acceleration stack (the validated 2.47x config). Self-contained
# so `fullopt` needs no extra env: KWL operator fusion (lossless) + stage-1 SCSP
# step-skip + stage-2 PISA sparse attention + NVFP4 video FFN + stage-2 midpoint
# token-prune. Every knob below is overridable for advanced ablation, but the
# defaults ARE the shipped fullopt.
enable_fullopt_env() {
  # 1. KWL operator fusion (algorithm-lossless kernel fusions + compile)
  : "${SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN:=1}"
  : "${SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX:=1}"
  : "${SGLANG_HQ_KWL_FUSED_QK_ROPE:=1}"
  : "${SGLANG_HQ_KWL_FUSED_RMS_ADALN:=1}"
  : "${SGLANG_HQ_KWL_FUSED_ADALN:=1}"
  : "${SGLANG_HQ_KWL_FUSED_QKNORM_ROPE:=1}"
  : "${SGLANG_HQ_KWL_FUSED_DUAL_MODULATE:=1}"
  : "${SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE:=1}"
  : "${SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL:=1}"
  : "${SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE:=1}"
  : "${SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU:=1}"
  : "${SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT:=1}"
  : "${SGLANG_HQ_KWL_FUSED_AUDIO_QKVG:=1}"
  : "${SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE:=1}"
  : "${SGLANG_HQ_KWL_COMPILE_TILED_VAE:=1}"
  export SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX \
    SGLANG_HQ_KWL_FUSED_QK_ROPE SGLANG_HQ_KWL_FUSED_RMS_ADALN SGLANG_HQ_KWL_FUSED_ADALN \
    SGLANG_HQ_KWL_FUSED_QKNORM_ROPE SGLANG_HQ_KWL_FUSED_DUAL_MODULATE \
    SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL \
    SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU \
    SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT SGLANG_HQ_KWL_FUSED_AUDIO_QKVG \
    SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE SGLANG_HQ_KWL_COMPILE_TILED_VAE
  enable_kwl_env
  # 2. stage-1 SCSP step-skip cache (replaces TeaCache; TeaCache is unused here)
  export SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET="${SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET:-8of15_last_29calls}"
  enable_stage1_cache_core_env
  # 3. stage-2 PISA piecewise sparse attention (transformer_2 only, dense layers 0-1)
  export SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS:-0-1}"
  enable_stage2_sparse_env
  # 4. NVFP4 video FFN (load-time FP4 quant)
  enable_te_nvfp4_video_ffn_env
  # 5. stage-2 midpoint token-prune (keep 50% of video tokens at refine steps 1-2)
  export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_RATIO="${SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_RATIO:-0.5}"
  export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_METHOD="${SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_METHOD:-feat_norm}"
  export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_STEPS="${SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_STEPS:-1,2}"
  # validated-run extras
  export SGLANG_LTX2_PREPROJECT_PROMPTS="${SGLANG_LTX2_PREPROJECT_PROMPTS:-1}"
  export SGLANG_LTX2_CACHE_ROPE_EMB="${SGLANG_LTX2_CACHE_ROPE_EMB:-1}"
}

clear_lossy_env
COMPONENT_ATTENTION_BACKENDS=""
ATTENTION_BACKEND_CONFIG=""
CACHE_ALGO="none"
if [[ "$VARIANT" == "fullopt" ]]; then
  enable_fullopt_env
else
  disable_kwl_env   # baseline: official two-stage, no acceleration
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

mkdir -p "$OUT_DIR"
if [[ "$FORCE" != "1" && -s "$OUT_VIDEO" && -s "$PERF_JSON" ]]; then
  if [[ "$SAVE_STAGE1_OUTPUT" =~ ^(1|true|yes|on)$ && ! -s "$STAGE1_VIDEO" ]]; then
    :
  else
    echo "[skip] $VARIANT already exists: $OUT_VIDEO"
    exit 0
  fi
fi

if [[ "$STAGE1_ONLY_OUTPUT" =~ ^(1|true|yes|on)$ ]]; then
  export SGLANG_LTX2_STAGE1_ONLY_OUTPUT=1
  unset SGLANG_LTX2_SAVE_STAGE1_OUTPUT
  unset SGLANG_LTX2_STAGE1_OUTPUT_PATH
elif [[ "$SAVE_STAGE1_OUTPUT" =~ ^(1|true|yes|on)$ ]]; then
  unset SGLANG_LTX2_STAGE1_ONLY_OUTPUT
  export SGLANG_LTX2_SAVE_STAGE1_OUTPUT=1
  export SGLANG_LTX2_STAGE1_OUTPUT_PATH="$STAGE1_VIDEO"
else
  unset SGLANG_LTX2_STAGE1_ONLY_OUTPUT
  unset SGLANG_LTX2_SAVE_STAGE1_OUTPUT
  unset SGLANG_LTX2_STAGE1_OUTPUT_PATH
fi

cat > "$OUT_DIR/run_command.txt" <<EOF
SGLANG_HQ_VARIANT=$VARIANT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \\
SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_1=$SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_1 \\
SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_2=$SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_2 \\
$PYTHON_BIN -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \\
  --model-path "$MODEL_PATH" \\
  --backend auto \\
  $EXTRA_ARG_TEXT \\
  --pipeline-class-name LTX2TwoStageHQPipeline \\
  --component-paths.spatial_upsampler "$SPATIAL_UPSAMPLER" \\
  --component-paths.distilled_lora "$DISTILLED_LORA" \\
  --num-gpus 1 \\
  --master-port "$MASTER_PORT" \\
  --performance-mode "$PERFORMANCE_MODE" \\
  --ltx2-two-stage-device-mode "$TWO_STAGE_DEVICE_MODE" \\
  $OFFLOAD_ARG_TEXT \\
  --warmup "$WARMUP" --warmup-steps "$WARMUP_STEPS" \\
  --height 1088 --width 1920 --num-frames 241 --fps 24 --seed "$SEED" \\
  --num-inference-steps 15 --guidance-scale 3.0 \\
  --negative-prompt "$NEGATIVE_PROMPT" \\
  --prompt "$PROMPT" \\
  --output-file-path "$OUT_VIDEO" \\
  --perf-dump-path "$PERF_JSON" \\
  --return-file-paths-only true
EOF

"$PYTHON_BIN" - <<'PYINFO' "$OUT_DIR" "$VARIANT" "$MODEL_PATH" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER" "$CACHE_ALGO" "$COMPONENT_ATTENTION_BACKENDS" "$ATTENTION_BACKEND_CONFIG"
import json
import os
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])

def float_env(name, default):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else float(default)

summary = {
    "variant": f"sglang_hq_{sys.argv[2]}",
    "prompt_index": int(__import__("os").environ.get("PROMPT_INDEX", "0") or 0),
    "prompt": __import__("os").environ.get("PROMPT", ""),
    "pipeline_class_name": "LTX2TwoStageHQPipeline",
    "model_path": sys.argv[3],
    "distilled_lora": sys.argv[4],
    "spatial_upsampler": sys.argv[5],
    "stage1_steps": 15,
    "stage1_sampler": "res2s",
    "stage2_sigmas": [0.909375, 0.725, 0.421875, 0.0],
    "stage2_steps": 3,
    "stage2_sampler": "res2s",
    "seed": int(os.environ.get("SEED", "42") or 42),
    "stage1_only_output": os.environ.get("STAGE1_ONLY_OUTPUT", "0").lower() in {"1", "true", "yes", "on"},
    "stage1_output_video": str(out_dir / "stage1_out.mp4") if (
        os.environ.get("SAVE_STAGE1_OUTPUT", "0").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("STAGE1_ONLY_OUTPUT", "0").lower() in {"1", "true", "yes", "on"}
    ) else None,
    "performance_mode": os.environ.get("PERFORMANCE_MODE", "speed"),
    "two_stage_device_mode": os.environ.get("TWO_STAGE_DEVICE_MODE", "resident"),
    "dit_cpu_offload": os.environ.get("SGLANG_LTX2_DIT_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes", "on"},
    "text_encoder_cpu_offload": os.environ.get("SGLANG_LTX2_TEXT_ENCODER_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes", "on"},
    "vae_cpu_offload": os.environ.get("SGLANG_LTX2_VAE_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes", "on"},
    "dit_layerwise_offload": os.environ.get("SGLANG_LTX2_DIT_LAYERWISE_OFFLOAD", "0").lower() in {"1", "true", "yes", "on"},
    "pin_cpu_memory": os.environ.get("SGLANG_LTX2_PIN_CPU_MEMORY", "true").lower() not in {"0", "false", "no", "off"},
    "stage1_lora_strength": float_env("SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_1", 0.25),
    "stage2_lora_strength": float_env("SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_2", 0.5),
    "video_cfg_scale": 3.0,
    "video_stg_scale": 0.0,
    "video_rescale_scale": 0.45,
    "audio_cfg_scale": 7.0,
    "audio_stg_scale": 0.0,
    "audio_rescale_scale": 1.0,
    "lossy_sparse_attention": "piecewise" in sys.argv[7],
    "cache_enabled": sys.argv[6] != "none",
    "cache_algo": sys.argv[6],
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
    "teacache_enabled": __import__("os").environ.get("SGLANG_LTX2_TEACACHE_ENABLED", "0") in ("1", "true", "yes", "on"),
    "teacache_thresh": float(__import__("os").environ.get("SGLANG_LTX2_TEACACHE_THRESH", "0") or 0),
    "teacache_start": int(__import__("os").environ.get("SGLANG_LTX2_TEACACHE_START", "-1") or -1),
    "teacache_end": __import__("os").environ.get("SGLANG_LTX2_TEACACHE_END", ""),
    "teacache_stage1_enabled": __import__("os").environ.get("SGLANG_LTX2_TEACACHE_STAGE1_ENABLED", "0") in ("1", "true", "yes", "on"),
    "teacache_stage2_enabled": __import__("os").environ.get("SGLANG_LTX2_TEACACHE_STAGE2_DISABLE", "1") not in ("1", "true", "yes", "on"),
    "teacache_max_continuous_hits": int(__import__("os").environ.get("SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS", "-1") or -1),
    "share_block0_self_attn": __import__("os").environ.get("SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN", "0") in ("1", "true", "yes", "on"),
    "share_guidance_prefix": __import__("os").environ.get("SGLANG_LTX2_SHARE_GUIDANCE_PREFIX", "0") in ("1", "true", "yes", "on"),
    "kwl_flags": {
        "fused_qk_rope": __import__("os").environ.get("SGLANG_LTX2_FUSED_QK_ROPE", "0"),
        "fused_rms_adaln": __import__("os").environ.get("SGLANG_LTX2_FUSED_RMS_ADALN", "0"),
        "fused_adaln": __import__("os").environ.get("SGLANG_LTX2_FUSED_ADALN", "0"),
        "fused_qknorm_rope": __import__("os").environ.get("SGLANG_LTX2_FUSED_QKNORM_ROPE", "0"),
        "fused_dual_modulate": __import__("os").environ.get("SGLANG_LTX2_FUSED_DUAL_MODULATE", "0"),
        "fused_ca_dual_modulate": __import__("os").environ.get("SGLANG_LTX2_FUSED_CA_DUAL_MODULATE", "0"),
        "fused_ada_values_all": __import__("os").environ.get("SGLANG_LTX2_FUSED_ADA_VALUES_ALL", "0"),
        "fused_residual_gate": __import__("os").environ.get("SGLANG_LTX2_FUSED_RESIDUAL_GATE", "0"),
        "fused_ffn_proj_in_gelu": __import__("os").environ.get("SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU", "0"),
        "compile_gate_to_out": __import__("os").environ.get("SGLANG_LTX2_COMPILE_GATE_TO_OUT", "0"),
        "compile_gate_to_out_residual": __import__("os").environ.get("SGLANG_LTX2_COMPILE_GATE_TO_OUT_RESIDUAL", "0"),
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
        "fused_proj_in_gelu": __import__("os").environ.get("SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU", "0"),
        "fused_proj_out_bias_gate": __import__("os").environ.get("SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE", "0"),
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
  --performance-mode "$PERFORMANCE_MODE" \
  --ltx2-two-stage-device-mode "$TWO_STAGE_DEVICE_MODE" \
  "${OFFLOAD_ARGS[@]}" \
  --warmup "$WARMUP" \
  --warmup-steps "$WARMUP_STEPS" \
  --height 1088 \
  --width 1920 \
  --num-frames 241 \
  --fps 24 \
  --seed "$SEED" \
  --num-inference-steps 15 \
  --guidance-scale 3.0 \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --prompt "$PROMPT" \
  --output-file-path "$OUT_VIDEO" \
  --perf-dump-path "$PERF_JSON" \
  --return-file-paths-only true
