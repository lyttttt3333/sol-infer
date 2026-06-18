#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-python}"

export SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=1
export SGLANG_LTX2_FUSED_ADALN=1
export SGLANG_LTX2_FUSED_QKNORM_ROPE=1
export SGLANG_LTX2_FUSED_DUAL_MODULATE=1
export SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1
export SGLANG_LTX2_FUSED_RESIDUAL_GATE=1
export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1
export SGLANG_LTX2_COMPILE_GATE_TO_OUT=1
export SGLANG_LTX2_FUSED_AUDIO_QKVG=1
export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=1
export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=1

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${1:-outputs/ltx23-best-1080p10s-single-gpu}"
PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"

"$PYTHON_BIN" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path Lightricks/LTX-2.3 \
  --backend auto \
  --pipeline-class-name LTX2TwoStagePipeline \
  --num-gpus 1 \
  --performance-mode speed \
  --ltx2-two-stage-device-mode resident \
  --warmup true \
  --warmup-steps 30 \
  --height 1088 \
  --width 1920 \
  --num-frames 241 \
  --fps 24 \
  --seed 42 \
  --num-inference-steps 30 \
  --guidance-scale 3.0 \
  --prompt "$PROMPT" \
  --output-file-path "${OUT_DIR}/out.mp4" \
  --perf-dump-path "${OUT_DIR}/perf.json" \
  --return-file-paths-only true
