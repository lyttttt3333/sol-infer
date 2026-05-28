#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -J ltx23-qkv-ffnout-fp4
#SBATCH -o outputs/slurm/ltx23-qkv-ffnout-fp4-%j.out
#SBATCH -e outputs/slurm/ltx23-qkv-ffnout-fp4-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

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
export SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn

PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"
OUT_DIR="${OUT_DIR:-outputs/ltx23-dev-1080p10s-nvfp4-video-qkv-ffnout-best-pipeline}"
mkdir -p outputs/slurm "$OUT_DIR"

.conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
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
  --component-paths.transformer outputs/ltx23-selective-nvfp4-video-qkv-ffnout-transformer-mat \
  --component-paths.transformer_2 outputs/ltx23-selective-nvfp4-video-qkv-ffnout-stage2-lora-transformer-mat \
  --output-file-path "$OUT_DIR/out.mp4" \
  --perf-dump-path "$OUT_DIR/perf.json" \
  --return-file-paths-only true
