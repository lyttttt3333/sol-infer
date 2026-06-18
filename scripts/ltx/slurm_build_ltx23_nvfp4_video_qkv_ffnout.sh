#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -J ltx23-build-qkv-ffnout-fp4
#SBATCH -o outputs/slurm/ltx23-build-qkv-ffnout-fp4-%j.out
#SBATCH -e outputs/slurm/ltx23-build-qkv-ffnout-fp4-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

BASE_MODEL="/home/yitongl/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493"
LORA="$BASE_MODEL/ltx-2.3-22b-distilled-lora-384.safetensors"
STAGE1_OUT="outputs/ltx23-selective-nvfp4-video-qkv-ffnout-transformer-mat"
STAGE2_OUT="outputs/ltx23-selective-nvfp4-video-qkv-ffnout-stage2-lora-transformer-mat"

PATTERNS=(
  --include-pattern 'transformer_blocks.*.ff.net.2'
  --include-pattern 'transformer_blocks.*.attn1.to_q'
  --include-pattern 'transformer_blocks.*.attn1.to_k'
  --include-pattern 'transformer_blocks.*.attn1.to_v'
  --include-pattern 'transformer_blocks.*.attn2.to_q'
  --include-pattern 'transformer_blocks.*.attn2.to_k'
  --include-pattern 'transformer_blocks.*.attn2.to_v'
)

.conda/ltx23/bin/python python/sglang/multimodal_gen/tools/quantize_ltx2_selective_nvfp4_transformer.py \
  --base-transformer-dir "$BASE_MODEL/transformer" \
  --output-dir "$STAGE1_OUT" \
  --overwrite \
  "${PATTERNS[@]}"

.conda/ltx23/bin/python python/sglang/multimodal_gen/tools/quantize_ltx2_selective_nvfp4_transformer.py \
  --base-transformer-dir "$BASE_MODEL/transformer" \
  --output-dir "$STAGE2_OUT" \
  --lora-path "$LORA" \
  --overwrite \
  "${PATTERNS[@]}"
