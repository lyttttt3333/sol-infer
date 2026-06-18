#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -J ltx23-nvfp4-attn-ffn
#SBATCH -o outputs/slurm/ltx23-nvfp4-attn-ffn-%j.out
#SBATCH -e outputs/slurm/ltx23-nvfp4-attn-ffn-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p outputs/slurm

BASE="/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493/transformer"
LORA="/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493/ltx-2.3-22b-distilled-lora-384.safetensors"
OUT1="outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat"
OUT2="outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat"

.conda/ltx23/bin/python python/sglang/multimodal_gen/tools/quantize_ltx2_selective_nvfp4_transformer.py \
  --base-transformer-dir "$BASE" \
  --output-dir "$OUT1" \
  --overwrite

.conda/ltx23/bin/python python/sglang/multimodal_gen/tools/quantize_ltx2_selective_nvfp4_transformer.py \
  --base-transformer-dir "$BASE" \
  --output-dir "$OUT2" \
  --lora-path "$LORA" \
  --lora-key-prefix diffusion_model. \
  --lora-strength 1.0 \
  --overwrite
