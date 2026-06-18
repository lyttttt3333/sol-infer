#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -J te-nvfp4-pad-fast
#SBATCH -o outputs/ltx23-te-nvfp4-linear-select-padded-fast/slurm-%j.out
#SBATCH -e outputs/ltx23-te-nvfp4-linear-select-padded-fast/slurm-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/scripts:$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export NVTE_FRAMEWORK=pytorch

mkdir -p outputs/ltx23-te-nvfp4-linear-select-padded-fast

.conda/ltx23/bin/python scripts/ltx/bench_te_nvfp4_linear_select_padded.py \
  --out outputs/ltx23-te-nvfp4-linear-select-padded-fast/result.json \
  --repeats 5 \
  --warmup 3 \
  --disable-rht \
  --disable-stochastic-rounding \
  --disable-2d-quantization
