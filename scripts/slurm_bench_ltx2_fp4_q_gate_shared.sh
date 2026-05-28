#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J ltx2-qgate-fp4
#SBATCH -o outputs/ltx23-fp4-q-gate-shared/slurm-%j.out
#SBATCH -e outputs/ltx23-fp4-q-gate-shared/slurm-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/scripts:$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p outputs/ltx23-fp4-q-gate-shared

.conda/ltx23/bin/python scripts/bench_ltx2_fp4_q_gate_shared.py \
  --out outputs/ltx23-fp4-q-gate-shared/result.json \
  --repeats 7 \
  --warmup 5
