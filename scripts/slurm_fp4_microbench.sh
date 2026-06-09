#!/bin/bash
#SBATCH --job-name=fp4-micro
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --output=/home/yitongl/cosmos3-run/fp4-micro-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/fp4-micro-%j.out
set -uo pipefail
REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python
cd "$REPO"; echo "[$(date)] $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
stdbuf -oL -eL "$PYTHON" /home/yitongl/cosmos3-run/fp4bench/microbench.py 2>&1 | grep -vE "UserWarning|warn\(|DeprecationWarning"
echo "[$(date)] microbench done"
