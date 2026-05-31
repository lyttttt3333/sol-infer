#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH -J ltx23-sglang-hq
#SBATCH -o outputs/slurm/ltx23-sglang-hq-%j.out
#SBATCH -e outputs/slurm/ltx23-sglang-hq-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SGLANG_HQ_VARIANT=dense
exec scripts/run_ltx23_sglang_hq_1080p10s.sh
