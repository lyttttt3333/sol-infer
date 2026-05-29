#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -J ltx23-branch-same-noise-1gpu
#SBATCH -o outputs/slurm/ltx23-branch-same-noise-1gpu-%j.out
#SBATCH -e outputs/slurm/ltx23-branch-same-noise-1gpu-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
export SEQUENTIAL_VARIANTS=1
export ROOT="${ROOT:-outputs/ltx23-branch-baselines-same-noise-1080p10s-1gpu}"
bash scripts/slurm_ltx23_branch_baseline_same_noise_1080p10s.sh
