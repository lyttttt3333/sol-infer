#!/bin/bash
#SBATCH --job-name=vbench-sana-score
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home/yitongl/sana_video/logs/vbench-score-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/vbench-score-%j.out

# Multi-node VBench scoring of the SANA batches. Data-parallel over (batch,dim)
# pairs (2 batches x 11 dims = 22). Uses the dedicated VBench env (NOT ltx23);
# dimension models are bundled in that env's site-packages/vbench/.
set -uo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VBENCH_BATCHES=${VBENCH_BATCHES:-sana_dense,sana_fullopt}
export VBENCH_NAMED_ROOT=${VBENCH_NAMED_ROOT:-/home/yitongl/code/vbench_sana/named}
export VBENCH_SCORES_ROOT=${VBENCH_SCORES_ROOT:-/home/yitongl/code/vbench_sana/scores}
# do NOT set CUDA_VISIBLE_DEVICES here (per-task via SLURM_LOCALID in the script)

PY=/home/yitongl/envs/vbench/bin/python
mkdir -p /home/yitongl/sana_video/logs "$VBENCH_SCORES_ROOT"
echo "[$(date)] scoring batches=$VBENCH_BATCHES dims=${VBENCH_DIMS:-<11 default>} nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS"

srun --kill-on-bad-exit=0 $PY scripts/sana/score_vbench_sana.py
rc=$?
echo "[$(date)] EXIT_RC=$rc"
exit $rc
