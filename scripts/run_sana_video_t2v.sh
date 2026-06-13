#!/bin/bash
#SBATCH --job-name=sana-video-t2v
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/home/yitongl/sana_video/logs/t2v-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/t2v-%j.out

# ---------------------------------------------------------------------------
# SANA-Video 2B text-to-video via diffusers SanaVideoPipeline.
# Uses the existing .conda/ltx23 env (diffusers 0.37.0 ships SanaVideoPipeline,
# transformers 5.8.1, torch 2.11 on GB200/sm_100). Cluster convention: request
# 4 GPUs + --exclusive, but the pipeline is single-GPU so we pin CUDA dev 0.
# Reads the model from the shared /home HF cache populated by the datamover job.
# ---------------------------------------------------------------------------

set -uo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
# Snapshot is fully cached; load local-only so a flaky network call can't stall the run.
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/home/yitongl/.cache/xdg
export TMPDIR=/home/yitongl/sana_video/.tmp
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p /home/yitongl/sana_video/logs /home/yitongl/sana_video/outputs "$TMPDIR" "$XDG_CACHE_HOME"

PYTHON=.conda/ltx23/bin/python

echo "[$(date)] Node: $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1

$PYTHON scripts/run_sana_video_t2v.py "$@"
rc=$?
echo "[$(date)] EXIT_RC=$rc"
exit $rc
