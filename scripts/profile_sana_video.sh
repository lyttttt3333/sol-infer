#!/bin/bash
#SBATCH --job-name=sana-video-prof
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/home/yitongl/sana_video/logs/profile-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/profile-%j.out

# Stage-level time profile for SANA-Video 480p and 720p (one GPU allocation,
# two fresh python processes). diffusers 0.38.0; warmup-excluded.

set -uo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/home/yitongl/.cache/xdg
export TMPDIR=/home/yitongl/sana_video/.tmp
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
mkdir -p /home/yitongl/sana_video/logs /home/yitongl/sana_video/outputs "$TMPDIR" "$XDG_CACHE_HOME"

PY=.conda/ltx23/bin/python
echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1

$PY scripts/profile_sana_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers --width 832 --height 480 --label SANA-480p
$PY scripts/profile_sana_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_720p_diffusers --width 1280 --height 704 --vae-tiling --label SANA-720p

echo "[$(date)] PROFILE_ALL_DONE"
