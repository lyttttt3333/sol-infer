#!/bin/bash
#SBATCH --job-name=cosmos3-download
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yitongl/.hf_cache/slurm_download-%j.out
#SBATCH --error=/home/yitongl/.hf_cache/slurm_download-%j.out

set -uo pipefail

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
PYTHON=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/.conda/ltx23/bin/python

echo "[$(date)] Node: $(hostname)  ulimit -v: $(ulimit -v)"
echo "[$(date)] Downloading nvidia/Cosmos3-Nano (max_workers=16)..."

$PYTHON - << 'PYEOF'
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    'nvidia/Cosmos3-Nano',
    token=os.environ['HF_TOKEN'],
    cache_dir=os.environ['HF_HUB_CACHE'],
    ignore_patterns=['*.mp4', '*.png', '*.jpg', 'assets/*', 'images/*'],
    max_workers=16,
)
print('Snapshot path:', path)
PYEOF

echo "[$(date)] Download finished. Total cache size:"
du -sh "$HF_HUB_CACHE/models--nvidia--Cosmos3-Nano" 2>/dev/null
echo "[$(date)] DOWNLOAD_COMPLETE_MARKER"
