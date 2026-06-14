#!/bin/bash
#SBATCH --job-name=dm-py
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yitongl/sana_video/logs/dm-py-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/dm-py-%j.out

# Generic CPU-processing runner on the dedicated cpu_datamover node (keeps heavy
# decode/metrics work off the shared login node). Usage:
#   sbatch scripts/slurm_datamover_py.sh <script.py> [args...]
set -uo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
export PYTHONUNBUFFERED=1
PY=.conda/ltx23/bin/python

mkdir -p /home/yitongl/sana_video/logs
echo "[$(date)] node=$(hostname)  run: $PY $*"
$PY "$@"
rc=$?
echo "[$(date)] DM_PY_DONE rc=$rc"
exit $rc
