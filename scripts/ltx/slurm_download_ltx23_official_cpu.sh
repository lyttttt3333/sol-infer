#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -J ltx23-hf-dl
#SBATCH -o outputs/slurm/ltx23-hf-dl-%j.out
#SBATCH -e outputs/slurm/ltx23-hf-dl-%j.err

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-$PWD/outputs/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/outputs/.cache/huggingface/hub}"
export XDG_CACHE_HOME="$PWD/outputs/.cache/xdg"
export TMPDIR="$PWD/outputs/.tmp"
export PYTHONUNBUFFERED=1

mkdir -p outputs/slurm outputs/LTX-2.3-official-files outputs/.cache/huggingface outputs/.cache/xdg outputs/.tmp

.conda/ltx23/bin/python scripts/ltx/download_ltx23_official_files_cpu.py \
  --output-dir outputs/LTX-2.3-official-files \
  --interval-s 15 \
  ltx-2.3-22b-dev.safetensors \
  ltx-2.3-22b-distilled-lora-384-1.1.safetensors
