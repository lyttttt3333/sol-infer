#!/bin/bash
#SBATCH --job-name=sana-video-dl
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yitongl/sana_video/logs/download-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/download-%j.out

# ---------------------------------------------------------------------------
# Download Efficient-Large-Model/SANA-Video_2B_480p_diffusers on the dedicated
# data-mover CPU partition (cpu_datamover). Writes to /home (the nvr_elm_llm
# project lustre quota 10120 spans fs1+fsw and is near-full; /home is a
# separate filer with ~100T free). Shares HF_HOME with the GPU run script.
# ---------------------------------------------------------------------------

set -uo pipefail

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
# The Xet CAS stream stalled indefinitely on a large shard (21+ min, 0 B/s) on a
# first attempt. Force the classic LFS CDN (resumes the .incomplete via Range)
# and fail fast on a hung connection so the built-in retry actually fires.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
PYTHON=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/.conda/ltx23/bin/python

REPO=${1:-Efficient-Large-Model/SANA-Video_2B_480p_diffusers}

mkdir -p /home/yitongl/sana_video/logs "$HF_HUB_CACHE"

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] Python: $PYTHON"
echo "[$(date)] Downloading $REPO (max_workers=16) -> $HF_HUB_CACHE"

if $PYTHON - "$REPO" << 'PYEOF'
import os, sys
from huggingface_hub import snapshot_download

repo = sys.argv[1]
path = snapshot_download(
    repo,
    token=os.environ.get("HF_TOKEN"),
    cache_dir=os.environ["HF_HUB_CACHE"],
    max_workers=16,
)
print("Snapshot path:", path, flush=True)
total = 0
print("---- files ----", flush=True)
for root, _, files in os.walk(path):
    for f in sorted(files):
        fp = os.path.join(root, f)
        try:
            sz = os.path.getsize(fp)  # resolves symlink -> real blob size
        except OSError:
            sz = 0
        total += sz
        print(f"  {sz/1e6:10.1f} MB  {os.path.relpath(fp, path)}", flush=True)
print(f"---- total: {total/1e9:.2f} GB ----", flush=True)
PYEOF
then
    echo "[$(date)] du of cache entry:"
    du -sh "$HF_HUB_CACHE/models--$(echo "$REPO" | sed 's:/:--:g')" 2>/dev/null
    echo "[$(date)] DOWNLOAD_COMPLETE_MARKER"
else
    rc=$?
    echo "[$(date)] DOWNLOAD_FAILED_MARKER rc=$rc"
    exit $rc
fi
