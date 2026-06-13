#!/bin/bash
#SBATCH --job-name=sana-video-up
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yitongl/sana_video/logs/upload-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/upload-%j.out

# ---------------------------------------------------------------------------
# Upload the generated SANA-Video sample to the HF dataset yitongl/ltx23-shares.
# Runs on the data-mover CPU partition (has outbound internet; the login node
# does not). Xet disabled to avoid the CAS-stream stall seen on download.
# ---------------------------------------------------------------------------

set -uo pipefail

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
PYTHON=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/.conda/ltx23/bin/python

REPO=yitongl/ltx23-shares
SRC=/home/yitongl/sana_video/outputs/sana_video_t2v.mp4
DST=sana-video/sana_video_t2v.mp4

mkdir -p /home/yitongl/sana_video/logs
echo "[$(date)] Node: $(hostname)"
echo "[$(date)] Uploading $SRC -> dataset:$REPO path:$DST"

if $PYTHON - "$REPO" "$SRC" "$DST" << 'PYEOF'
import os, sys
from huggingface_hub import HfApi

repo, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
assert os.path.exists(src), f"source missing: {src}"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", exist_ok=True)
url = api.upload_file(
    path_or_fileobj=src,
    path_in_repo=dst,
    repo_id=repo,
    repo_type="dataset",
    commit_message="Add SANA-Video 2B T2V sample (480x832, 81 frames, 50 steps)",
)
print("UPLOADED_URL:", url, flush=True)
PYEOF
then
    echo "[$(date)] UPLOAD_COMPLETE_MARKER"
else
    rc=$?
    echo "[$(date)] UPLOAD_FAILED_MARKER rc=$rc"
    exit $rc
fi
