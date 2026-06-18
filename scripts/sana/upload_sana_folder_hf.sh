#!/bin/bash
#SBATCH --job-name=sana-folder-up
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yitongl/sana_video/logs/upload-folder-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/upload-folder-%j.out

# Upload the labeled SANA-Video acceleration sample folder (browsable mp4s +
# MANIFEST) to the HF dataset.
set -uo pipefail

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
PYTHON=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/.conda/ltx23/bin/python

REPO=yitongl/ltx23-shares
SRC=${1:-/home/yitongl/sana_video/hf_samples}
DST=${2:-sana-video-accel}

mkdir -p /home/yitongl/sana_video/logs
echo "[$(date)] Node: $(hostname)  uploading folder $SRC -> dataset:$REPO path:$DST"

if $PYTHON - "$REPO" "$SRC" "$DST" << 'PYEOF'
import os, sys
from huggingface_hub import HfApi

repo, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
assert os.path.isdir(src), f"missing dir: {src}"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", exist_ok=True)
url = api.upload_folder(
    folder_path=src,
    path_in_repo=dst,
    repo_id=repo,
    repo_type="dataset",
    commit_message="Add SANA-Video acceleration ablation samples (labeled: method/speedup/corr) + MANIFEST",
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
