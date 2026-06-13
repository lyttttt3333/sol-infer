#!/bin/bash
#SBATCH --job-name=sana-zip-up
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yitongl/sana_video/logs/upload-zip-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/upload-zip-%j.out

# Upload the SANA-Video sglang acceleration-ablation video zip to the HF dataset.
set -uo pipefail

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
PYTHON=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/.conda/ltx23/bin/python

REPO=yitongl/ltx23-shares
SRC=/home/yitongl/sana_video/sana_sglang_accel_videos.zip
DST=sana-video/sana_sglang_accel_videos.zip

mkdir -p /home/yitongl/sana_video/logs
echo "[$(date)] Node: $(hostname)  uploading $SRC -> dataset:$REPO path:$DST"

if $PYTHON - "$REPO" "$SRC" "$DST" << 'PYEOF'
import os, sys
from huggingface_hub import HfApi

repo, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
assert os.path.exists(src), f"missing: {src}"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", exist_ok=True)
url = api.upload_file(
    path_or_fileobj=src,
    path_in_repo=dst,
    repo_id=repo,
    repo_type="dataset",
    commit_message="Add SANA-Video sglang acceleration ablation videos (labeled zip)",
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
