#!/bin/bash
#SBATCH --job-name=sana-zip-up
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yitongl/sana_video/logs/zip-up-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/zip-up-%j.out

# Zip a folder of samples AND upload the zip to the HF dataset -- both steps on
# the dedicated cpu_datamover node (no heavy work on the login node).
set -uo pipefail

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
PYTHON=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/.conda/ltx23/bin/python

REPO=yitongl/ltx23-shares
SRC=${1:-/home/yitongl/sana_video/hf_validation}
ZIP=${2:-/home/yitongl/sana_video/sana_validation_16prompt_samples.zip}
DST=${3:-sana-video-validation/sana_validation_16prompt_samples.zip}

mkdir -p /home/yitongl/sana_video/logs
echo "[$(date)] Node: $(hostname)  zip $SRC -> $ZIP -> dataset:$REPO path:$DST"

if $PYTHON - "$SRC" "$ZIP" "$REPO" "$DST" << 'PYEOF'
import os, sys, zipfile
from huggingface_hub import HfApi

src, zippath, repo, dst = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
assert os.path.isdir(src), f"missing dir: {src}"
# STORED (no deflate): mp4s are already compressed, so deflate just burns CPU.
with zipfile.ZipFile(zippath, "w", zipfile.ZIP_STORED) as z:
    for fn in sorted(os.listdir(src)):
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            z.write(p, arcname=fn)
n = len(zipfile.ZipFile(zippath).namelist())
print(f"ZIP_OK {n} entries, {os.path.getsize(zippath)/1e6:.1f} MB -> {zippath}", flush=True)

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", exist_ok=True)
url = api.upload_file(
    path_or_fileobj=zippath,
    path_in_repo=dst,
    repo_id=repo,
    repo_type="dataset",
    commit_message="Add zipped SANA-Video 16-prompt validation samples",
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
