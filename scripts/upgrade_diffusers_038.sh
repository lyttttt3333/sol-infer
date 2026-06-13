#!/bin/bash
#SBATCH --job-name=diffusers-upgrade
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/yitongl/sana_video/logs/diffusers-upgrade-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/diffusers-upgrade-%j.out

# ---------------------------------------------------------------------------
# Surgical diffusers 0.37.0 -> 0.38.0 upgrade in the shared .conda/ltx23 env so
# SanaVideoPipeline can drive the LTX-2 VAE used by the 720p checkpoint.
# --no-deps: diffusers is pure-Python and its deps are already satisfied by the
# recent ltx23 stack; this protects the Blackwell torch 2.11 / transformers
# 5.8.1 / nvidia-cu130 wheels from any collateral pip change.
# Rollback:  .conda/ltx23/bin/pip install -U --no-deps diffusers==0.37.0
# ---------------------------------------------------------------------------

set -uo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PIP=.conda/ltx23/bin/pip
PY=.conda/ltx23/bin/python
export PIP_CACHE_DIR=/home/yitongl/.cache/pip

echo "[$(date)] node=$(hostname)"
echo "[before] diffusers=$($PY -c 'import diffusers;print(diffusers.__version__)' 2>/dev/null)"

echo "=== pip install -U --no-deps diffusers==0.38.0 ==="
$PIP install -U --no-deps diffusers==0.38.0 2>&1 | tail -30

echo "=== verify ==="
if $PY - << 'PYEOF'
import diffusers, inspect
print("diffusers:", diffusers.__version__)
from diffusers import SanaVideoPipeline
import diffusers.pipelines.sana_video.pipeline_sana_video as m
src = inspect.getsource(m)
ltx2 = "AutoencoderKLLTX2Video" in src
tcr = "temporal_compression_ratio" in src
print("SanaVideoPipeline references AutoencoderKLLTX2Video:", ltx2)
print("SanaVideoPipeline references temporal_compression_ratio:", tcr)
# torch/transformers must be UNCHANGED (Blackwell stack)
import torch, transformers
print("torch:", torch.__version__, "| transformers:", transformers.__version__)
# cache_dit is the only Required-by; make sure we didn't break it
try:
    import cache_dit
    print("cache_dit import: OK", getattr(cache_dit, "__version__", "?"))
except Exception as e:
    print("cache_dit import: FAILED", repr(e)[:160])
assert diffusers.__version__.startswith("0.38"), "version not upgraded"
assert ltx2 and tcr, "LTX-2 VAE support NOT present in upgraded SanaVideoPipeline"
print("VERIFY_OK")
PYEOF
then
    echo "[$(date)] UPGRADE_COMPLETE_MARKER"
else
    rc=$?
    echo "[$(date)] UPGRADE_FAILED_MARKER rc=$rc"
    exit $rc
fi
