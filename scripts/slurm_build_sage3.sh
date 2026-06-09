#!/bin/bash
#SBATCH --job-name=build-sage3
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/build-sage3-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/build-sage3-%j.out
set -uo pipefail
REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python
SRC=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/SageAttention/sageattention3_blackwell
TARGET=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/pylibs; mkdir -p "$TARGET"
FULLLOG=/home/yitongl/cosmos3-run/sage3-build-full.log
echo "[$(date)] Node $(hostname)"
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/.buildtmp; mkdir -p "$TMPDIR"
export MAX_JOBS=32
export CCCL_INC="-I/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/cccl/thrust -I/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/cccl/cub -I/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/cccl/libcudacxx/include"
export NVCC_PREPEND_FLAGS="$CCCL_INC ${NVCC_PREPEND_FLAGS:-}"
export CPATH="/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/cccl/thrust:/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/cccl/cub:/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/cccl/libcudacxx/include:${CPATH:-}"
cd "$SRC"
echo "[$(date)] === building sageattn3, FULL log -> $FULLLOG ==="
$PYTHON -m pip install --no-build-isolation --no-deps --target "$TARGET" . > "$FULLLOG" 2>&1
RC=$?
echo "[$(date)] pip rc=$RC"
echo "=== FIRST nvcc/compile error (FAILED block) ==="
grep -nE "FAILED:|error:|fatal error|nvcc fatal|static assert|no instance|namespace|identifier|not a member|Unsupported" "$FULLLOG" | head -25
echo "=== import test ==="
PYTHONPATH="$TARGET:$REPO/python" $PYTHON -c "from sageattn3 import sageattn3_blackwell; print('IMPORT OK')" 2>&1 | tail -3
echo "[$(date)] BUILD_DONE rc=$RC"
