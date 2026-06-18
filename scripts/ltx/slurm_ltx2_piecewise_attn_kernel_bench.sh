#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH -J ltx2-pw-attn-kernel
#SBATCH -o outputs/slurm/ltx2-pw-attn-kernel-%j.out
#SBATCH -e outputs/slurm/ltx2-pw-attn-kernel-%j.err

set -euo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm outputs/ltx2_piecewise_attn_kernel_bench
REPO_ROOT="$PWD"
source "$REPO_ROOT/scripts/ltx/env_ltx23_persistent_cache.sh"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [[ -d "$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13" ]]; then
  export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
  export CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
OUT="${OUT:-outputs/ltx2_piecewise_attn_kernel_bench/result_${SLURM_JOB_ID}.json}"
.conda/ltx23/bin/python scripts/ltx/bench_ltx2_piecewise_attn_kernel.py --out "$OUT" --warmup "${WARMUP:-5}" --iters "${ITERS:-10}"
echo "[done] $OUT"
