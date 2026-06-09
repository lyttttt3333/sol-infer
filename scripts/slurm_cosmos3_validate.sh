#!/bin/bash
#SBATCH --job-name=cosmos3-validate
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/home/yitongl/cosmos3-run/validate-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/validate-%j.out

set -euo pipefail

REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python

RUN_BASE=/home/yitongl/cosmos3-run
CACHE=$RUN_BASE/.cache
ROOT=$RUN_BASE/cosmos3-cache-matrix
mkdir -p "$ROOT/logs" "$CACHE"/{huggingface,xdg,torch,triton,torchinductor,torch_extensions,cuda,sgl_diffusion} "$RUN_BASE/.tmp"

cd "$REPO"

echo "[$(date)] Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1

# torch in this env is cu130 but no system CUDA toolkit exists. JIT kernel
# compilation (tvm_ffi / DeepGEMM) needs CUDA_HOME -> use the pip-bundled
# nvidia-cu13 toolkit (has nvcc, ptxas, headers, lib64). Without this the run
# dies at generation with "Could not find CUDA installation".
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

COSMOS3_NANO_HASH=$(cat "$HF_HUB_CACHE/models--nvidia--Cosmos3-Nano/refs/main")
COSMOS3_NANO_LOCAL="$HF_HUB_CACHE/models--nvidia--Cosmos3-Nano/snapshots/$COSMOS3_NANO_HASH"
echo "[$(date)] Local model path: $COSMOS3_NANO_LOCAL"

export XDG_CACHE_HOME=$CACHE/xdg
export TORCH_HOME=$CACHE/torch
export TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor
export TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda
export SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion
export TMPDIR=$RUN_BASE/.tmp

# Single baseline run, prompt 0 only — validates the local-path fix end to end.
# Writes into the shared ROOT so the full matrix later SKIPS this completed cell.
ROOT="$ROOT" \
MODEL_SIZES=16b \
VARIANTS="baseline" \
PROMPT_COUNT=1 \
MAKE_COMPARE=0 \
MAKE_REPORT=0 \
PYTHON_BIN="$PYTHON" \
COSMOS3_16B_MODEL_PATH="$COSMOS3_NANO_LOCAL" \
COSMOS3_16B_NUM_GPUS=1 \
WARMUP=false \
ALLOW_PARTIAL=1 \
bash scripts/run_cosmos3_cache_matrix.sh

echo "[$(date)] VALIDATION done. Checking output:"
ls -lh "$ROOT/16b/prompt_0/baseline/" 2>/dev/null || echo "  (no output dir)"
