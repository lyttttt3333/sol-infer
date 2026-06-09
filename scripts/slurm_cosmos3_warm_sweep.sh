#!/bin/bash
#SBATCH --job-name=cosmos3-warm-sweep
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/home/yitongl/cosmos3-run/warm-sweep-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/warm-sweep-%j.out

set -euo pipefail

REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python

RUN_BASE=/home/yitongl/cosmos3-run
CACHE=$RUN_BASE/.cache
ROOT=$RUN_BASE/cosmos3-warm-sweep
mkdir -p "$ROOT/logs" "$CACHE"/{huggingface,xdg,torch,triton,torchinductor,torch_extensions,cuda,sgl_diffusion} "$RUN_BASE/.tmp"

cd "$REPO"
echo "[$(date)] Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export XDG_CACHE_HOME=$CACHE/xdg TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion TMPDIR=$RUN_BASE/.tmp

COSMOS3_NANO_HASH=$(cat "$HF_HUB_CACHE/models--nvidia--Cosmos3-Nano/refs/main")
COSMOS3_NANO_LOCAL="$HF_HUB_CACHE/models--nvidia--Cosmos3-Nano/snapshots/$COSMOS3_NANO_HASH"

# KEY FIX vs the earlier run: WARMUP=true. Each cell now runs one UNTIMED warmup
# generation first, which pays the per-process cold start (cudnn autotune, the
# 32 GB dit_cpu_offload CPU->GPU move, allocator init) ~11 s — so the TIMED run's
# step 1 is warm (~0.25 s like the rest) instead of ~11 s. Earlier WARMUP=false
# left that ~11 s fixed cost inside the denoise-stage timer of BOTH baseline and
# every variant, diluting a true ~1.5x (at 12 skips) down to the ~1.1x I reported.
# Variants mirror the reference table: threshold/start/max = c<NNN>_s<S>_m<M>.
ROOT="$ROOT" \
MODEL_SIZES=16b \
VARIANTS="baseline teacache_c115_s16_m2 teacache_c115_s10_m3 teacache_c130_s8_m4 teacache_c150_s5_m8" \
PROMPT_COUNT=2 \
PYTHON_BIN="$PYTHON" \
COSMOS3_16B_MODEL_PATH="$COSMOS3_NANO_LOCAL" \
COSMOS3_16B_NUM_GPUS=1 \
WARMUP=true \
WARMUP_STEPS=1 \
FORCE=1 \
ALLOW_PARTIAL=1 \
bash scripts/run_cosmos3_cache_matrix.sh

echo "[$(date)] Done! Output at: $ROOT"
