#!/bin/bash
#SBATCH --job-name=cosmos3-cache-bench
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/slurm-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/slurm-%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# NOTE: /lustre/fs1 (the repo filesystem) is 100% full (0 bytes free), so ALL
# data writes — videos, perf json, logs, compile caches, tmp — are redirected
# to /home (106 TB free). Empty-dir mkdir still works on fs1 (inodes free),
# which is why the benchmark script's unconditional outputs/.cache mkdirs are
# harmless no-ops; only the env vars below decide where real bytes land.
# ---------------------------------------------------------------------------

REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python

RUN_BASE=/home/yitongl/cosmos3-run
CACHE=$RUN_BASE/.cache
ROOT=$RUN_BASE/cosmos3-cache-matrix
mkdir -p "$ROOT/logs" "$CACHE"/{huggingface,xdg,torch,triton,torchinductor,torch_extensions,cuda,sgl_diffusion} "$RUN_BASE/.tmp"

cd "$REPO"

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] GPUs:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# All caches/outputs on /home; HF cache already populated by the download job.
export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1                       # model is fully cached; don't hit the slow login-node network

# torch here is cu130 but no system CUDA toolkit exists. JIT kernel compilation
# (tvm_ffi / DeepGEMM) needs CUDA_HOME -> use the pip-bundled nvidia-cu13 toolkit
# (nvcc, ptxas, headers, lib64). Without it the run dies at generation with
# "Could not find CUDA installation. Please set CUDA_HOME environment variable."
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

# Resolve the LOCAL HF-cache snapshot dir and pass THAT as --model-path (not the
# bare "nvidia/Cosmos3-Nano" repo id). Passing the repo id routes through
# maybe_download_model_index()'s remote hf_hub_download(local_dir=tmp) branch,
# which under HF_HUB_OFFLINE returns a config WITHOUT _class_name -> the engine
# silently falls back to the generic diffusers pipeline and dies with
# "Could not locate the pipeline.py". A local dir instead goes through
# verify_model_config_and_directory() (reads model_index.json -> _class_name=
# "Cosmos3OmniDiffusersPipeline" -> native Cosmos3Pipeline) and _get_config_info
# case 2b matches the "models--nvidia--cosmos3-nano" cache fragment -> Cosmos3Config.
COSMOS3_NANO_HASH=$(cat "$HF_HUB_CACHE/models--nvidia--Cosmos3-Nano/refs/main")
COSMOS3_NANO_LOCAL="$HF_HUB_CACHE/models--nvidia--Cosmos3-Nano/snapshots/$COSMOS3_NANO_HASH"
echo "[$(date)] Cosmos3-Nano local model path: $COSMOS3_NANO_LOCAL"
test -f "$COSMOS3_NANO_LOCAL/model_index.json" || { echo "ERROR: model_index.json missing"; exit 1; }

export XDG_CACHE_HOME=$CACHE/xdg
export TORCH_HOME=$CACHE/torch
export TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor
export TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda
export SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion
export TMPDIR=$RUN_BASE/.tmp

# FORCE=1: the very first run's prompt0/baseline was a COLD process (one-time JIT
# kernel compile counted into its denoise stage -> 33.5s vs the warm steady-state
# ~19.3s), which inflated prompt0's apparent speedups to ~1.75x. The JIT cache now
# lives on /home and is warm, so re-running every cell (incl. baseline) gives a
# clean, apples-to-apples warm-vs-warm comparison.
# dbcache_mild dropped: cache-dit/DBCache assumes a `transformer_blocks` attr that
# Cosmos3OmniTransformer (dual-pathway language_model.layers + gen_layers) lacks.
# Added a TeaCache threshold sweep (0.04 / 0.08 / 0.12) to show speedup vs PSNR.
ROOT="$ROOT" \
MODEL_SIZES=16b \
VARIANTS="baseline teacache_c04_s5 teacache_c08_s5 teacache_c12_s5 pab_cross2" \
PROMPT_COUNT=2 \
PYTHON_BIN="$PYTHON" \
COSMOS3_16B_MODEL_PATH="$COSMOS3_NANO_LOCAL" \
COSMOS3_16B_NUM_GPUS=1 \
WARMUP=false \
FORCE=1 \
ALLOW_PARTIAL=1 \
bash scripts/cosmos/run_cosmos3_cache_matrix.sh

echo "[$(date)] Done! Output at: $ROOT"
echo "[$(date)] Report:  $ROOT/benchmark_report.html"
