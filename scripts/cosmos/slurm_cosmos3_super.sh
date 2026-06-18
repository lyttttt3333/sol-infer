#!/bin/bash
#SBATCH --job-name=cosmos3-super
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/%x-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/%x-%j.out

set -euo pipefail
# Generic Cosmos3-Super (64b) runner: one model repo × one prompt × cache sweep,
# OFFICIAL settings, 4-GPU sequence parallel. Used for direct T2V and (with
# IMAGE_PATH set + NUM_FRAMES=189) the I2V stage of the cascade.
#
# Required env: MODEL_REPO (e.g. nvidia/Cosmos3-Super), ROOT, PROMPT_FILE, PROMPT_TAG
# Optional env: VARIANTS, WARMUP, NUM_FRAMES, IMAGE_PATH, NUM_GPUS

: "${MODEL_REPO:?}" ; : "${ROOT:?}" ; : "${PROMPT_FILE:?}" ; : "${PROMPT_TAG:?}"
REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python
RUN_BASE=/home/yitongl/cosmos3-run
CACHE=$RUN_BASE/.cache
mkdir -p "$ROOT/logs" "$CACHE"/{xdg,torch,triton,torchinductor,torch_extensions,cuda,sgl_diffusion} "$RUN_BASE/.tmp"
cd "$REPO"
echo "[$(date)] Node $(hostname)  MODEL=$MODEL_REPO  TAG=$PROMPT_TAG  frames=${NUM_FRAMES:-189}  image=${IMAGE_PATH:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HOME=/home/yitongl/.hf_cache/huggingface HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=1
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export XDG_CACHE_HOME=$CACHE/xdg TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion TMPDIR=$RUN_BASE/.tmp

REPO_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_REPO/\//--}"
HASH=$(cat "$REPO_CACHE_DIR/refs/main")
LOCAL="$REPO_CACHE_DIR/snapshots/$HASH"
test -f "$LOCAL/model_index.json" || { echo "ERROR model_index.json missing for $MODEL_REPO"; exit 1; }

OFFICIAL_NEG="$(cat "$RUN_BASE/official_prompts/negative_prompt.txt")"
PROMPT_STR="$(cat "$PROMPT_FILE")"

ROOT="$ROOT" \
MODEL_SIZES=64b \
VARIANTS="${VARIANTS:-baseline teacache_c115_s16_m2 teacache_c115_s10_m3 teacache_c130_s8_m4 teacache_c150_s5_m8}" \
PROMPT_COUNT=1 \
PROMPT_0="$PROMPT_STR" \
NEGATIVE_PROMPT="$OFFICIAL_NEG" \
HEIGHT=720 WIDTH=1280 NUM_FRAMES="${NUM_FRAMES:-189}" FPS=24 \
NUM_INFERENCE_STEPS=35 GUIDANCE_SCALE=6.0 FLOW_SHIFT=10.0 MAX_SEQUENCE_LENGTH=4096 \
PYTHON_BIN="$PYTHON" \
COSMOS3_64B_MODEL_PATH="$LOCAL" \
DIT_CPU_OFFLOAD=false \
COSMOS3_64B_NUM_GPUS="${NUM_GPUS:-4}" \
WARMUP="${WARMUP:-true}" WARMUP_STEPS=1 \
FORCE="${FORCE:-1}" \
MAKE_COMPARE="${MAKE_COMPARE:-1}" MAKE_REPORT="${MAKE_REPORT:-1}" \
ALLOW_PARTIAL=1 \
bash scripts/cosmos/run_cosmos3_cache_matrix.sh

echo "[$(date)] DONE $MODEL_REPO $PROMPT_TAG -> $ROOT"
