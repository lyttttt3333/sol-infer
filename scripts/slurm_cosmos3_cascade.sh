#!/bin/bash
#SBATCH --job-name=cosmos3-cascade
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
# Cosmos3-Super T2I->I2V cascade for ONE prompt:
#   Stage 1: Super-Text2Image (num_frames=1)  -> a 1280x720 still image
#   Stage 2: Super-Image2Video (--image-path=stage1) + cache sweep -> videos
# OFFICIAL settings, 4-GPU sequence parallel.
# Required env: ROOT, PROMPT_FILE, PROMPT_TAG ; Optional: VARIANTS, WARMUP, FORCE

: "${ROOT:?}" ; : "${PROMPT_FILE:?}" ; : "${PROMPT_TAG:?}"
T2I_REPO=nvidia/Cosmos3-Super-Text2Image
I2V_REPO=nvidia/Cosmos3-Super-Image2Video

REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python
RUN_BASE=/home/yitongl/cosmos3-run
CACHE=$RUN_BASE/.cache
mkdir -p "$ROOT/logs" "$ROOT/t2i" "$CACHE"/{xdg,torch,triton,torchinductor,torch_extensions,cuda,sgl_diffusion} "$RUN_BASE/.tmp"
cd "$REPO"
echo "[$(date)] Node $(hostname)  CASCADE tag=$PROMPT_TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HOME=/home/yitongl/.hf_cache/huggingface HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/python:${PYTHONPATH:-}"
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export XDG_CACHE_HOME=$CACHE/xdg TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion TMPDIR=$RUN_BASE/.tmp
export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1

snap(){ local r="$1"; local d="$HF_HUB_CACHE/models--${r/\//--}"; echo "$d/snapshots/$(cat "$d/refs/main")"; }
T2I_LOCAL="$(snap "$T2I_REPO")"; I2V_LOCAL="$(snap "$I2V_REPO")"
OFFICIAL_NEG="$(cat "$RUN_BASE/official_prompts/negative_prompt.txt")"
PROMPT_STR="$(cat "$PROMPT_FILE")"
IMG="$ROOT/t2i/${PROMPT_TAG}.png"

# ---------- Stage 1: Text -> Image (num_frames=1) ----------
if [[ "${FORCE:-1}" == "1" || ! -s "$IMG" ]]; then
  echo "[$(date)] Stage1 T2I -> $IMG"
  "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path "$T2I_LOCAL" --num-gpus 4 \
    --prompt "$PROMPT_STR" --negative-prompt "$OFFICIAL_NEG" \
    --height 720 --width 1280 --num-frames 1 \
    --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
    --use-guardrails false --seed 42 --warmup false \
    --dit-cpu-offload false --enable-sequence-shard true \
    --scheduler-port 5590 --master-port 30990 \
    --output-file-path "$IMG" > "$ROOT/logs/t2i_${PROMPT_TAG}.log" 2>&1
fi
ls -lh "$IMG" 2>/dev/null || { echo "ERROR: T2I image not produced (see $ROOT/logs/t2i_${PROMPT_TAG}.log)"; tail -20 "$ROOT/logs/t2i_${PROMPT_TAG}.log"; exit 1; }

# ---------- Stage 2: Image -> Video cache sweep ----------
echo "[$(date)] Stage2 I2V sweep (image=$IMG)"
ROOT="$ROOT" \
MODEL_SIZES=64b \
VARIANTS="${VARIANTS:-baseline teacache_c115_s16_m2 teacache_c115_s10_m3 teacache_c130_s8_m4 teacache_c150_s5_m8}" \
PROMPT_COUNT=1 \
PROMPT_0="$PROMPT_STR" \
NEGATIVE_PROMPT="$OFFICIAL_NEG" \
HEIGHT=720 WIDTH=1280 NUM_FRAMES=189 FPS=24 \
NUM_INFERENCE_STEPS=35 GUIDANCE_SCALE=6.0 FLOW_SHIFT=10.0 MAX_SEQUENCE_LENGTH=4096 \
IMAGE_PATH="$IMG" \
PYTHON_BIN="$PYTHON" \
COSMOS3_64B_MODEL_PATH="$I2V_LOCAL" \
DIT_CPU_OFFLOAD=false \
COSMOS3_64B_NUM_GPUS=4 \
WARMUP="${WARMUP:-true}" WARMUP_STEPS=1 \
FORCE="${FORCE:-1}" \
MAKE_COMPARE=1 MAKE_REPORT=1 \
ALLOW_PARTIAL=1 \
bash scripts/run_cosmos3_cache_matrix.sh

echo "[$(date)] DONE cascade $PROMPT_TAG -> $ROOT"
