#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 01:00:00
#SBATCH -J ltx23-offctx
#SBATCH -o outputs/slurm/ltx23-offctx-%j.out
#SBATCH -e outputs/slurm/ltx23-offctx-%j.err

set -euo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-context-alignment-raw-1080p10s}"
OUT_DIR="$ROOT/official"
mkdir -p "$OUT_DIR/context" outputs/.tmp outputs/.cache/huggingface outputs/.cache/xdg outputs/.cache/torch outputs/.cache/triton outputs/.cache/torchinductor outputs/.cache/torch_extensions outputs/.cache/cuda

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$PWD/outputs/.cache/huggingface"
export HF_HUB_CACHE="$PWD/outputs/.cache/huggingface/hub"
export XDG_CACHE_HOME="$PWD/outputs/.cache/xdg"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export TRITON_CACHE_DIR="$PWD/outputs/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$PWD/outputs/.cache/torch_extensions"
export CUDA_CACHE_PATH="$PWD/outputs/.cache/cuda"
export TMPDIR="$PWD/outputs/.tmp"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

OFFICIAL_SRC="${OFFICIAL_SRC:-outputs/LTX-2-official-main}"
OFFICIAL_DEPS="$PWD/outputs/python_deps/ltx23_official"
DIFFUSERS_DEPS="$PWD/outputs/python_deps/ltx23_diffusers"
export PYTHONPATH="$PWD/$OFFICIAL_SRC/packages/ltx-core/src:$PWD/$OFFICIAL_SRC/packages/ltx-pipelines/src:$OFFICIAL_DEPS:$DIFFUSERS_DEPS:$PWD/python:${PYTHONPATH:-}"

MODEL_COMPONENT_DIR="${MODEL_COMPONENT_DIR:-outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493}"
OFFICIAL_MODEL_DIR="${OFFICIAL_MODEL_DIR:-outputs/LTX-2.3-official-files}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OFFICIAL_MODEL_DIR/ltx-2.3-22b-dev.safetensors}"
DISTILLED_LORA="${DISTILLED_LORA:-$OFFICIAL_MODEL_DIR/ltx-2.3-22b-distilled-lora-384-1.1.safetensors}"
SPATIAL_UPSAMPLER="${SPATIAL_UPSAMPLER:-$MODEL_COMPONENT_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-$MODEL_COMPONENT_DIR}"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"

INIT_FILE="$OFFICIAL_SRC/packages/ltx-pipelines/src/ltx_pipelines/__init__.py"
if grep -q "from ltx_pipelines.a2vid_two_stage" "$INIT_FILE"; then
  cp "$INIT_FILE" "$INIT_FILE.official_bak"
  printf '%s\n' '"""LTX-2 Pipelines package, local lightweight init for direct module execution."""' '' '__all__ = []' > "$INIT_FILE"
fi

.conda/ltx23/bin/python scripts/benchmark_ltx23_official_hq_runtime.py \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --distilled-lora "$DISTILLED_LORA" \
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \
  --gemma-root "$GEMMA_ROOT" \
  --output-video-path "$OUT_DIR/out.mp4" \
  --summary-json "$OUT_DIR/summary.json" \
  --dump-context-dir "$OUT_DIR/context" \
  --stop-after-context \
  --prompt "$PROMPT" \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --seed 42 --height 1088 --width 1920 --num-frames 241 --frame-rate 24 \
  --num-inference-steps 15
