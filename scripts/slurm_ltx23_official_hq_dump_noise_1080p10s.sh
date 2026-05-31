#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -J ltx23-official-dump
#SBATCH -o outputs/slurm/ltx23-official-dump-%j.out
#SBATCH -e outputs/slurm/ltx23-official-dump-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$PWD/outputs/.cache/huggingface"
export HF_HUB_CACHE="$PWD/outputs/.cache/huggingface/hub"
export XDG_CACHE_HOME="$PWD/outputs/.cache/xdg"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export TRITON_CACHE_DIR="$PWD/outputs/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$PWD/outputs/.cache/torch_extensions"
export CUDA_CACHE_PATH="$PWD/outputs/.cache/cuda"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
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
ROOT="${ROOT:-outputs/ltx23-official-hq-dump-noise-1080p10s}"
OUT_DIR="$ROOT/official_hq_dump_noise"
DUMP_NOISE_DIR="${DUMP_NOISE_DIR:-$OUT_DIR/noise}"
DUMP_CONTEXT_DIR="${DUMP_CONTEXT_DIR:-}"
DUMP_STAGE1_DEBUG_DIR="${DUMP_STAGE1_DEBUG_DIR:-}"
DUMP_STAGE1_DEBUG_CALLS="${DUMP_STAGE1_DEBUG_CALLS:-2}"
DUMP_DIT_ACTIVATIONS_DIR="${DUMP_DIT_ACTIVATIONS_DIR:-}"
DUMP_DIT_ACTIVATIONS_CALLS="${DUMP_DIT_ACTIVATIONS_CALLS:-1}"
OUT_VIDEO="$OUT_DIR/out.mp4"
SUMMARY_JSON="$OUT_DIR/summary.json"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"

mkdir -p outputs/slurm outputs/.cache/huggingface outputs/.cache/xdg outputs/.cache/torch outputs/.cache/triton outputs/.cache/torchinductor outputs/.cache/torch_extensions outputs/.cache/cuda outputs/.tmp "$OUT_DIR" "$DUMP_NOISE_DIR"
if [[ -n "$DUMP_CONTEXT_DIR" ]]; then
  mkdir -p "$DUMP_CONTEXT_DIR"
fi
if [[ -n "$DUMP_STAGE1_DEBUG_DIR" ]]; then
  mkdir -p "$DUMP_STAGE1_DEBUG_DIR"
fi
if [[ -n "$DUMP_DIT_ACTIVATIONS_DIR" ]]; then
  mkdir -p "$DUMP_DIT_ACTIVATIONS_DIR"
fi

INIT_FILE="$OFFICIAL_SRC/packages/ltx-pipelines/src/ltx_pipelines/__init__.py"
if grep -q "from ltx_pipelines.a2vid_two_stage" "$INIT_FILE"; then
  cp "$INIT_FILE" "$INIT_FILE.official_bak"
  printf '%s\n' '"""LTX-2 Pipelines package, local lightweight init for direct module execution."""' '' '__all__ = []' > "$INIT_FILE"
fi

for required in "$CHECKPOINT_PATH" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER" "$GEMMA_ROOT/tokenizer/tokenizer.model" "$GEMMA_ROOT/tokenizer/preprocessor_config.json"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing required official runtime asset: $required" >&2
    exit 1
  fi
done

cat > "$OUT_DIR/run_command.txt" <<EOF
.conda/ltx23/bin/python scripts/benchmark_ltx23_official_hq_runtime.py \\
  --checkpoint-path "$CHECKPOINT_PATH" \\
  --distilled-lora "$DISTILLED_LORA" \\
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \\
  --gemma-root "$GEMMA_ROOT" \\
  --output-video-path "$OUT_VIDEO" \\
  --summary-json "$SUMMARY_JSON" \\
  --dump-noise-dir "$DUMP_NOISE_DIR" \
  --prompt "$PROMPT" \\
  --negative-prompt "$NEGATIVE_PROMPT" \\
  --seed 42 --height 1088 --width 1920 --num-frames 241 --frame-rate 24 \\
  --num-inference-steps 15
EOF

echo "[run] official HQ dump-noise benchmark -> $OUT_VIDEO"
echo "[run] dump_noise_dir=$DUMP_NOISE_DIR"
if [[ -n "$DUMP_CONTEXT_DIR" ]]; then
  echo "[run] dump_context_dir=$DUMP_CONTEXT_DIR"
fi
if [[ -n "$DUMP_STAGE1_DEBUG_DIR" ]]; then
  echo "[run] dump_stage1_debug_dir=$DUMP_STAGE1_DEBUG_DIR"
fi
if [[ -n "$DUMP_DIT_ACTIVATIONS_DIR" ]]; then
  echo "[run] dump_dit_activations_dir=$DUMP_DIT_ACTIVATIONS_DIR"
fi
EXTRA_ARGS=()
if [[ -n "$DUMP_CONTEXT_DIR" ]]; then
  EXTRA_ARGS+=(--dump-context-dir "$DUMP_CONTEXT_DIR")
fi
if [[ -n "$DUMP_STAGE1_DEBUG_DIR" ]]; then
  EXTRA_ARGS+=(--dump-stage1-debug-dir "$DUMP_STAGE1_DEBUG_DIR")
  EXTRA_ARGS+=(--dump-stage1-debug-calls "$DUMP_STAGE1_DEBUG_CALLS")
fi
if [[ -n "$DUMP_DIT_ACTIVATIONS_DIR" ]]; then
  EXTRA_ARGS+=(--dump-dit-activations-dir "$DUMP_DIT_ACTIVATIONS_DIR")
  EXTRA_ARGS+=(--dump-dit-activations-calls "$DUMP_DIT_ACTIVATIONS_CALLS")
fi
.conda/ltx23/bin/python scripts/benchmark_ltx23_official_hq_runtime.py \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --distilled-lora "$DISTILLED_LORA" \
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \
  --gemma-root "$GEMMA_ROOT" \
  --output-video-path "$OUT_VIDEO" \
  --summary-json "$SUMMARY_JSON" \
  --dump-noise-dir "$DUMP_NOISE_DIR" \
  "${EXTRA_ARGS[@]}" \
  --prompt "$PROMPT" \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --seed 42 \
  --height 1088 \
  --width 1920 \
  --num-frames 241 \
  --frame-rate 24 \
  --num-inference-steps 15

echo "[done] official HQ dump-noise video: $OUT_VIDEO"
echo "[done] official HQ dump-noise summary: $SUMMARY_JSON"
echo "[done] official HQ noise dumps: $DUMP_NOISE_DIR"
