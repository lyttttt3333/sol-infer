#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH -J ltx23-context-align
#SBATCH -o outputs/slurm/ltx23-context-align-%j.out
#SBATCH -e outputs/slurm/ltx23-context-align-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-context-alignment-1080p10s}"
OFFICIAL_DIR="$ROOT/official"
SGLANG_DIR="$ROOT/sglang_dense"
mkdir -p "$OFFICIAL_DIR/context" "$SGLANG_DIR/context"

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

for required in "$CHECKPOINT_PATH" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER" "$GEMMA_ROOT/tokenizer/tokenizer.model" "$GEMMA_ROOT/tokenizer/preprocessor_config.json"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing required asset: $required" >&2
    exit 1
  fi
done

echo "[phase1] official context-only dump"
(
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="$PWD/$OFFICIAL_SRC/packages/ltx-core/src:$PWD/$OFFICIAL_SRC/packages/ltx-pipelines/src:$OFFICIAL_DEPS:$DIFFUSERS_DEPS:$PWD/python:${PYTHONPATH:-}"
  .conda/ltx23/bin/python scripts/benchmark_ltx23_official_hq_runtime.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --distilled-lora "$DISTILLED_LORA" \
    --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \
    --gemma-root "$GEMMA_ROOT" \
    --output-video-path "$OFFICIAL_DIR/out.mp4" \
    --summary-json "$OFFICIAL_DIR/summary.json" \
    --dump-context-dir "$OFFICIAL_DIR/context" \
    --stop-after-context \
    --prompt "$PROMPT" \
    --negative-prompt "$NEGATIVE_PROMPT" \
    --seed 42 --height 1088 --width 1920 --num-frames 241 --frame-rate 24 \
    --num-inference-steps 15
) > "$OFFICIAL_DIR/run.log" 2>&1

echo "[phase2] SGLang dense context dump"
(
  export CUDA_VISIBLE_DEVICES=1
  export SGLANG_HQ_VARIANT=dense
  export ROOT="$ROOT"
  export OUT_DIR="$SGLANG_DIR"
  export FORCE=1
  export WARMUP=true
  export WARMUP_STEPS=15
  export MASTER_PORT=30205
  export SGLANG_LTX2_DUMP_CONTEXT_DIR="$SGLANG_DIR/context"
  bash scripts/run_ltx23_sglang_hq_1080p10s.sh
) > "$SGLANG_DIR/run.log" 2>&1

for required in "$OFFICIAL_DIR/context/official_contexts.pt" "$SGLANG_DIR/context/sglang_contexts.pt"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing context dump: $required" >&2
    exit 2
  fi
done

.conda/ltx23/bin/python - "$ROOT" <<'PYCMP'
import json
from pathlib import Path
import torch
root = Path(__import__('sys').argv[1])
off = torch.load(root/'official/context/official_contexts.pt', map_location='cpu')
sg = torch.load(root/'sglang_dense/context/sglang_contexts.pt', map_location='cpu')
keys = ['video_context_pos','audio_context_pos','video_context_neg','audio_context_neg']
rows = []
for key in keys:
    a = off[key]
    b = sg[key]
    if a is None or b is None:
        rows.append({'key': key, 'missing': [a is None, b is None]})
        continue
    if a.shape != b.shape:
        rows.append({'key': key, 'shape_mismatch': [list(a.shape), list(b.shape)]})
        continue
    d = a.float() - b.float()
    rows.append({
        'key': key,
        'shape': list(a.shape),
        'dtype_official': str(a.dtype),
        'dtype_sglang': str(b.dtype),
        'equal': bool(torch.equal(a, b)),
        'max_abs': float(d.abs().max()),
        'mean_abs': float(d.abs().mean()),
        'mse': float((d*d).mean()),
    })
(root/'context_diff.json').write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
print(json.dumps(rows, indent=2, sort_keys=True))
PYCMP

echo "[done] context diff: $ROOT/context_diff.json"
