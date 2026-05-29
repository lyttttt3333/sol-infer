#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -J ltx23-stage1-sparse-same-noise
#SBATCH -o outputs/slurm/ltx23-stage1-sparse-same-noise-%j.out
#SBATCH -e outputs/slurm/ltx23-stage1-sparse-same-noise-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm outputs/.cache/huggingface outputs/.cache/torch outputs/.cache/triton outputs/.cache/sgl_diffusion outputs/.tmp
export HF_HOME="$PWD/outputs/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/outputs/.cache"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export TRITON_CACHE_DIR="$PWD/outputs/.cache/triton"
export SGLANG_DIFFUSION_CACHE_ROOT="$PWD/outputs/.cache/sgl_diffusion"
export TMPDIR="$PWD/outputs/.tmp"


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

ROOT="${ROOT:-outputs/ltx23-branch-baselines-same-noise-1080p10s}"
LTX_MODEL_PATH="${LTX_MODEL_PATH:-outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493}"
SHARED_DIR="$ROOT/shared_noise"
if [[ ! -d "$LTX_MODEL_PATH" ]]; then
  echo "Missing repo-local SGLang materialized model: $LTX_MODEL_PATH" >&2
  exit 4
fi
OUT_DIR="$ROOT/stage1_sparse_schedule"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"
mkdir -p outputs/slurm "$OUT_DIR"

for f in \
  "$SHARED_DIR/diffusers_stage1_video_initial.pt" \
  "$SHARED_DIR/diffusers_stage1_audio_initial.pt" \
  "$SHARED_DIR/diffusers_stage2_video_noise.pt" \
  "$SHARED_DIR/diffusers_stage2_audio_noise.pt"; do
  if [[ ! -s "$f" ]]; then
    echo "Missing shared-noise artifact: $f" >&2
    exit 3
  fi
done

export SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH="$SHARED_DIR/diffusers_stage1_video_initial.pt"
export SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH="$SHARED_DIR/diffusers_stage1_audio_initial.pt"
export SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH="$SHARED_DIR/diffusers_stage2_video_noise.pt"
export SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH="$SHARED_DIR/diffusers_stage2_audio_noise.pt"
export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$OUT_DIR/latents"
export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$OUT_DIR/latents"

# ltx-stage1-sparse-schedule branch setting.
export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"
export SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE="${SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE:-true}"
export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS="${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS:-5}"
export SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY:-0.8}"
export SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY:-0.9}"

.conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path "$LTX_MODEL_PATH" \
  --backend auto \
  --pipeline-class-name LTX2TwoStagePipeline \
  --num-gpus 1 \
  --performance-mode speed \
  --ltx2-two-stage-device-mode resident \
  --warmup true \
  --warmup-steps 30 \
  --height 1088 \
  --width 1920 \
  --num-frames 241 \
  --fps 24 \
  --seed 42 \
  --num-inference-steps 30 \
  --guidance-scale 3.0 \
  --guidance-rescale 0.7 \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --prompt "$PROMPT" \
  --return-file-paths-only true \
  --component-attention-backends.transformer piecewise_attn \
  --component-attention-backends.transformer_2 piecewise_attn \
  --output-file-path "$OUT_DIR/out.mp4" \
  --perf-dump-path "$OUT_DIR/perf.json"

VARIANT="stage1_sparse_schedule" OUT_DIR="$OUT_DIR" .conda/ltx23/bin/python scripts/summarize_ltx23_sglang_perf.py \
  --out-dir "$OUT_DIR" \
  --variant "stage1_sparse_schedule"

echo "[done] stage1 sparse schedule same-noise: $OUT_DIR/out.mp4"
