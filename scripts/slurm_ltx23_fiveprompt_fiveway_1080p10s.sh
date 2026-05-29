#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH -J ltx23-5prompt-5way
#SBATCH -o outputs/slurm/ltx23-5prompt-5way-%j.out
#SBATCH -e outputs/slurm/ltx23-5prompt-5way-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Lossless BF16 kernel path. These should only change kernel scheduling/fusion,
# not the denoising algorithm or request-level parameters.
export SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=1
export SGLANG_LTX2_FUSED_ADALN=1
export SGLANG_LTX2_FUSED_QKNORM_ROPE=1
export SGLANG_LTX2_FUSED_DUAL_MODULATE=1
export SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1
export SGLANG_LTX2_FUSED_RESIDUAL_GATE=1
export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1
export SGLANG_LTX2_COMPILE_GATE_TO_OUT=1
export SGLANG_LTX2_FUSED_AUDIO_QKVG=1
export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=1
export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=1

# NVFP4 path.
export SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND="${SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND:-cudnn}"
export SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND="${SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND:-flashinfer}"
export SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU="${SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU:-1}"
export SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE="${SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE:-1}"
export SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE="${SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE:-1}"

# Match ltx-sparse-attn-bringup: score-routed top-k blocks with approximate remainder.
export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"

ROOT="${ROOT:-outputs/ltx23-5prompt-5way-1080p10s}"
MODEL_DIR="${MODEL_DIR:-/home/yitongl/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493}"
DIFFUSERS_PRETRAINED="${DIFFUSERS_PRETRAINED:-diffusers/LTX-2.3-Diffusers}"
FORCE="${FORCE:-0}"
mkdir -p outputs/slurm "$ROOT"

NEGATIVE_PROMPT="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."

PROMPT_SLUGS=(
  "redwood_mist"
  "neon_market_rain"
  "watchmaker_macro"
  "storm_lighthouse"
  "robotics_lab"
)
PROMPTS=(
  "A steady aerial tracking shot over a redwood forest after rain, morning mist drifting between tall trees, sunlight breaking through the canopy, cinematic natural motion"
  "A slow dolly shot through a neon-lit night market in light rain, steam rising from food stalls, reflections on wet pavement, people walking naturally"
  "A close-up macro video of a watchmaker assembling tiny brass gears under warm desk light, tweezers moving carefully, shallow depth of field"
  "A coastal lighthouse at dusk during a storm, waves crashing on dark rocks, the camera orbiting slowly as the light beam sweeps through sea mist"
  "A bright robotics lab where a small robotic arm places colorful electronic components onto a circuit board, smooth precise motion, realistic reflections"
)

.conda/ltx23/bin/python - "$ROOT/prompts.json" <<'PY'
import json
import sys
from pathlib import Path
slugs = [
    "redwood_mist",
    "neon_market_rain",
    "watchmaker_macro",
    "storm_lighthouse",
    "robotics_lab",
]
prompts = [
    "A steady aerial tracking shot over a redwood forest after rain, morning mist drifting between tall trees, sunlight breaking through the canopy, cinematic natural motion",
    "A slow dolly shot through a neon-lit night market in light rain, steam rising from food stalls, reflections on wet pavement, people walking naturally",
    "A close-up macro video of a watchmaker assembling tiny brass gears under warm desk light, tweezers moving carefully, shallow depth of field",
    "A coastal lighthouse at dusk during a storm, waves crashing on dark rocks, the camera orbiting slowly as the light beam sweeps through sea mist",
    "A bright robotics lab where a small robotic arm places colorful electronic components onto a circuit board, smooth precise motion, realistic reflections",
]
Path(sys.argv[1]).write_text(json.dumps([
    {"slug": slug, "prompt": prompt} for slug, prompt in zip(slugs, prompts)
], indent=2) + "\n")
PY

COMMON_SGLANG_ARGS=(
  --model-path Lightricks/LTX-2.3
  --backend auto
  --pipeline-class-name LTX2TwoStagePipeline
  --num-gpus 1
  --performance-mode speed
  --ltx2-two-stage-device-mode resident
  --warmup true
  --warmup-steps 30
  --height 1088
  --width 1920
  --num-frames 241
  --fps 24
  --seed 42
  --num-inference-steps 30
  --guidance-scale 3.0
  --guidance-rescale 0.7
  --negative-prompt "$NEGATIVE_PROMPT"
  --return-file-paths-only true
)

run_diffusers() {
  local prompt="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  if [[ "$FORCE" != "1" && -s "$out_dir/out.mp4" && -s "$out_dir/perf_diffusers.json" ]]; then
    echo "[skip] diffusers $out_dir"
    return
  fi
  echo "[run] diffusers -> $out_dir"
  PYTHONPATH="$PWD/outputs/python_deps/ltx23_diffusers:$PYTHONPATH" .conda/ltx23/bin/python scripts/benchmark_ltx23_diffusers_twostage.py \
    --pretrained-model-id "$DIFFUSERS_PRETRAINED" \
    --model-dir "$MODEL_DIR" \
    --local-files-only \
    --output-dir "$out_dir" \
    --output-video-path "$out_dir/out.mp4" \
    --prompt "$prompt" \
    --negative-prompt "$NEGATIVE_PROMPT" \
    --width 1920 \
    --height 1088 \
    --num-frames 241 \
    --fps 24 \
    --seed 42 \
    --guidance-scale 3.0 \
    --stage2-guidance-scale 1.0 \
    --stg-scale 1.0 \
    --modality-scale 3.0 \
    --guidance-rescale 0.7 \
    --audio-guidance-scale 7.0 \
    --audio-stg-scale 1.0 \
    --audio-modality-scale 3.0 \
    --audio-guidance-rescale 0.7 \
    --spatio-temporal-guidance-blocks 28 \
    --use-cross-timestep \
    --stage1-steps 30 \
    --stage2-steps 3 \
    --stage2-sigmas 0.909375 0.725 0.421875 \
    --stage1-lora-strength 0.0 \
    --stage2-lora-strength 1.0 \
    --dtype bf16 \
    --device cuda \
    --enable-vae-tiling \
    --warmup \
    --actual-runs 1
}

run_sglang() {
  local variant="$1"
  local prompt="$2"
  local out_dir="$3"
  shift 3
  mkdir -p "$out_dir"
  if [[ "$FORCE" != "1" && -s "$out_dir/out.mp4" && -s "$out_dir/perf.json" ]]; then
    echo "[skip] $variant $out_dir"
    return
  fi
  echo "[run] $variant -> $out_dir"
  .conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    "${COMMON_SGLANG_ARGS[@]}" \
    --prompt "$prompt" \
    "$@" \
    --output-file-path "$out_dir/out.mp4" \
    --perf-dump-path "$out_dir/perf.json"
  .conda/ltx23/bin/python scripts/summarize_ltx23_sglang_perf.py --out-dir "$out_dir" --variant "$variant"
}

for idx in "${!PROMPT_SLUGS[@]}"; do
  slug="${PROMPT_SLUGS[$idx]}"
  prompt="${PROMPTS[$idx]}"
  prompt_dir="$ROOT/$slug"
  echo "========== prompt $((idx + 1))/5: $slug =========="
  echo "$prompt"

  run_diffusers "$prompt" "$prompt_dir/diffusers"

  run_sglang "kernel_bf16" "$prompt" "$prompt_dir/kernel_bf16"

  run_sglang "nvfp4" "$prompt" "$prompt_dir/nvfp4" \
    --component-paths.transformer outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat \
    --component-paths.transformer_2 outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat

  run_sglang "sparse_bf16" "$prompt" "$prompt_dir/sparse_bf16" \
    --attention-backend piecewise_attn

  run_sglang "nvfp4_sparse" "$prompt" "$prompt_dir/nvfp4_sparse" \
    --attention-backend piecewise_attn \
    --component-paths.transformer outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat \
    --component-paths.transformer_2 outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat
done

.conda/ltx23/bin/python scripts/make_ltx23_fiveway_prompt_grid.py \
  --root "$ROOT" \
  --out "$ROOT/ltx23-5prompt-5way-grid-4k.mp4" \
  --cell-width 768 \
  --cell-height 432

echo "wrote $ROOT/ltx23-5prompt-5way-grid-4k.mp4"
echo "wrote $ROOT/fiveway_grid_report.json"
