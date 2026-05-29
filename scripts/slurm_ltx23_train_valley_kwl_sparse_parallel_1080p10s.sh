#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH -t 03:00:00
#SBATCH -J ltx23-train-kwl-sparse
#SBATCH -o outputs/slurm/ltx23-train-kwl-sparse-%j.out
#SBATCH -e outputs/slurm/ltx23-train-kwl-sparse-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# KWL baseline: kernel/runtime-equivalent fusions used as the base for all variants.
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
export SGLANG_DIFFUSION_DECODE_PROFILE=1

# Sparse settings aligned with ltx-sparse-attn-bringup / existing piecewise scripts.
export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"

ROOT="${ROOT:-outputs/ltx23-train-valley-fiveway-1080p10s}"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, artifacts, inconsistent perspective, camera shake, harsh shadows, color banding, cartoonish rendering, unrealistic materials, uncanny valley effect, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, jittery movement, unnatural transitions, tilted camera, flat lighting, AI artifacts.}"
FORCE="${FORCE:-0}"
mkdir -p outputs/slurm "$ROOT"

COMMON_ARGS=(
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
  --prompt "$PROMPT"
  --return-file-paths-only true
)

run_variant() {
  local variant="$1"
  local gpu="$2"
  local port_offset="$3"
  shift 3
  local out_dir="$ROOT/$variant"
  mkdir -p "$out_dir"

  if [[ "$FORCE" != "1" && -s "$out_dir/out.mp4" && -s "$out_dir/perf.json" ]]; then
    echo "[skip] $variant already exists at $out_dir"
    return 0
  fi

  echo "[run] $variant gpu=$gpu -> $out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    .conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
      "${COMMON_ARGS[@]}" \
      --master-port "$((30005 + port_offset))" \
      --scheduler-port "$((5578 + port_offset))" \
      --port "$((30000 + port_offset))" \
      "$@" \
      --output-file-path "$out_dir/out.mp4" \
      --perf-dump-path "$out_dir/perf.json"

    VARIANT="$variant" OUT_DIR="$out_dir" .conda/ltx23/bin/python scripts/summarize_ltx23_sglang_perf.py \
      --out-dir "$out_dir" \
      --variant "$variant"
  ) >"$out_dir/run.log" 2>"$out_dir/run.err"
}

pids=()
run_variant "kwl" 0 0 &
pids+=("$!")
run_variant "kwl_sparse_stage1" 1 10 \
  --component-attention-backends.transformer piecewise_attn &
pids+=("$!")
run_variant "kwl_sparse_stage2" 2 20 \
  --component-attention-backends.transformer_2 piecewise_attn &
pids+=("$!")
run_variant "kwl_sparse_stage1_stage2" 3 30 \
  --component-attention-backends.transformer piecewise_attn \
  --component-attention-backends.transformer_2 piecewise_attn &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" != "0" ]]; then
  echo "At least one SGLang variant failed. Check $ROOT/*/run.err" >&2
  exit "$status"
fi

echo "[done] SGLang variants complete under $ROOT"
