#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -J ltx23-kwl-pisa-s2
#SBATCH -o outputs/slurm/ltx23-kwl-pisa-s2-%j.out
#SBATCH -e outputs/slurm/ltx23-kwl-pisa-s2-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# kwl = kernel-wise lossless baseline: kernel/runtime-equivalent fusions only.
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

# PISA/piecewise settings aligned with ltx-sparse-attn-bringup.
export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"

MODE="${MODE:-kwl_pisa_stage2}"
export MODE
PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"
export PROMPT
ROOT="${ROOT:-outputs/ltx23-kwl-stage2-pisa-1080p10s}"
OUT_DIR="${OUT_DIR:-$ROOT/$MODE}"
export OUT_DIR
mkdir -p outputs/slurm "$OUT_DIR"

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
  --prompt "$PROMPT"
  --return-file-paths-only true
)

case "$MODE" in
  kwl)
    ;;
  kwl_pisa_stage2)
    # Only the stage-2 DiT component uses PISA. Stage 1 remains on the kwl dense path.
    COMMON_ARGS+=(--component-attention-backends.transformer_2 piecewise_attn)
    ;;
  kwl_pisa_all)
    # Diagnostic mode: all eligible video self-attention layers use PISA.
    COMMON_ARGS+=(--attention-backend piecewise_attn)
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use kwl, kwl_pisa_stage2, or kwl_pisa_all." >&2
    exit 2
    ;;
esac

.conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  "${COMMON_ARGS[@]}" \
  --output-file-path "$OUT_DIR/out.mp4" \
  --perf-dump-path "$OUT_DIR/perf.json"

.conda/ltx23/bin/python - <<'PY2'
import json, os
out_dir = os.environ['OUT_DIR']
path = os.path.join(out_dir, 'perf.json')
d = json.load(open(path))
steps = {x['name']: x['duration_ms'] for x in d.get('steps', [])}
total_s = d.get('total_duration_ms', 0) / 1000
summary = {
    'mode': os.environ.get('MODE'),
    'output_dir': out_dir,
    'commit_hash': d.get('commit_hash'),
    'kwl_reference_total_s': 59.33186146011576,
    'target_speedup_vs_kwl': 1.4,
    'target_total_s_for_1p4x': 59.33186146011576 / 1.4,
    'total_s': total_s,
    'speedup_vs_kwl_reference': 59.33186146011576 / total_s if total_s else None,
    'denoise_s': steps.get('LTX2AVDenoisingStage', 0) / 1000,
    'refine_s': steps.get('LTX2RefinementStage', 0) / 1000,
    'dit_s': (steps.get('LTX2AVDenoisingStage', 0) + steps.get('LTX2RefinementStage', 0)) / 1000,
    'decode_s': steps.get('LTX2AVDecodingStage', 0) / 1000,
    'piecewise_sparsity': os.environ.get('SGLANG_PIECEWISE_ATTN_SPARSITY'),
    'piecewise_block_size': os.environ.get('SGLANG_PIECEWISE_ATTN_BLOCK_SIZE'),
    'piecewise_only_video_self': os.environ.get('SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF'),
    'piecewise_approx_remainder': os.environ.get('SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER'),
    'piecewise_route_mode': os.environ.get('SGLANG_PIECEWISE_ATTN_ROUTE_MODE'),
    'piecewise_compile_route': os.environ.get('SGLANG_PIECEWISE_ATTN_COMPILE_ROUTE'),
    'piecewise_compile_route_mode': os.environ.get('SGLANG_PIECEWISE_ATTN_COMPILE_ROUTE_MODE'),
}
open(os.path.join(out_dir, 'summary.json'), 'w').write(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY2
