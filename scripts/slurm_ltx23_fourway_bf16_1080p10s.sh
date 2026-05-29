#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -J ltx23-4way-bf16
#SBATCH -o outputs/slurm/ltx23-4way-bf16-%j.out
#SBATCH -e outputs/slurm/ltx23-4way-bf16-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

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

# Match ltx-sparse-attn-bringup semantics when MODE=bf16_sparse.
export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"

MODE="${MODE:-bf16_dense}"
PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"
OUT_DIR="${OUT_DIR:-outputs/ltx23-dev-1080p10s-fourway-bf16-${MODE}}"
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

if [[ "$MODE" == "bf16_sparse" ]]; then
  COMMON_ARGS+=(--attention-backend piecewise_attn)
elif [[ "$MODE" != "bf16_dense" ]]; then
  echo "Unsupported MODE=$MODE" >&2
  exit 2
fi

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
summary = {
    'output_dir': out_dir,
    'mode': os.environ.get('MODE'),
    'piecewise_sparsity': os.environ.get('SGLANG_PIECEWISE_ATTN_SPARSITY'),
    'piecewise_block_size': os.environ.get('SGLANG_PIECEWISE_ATTN_BLOCK_SIZE'),
    'piecewise_only_video_self': os.environ.get('SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF'),
    'piecewise_approx_remainder': os.environ.get('SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER'),
    'piecewise_route_mode': os.environ.get('SGLANG_PIECEWISE_ATTN_ROUTE_MODE'),
    'total_s': d.get('total_duration_ms', 0) / 1000,
    'denoise_s': steps.get('LTX2AVDenoisingStage', 0) / 1000,
    'refine_s': steps.get('LTX2RefinementStage', 0) / 1000,
    'dit_s': (steps.get('LTX2AVDenoisingStage', 0) + steps.get('LTX2RefinementStage', 0)) / 1000,
    'decode_s': steps.get('LTX2AVDecodingStage', 0) / 1000,
}
open(os.path.join(out_dir, 'summary.json'), 'w').write(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY2
