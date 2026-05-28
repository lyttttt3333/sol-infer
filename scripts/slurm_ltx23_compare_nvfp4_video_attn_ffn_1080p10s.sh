#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -J ltx23-fp4-compare
#SBATCH -o outputs/slurm/ltx23-fp4-compare-%j.out
#SBATCH -e outputs/slurm/ltx23-fp4-compare-%j.err

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
export SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn

PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"
BASE_OUT="outputs/ltx23-compare-video-attn-ffn-baseline"
FP4_OUT="outputs/ltx23-compare-video-attn-ffn-nvfp4"
COMPARE_OUT="outputs/ltx23-compare-video-attn-ffn-side-by-side.mp4"

mkdir -p outputs/slurm "$BASE_OUT" "$FP4_OUT"

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

.conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  "${COMMON_ARGS[@]}" \
  --output-file-path "$BASE_OUT/out.mp4" \
  --perf-dump-path "$BASE_OUT/perf.json"

.conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  "${COMMON_ARGS[@]}" \
  --component-paths.transformer outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat \
  --component-paths.transformer_2 outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat \
  --output-file-path "$FP4_OUT/out.mp4" \
  --perf-dump-path "$FP4_OUT/perf.json"

.conda/ltx23/bin/python scripts/make_side_by_side_video.py \
  --left "$BASE_OUT/out.mp4" \
  --right "$FP4_OUT/out.mp4" \
  --out "$COMPARE_OUT" \
  --left-label "BF16 baseline" \
  --right-label "NVFP4 video attn+FFN"

.conda/ltx23/bin/python - <<'PY2'
import json
paths = {
    'baseline': 'outputs/ltx23-compare-video-attn-ffn-baseline/perf.json',
    'nvfp4': 'outputs/ltx23-compare-video-attn-ffn-nvfp4/perf.json',
}
summary = {}
for name, path in paths.items():
    d = json.load(open(path))
    steps = {x['name']: x['duration_ms'] for x in d.get('steps', [])}
    summary[name] = {
        'total_s': d.get('total_duration_ms', 0) / 1000,
        'denoise_s': steps.get('LTX2AVDenoisingStage', 0) / 1000,
        'refine_s': steps.get('LTX2RefinementStage', 0) / 1000,
        'dit_s': (steps.get('LTX2AVDenoisingStage', 0) + steps.get('LTX2RefinementStage', 0)) / 1000,
        'decode_s': steps.get('LTX2AVDecodingStage', 0) / 1000,
    }
summary['speedup_total'] = summary['baseline']['total_s'] / summary['nvfp4']['total_s']
summary['speedup_dit'] = summary['baseline']['dit_s'] / summary['nvfp4']['dit_s']
open('outputs/ltx23-compare-video-attn-ffn-summary.json', 'w').write(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY2
