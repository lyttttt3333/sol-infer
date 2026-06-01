#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -J ltx23-fp4-pw
#SBATCH -o outputs/slurm/ltx23-fp4-pw-%j.out
#SBATCH -e outputs/slurm/ltx23-fp4-pw-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh
mkdir -p outputs/.tmp outputs/.cache/huggingface outputs/.cache/xdg outputs/.cache/torch outputs/.cache/triton outputs/.cache/torchinductor outputs/.cache/torch_extensions outputs/.cache/cuda outputs/.cache/sgl_diffusion
export TMPDIR="$PWD/outputs/.tmp"
export HF_HOME="$PWD/outputs/.cache/huggingface"
export HF_HUB_CACHE="$PWD/outputs/.cache/huggingface/hub"
export XDG_CACHE_HOME="$PWD/outputs/.cache/xdg"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export TRITON_CACHE_DIR="$PWD/outputs/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$PWD/outputs/.cache/torch_extensions"
export CUDA_CACHE_PATH="$PWD/outputs/.cache/cuda"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export SGLANG_DIFFUSION_CACHE_ROOT="$PWD/outputs/.cache/sgl_diffusion"

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
export SGLANG_LTX2_FUSED_CA_DUAL_MODULATE=1
export SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1
export SGLANG_LTX2_FUSED_RESIDUAL_GATE=1
export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1
export SGLANG_LTX2_COMPILE_GATE_TO_OUT=1
export SGLANG_LTX2_FUSED_AUDIO_QKVG=1
export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=1
export SGLANG_LTX2_VAE_COMPILE_MODE="${SGLANG_LTX2_VAE_COMPILE_MODE:-max-autotune-no-cudagraphs}"
export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=1
export SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND="${SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND:-sgl_kernel}"
export SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND="${SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND:-flashinfer}"
export SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU="${SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU:-1}"
export SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE="${SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE:-1}"
export SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE="${SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE:-1}"
export SGLANG_LTX2_FP4_SHARED_QKV="${SGLANG_LTX2_FP4_SHARED_QKV:-1}"
export SGLANG_LTX2_FP4_SHARED_Q_GATE="${SGLANG_LTX2_FP4_SHARED_Q_GATE:-1}"
export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"

export MODEL_PATH="${MODEL_PATH:-outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493}"
export PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"
export OUT_DIR="${OUT_DIR:-outputs/ltx23-nonhq-quality-aligned-nvfp4-piecewise-1080p10s/sgl_kernel_localpath}"
mkdir -p outputs/slurm "$OUT_DIR"

if [[ "${SGLANG_DIFFUSION_LTX2_EVENT_PROFILE:-0}" == "1" ]]; then
  export SGLANG_DIFFUSION_LTX2_PROFILE_PATH="${SGLANG_DIFFUSION_LTX2_PROFILE_PATH:-$OUT_DIR/ltx2_event_profile.json}"
fi

.conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate   --model-path "$MODEL_PATH"   --backend auto   --attention-backend piecewise_attn   --pipeline-class-name LTX2TwoStagePipeline   --num-gpus 1   --performance-mode speed   --ltx2-two-stage-device-mode resident   --warmup true   --warmup-steps 30   --height 1088   --width 1920   --num-frames 241   --fps 24   --seed 42   --num-inference-steps 30   --guidance-scale 3.0   --prompt "$PROMPT"   --return-file-paths-only true   --component-paths.transformer outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat   --component-paths.transformer_2 outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat   --output-file-path "$OUT_DIR/out.mp4"   --perf-dump-path "$OUT_DIR/perf.json"

.conda/ltx23/bin/python - <<'PY2'
import json, os
out_dir = os.environ.get('OUT_DIR', 'outputs/ltx23-nonhq-quality-aligned-nvfp4-piecewise-1080p10s/sgl_kernel_localpath')
path = os.path.join(out_dir, 'perf.json')
d = json.load(open(path))
steps = {x['name']: x['duration_ms'] for x in d.get('steps', [])}
summary = {
    'output_dir': out_dir,
    'model_path': os.environ.get('MODEL_PATH'),
    'fp4_quantize_backend': os.environ.get('SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND'),
    'fp4_gemm_backend': os.environ.get('SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND'),
    'fp4_fused_proj_in_bias_gelu': os.environ.get('SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU'),
    'fp4_fused_proj_out_bias_gate': os.environ.get('SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE'),
    'fp4_fused_attn_to_out_bias_gate': os.environ.get('SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE'),
    'fp4_shared_qkv': os.environ.get('SGLANG_LTX2_FP4_SHARED_QKV'),
    'fp4_shared_q_gate': os.environ.get('SGLANG_LTX2_FP4_SHARED_Q_GATE'),
    'fused_ca_dual_modulate': os.environ.get('SGLANG_LTX2_FUSED_CA_DUAL_MODULATE'),
    'fused_ada_values_packed': os.environ.get('SGLANG_LTX2_FUSED_ADA_VALUES_PACKED'),
    'fused_ada_direct': os.environ.get('SGLANG_LTX2_FUSED_ADA_DIRECT'),
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
    'speedup_vs_59_332': 59.332 / (d.get('total_duration_ms', 0) / 1000),
}
open(os.path.join(out_dir, 'summary.json'), 'w').write(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY2
