#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH -t 04:00:00
#SBATCH -J ltx23-branch-same-noise
#SBATCH -o outputs/slurm/ltx23-branch-same-noise-%j.out
#SBATCH -e outputs/slurm/ltx23-branch-same-noise-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm outputs/.cache/huggingface outputs/.cache/torch outputs/.cache/triton outputs/.tmp
export HF_HOME="$PWD/outputs/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/outputs/.cache"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export TRITON_CACHE_DIR="$PWD/outputs/.cache/triton"
export TMPDIR="$PWD/outputs/.tmp"

if [[ ! -e outputs/LTX-2.3-local ]]; then
  ln -s ltx23-diffusers-local-view outputs/LTX-2.3-local
fi

export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

ROOT="${ROOT:-outputs/ltx23-branch-baselines-same-noise-1080p10s}"
MODEL_DIR="${MODEL_DIR:-outputs/ltx23-diffusers-local-view}"
LTX_MODEL_PATH="${LTX_MODEL_PATH:-outputs/LTX-2.3-local}"
DIFFUSERS_PRETRAINED="${DIFFUSERS_PRETRAINED:-outputs/ltx23-diffusers-official-runtime}"
DISTILLED_LORA="${DISTILLED_LORA:-$MODEL_DIR/ltx-2.3-22b-distilled-lora-384.safetensors}"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"
FORCE="${FORCE:-0}"

DIFF_DIR="$ROOT/diffusers_corrected_oldlora"
SHARED_DIR="$ROOT/shared_noise"
mkdir -p outputs/slurm "$DIFF_DIR" "$SHARED_DIR" "$ROOT"

DIFFUSER_ARGS=(
  --pretrained-model-id "$DIFFUSERS_PRETRAINED"
  --model-dir "$MODEL_DIR"
  --runtime-model-dir "$ROOT/diffusers_runtime"
  --local-files-only
  --output-dir "$DIFF_DIR"
  --output-video-path "$DIFF_DIR/out.mp4"
  --dump-stage1-initial-latents-dir "$SHARED_DIR"
  --dump-stage2-renoise-dir "$SHARED_DIR"
  --prompt "$PROMPT"
  --negative-prompt "$NEGATIVE_PROMPT"
  --width 1920
  --height 1088
  --num-frames 241
  --fps 24
  --seed 42
  --guidance-scale 3.0
  --stage2-guidance-scale 1.0
  --stg-scale 1.0
  --modality-scale 3.0
  --guidance-rescale 0.7
  --audio-guidance-scale 7.0
  --audio-stg-scale 1.0
  --audio-modality-scale 3.0
  --audio-guidance-rescale 0.7
  --spatio-temporal-guidance-blocks 28
  --use-cross-timestep
  --stage1-steps 30
  --stage2-steps 3
  --stage2-sigmas 0.909375 0.725 0.421875
  --distilled-lora-path "$DISTILLED_LORA"
  --stage1-lora-strength 0.0
  --stage2-lora-strength 1.0
  --dtype bf16
  --device cuda
  --enable-vae-tiling
  --warmup
  --actual-runs 1
)

if [[ "$FORCE" == "1" || ! -s "$DIFF_DIR/out.mp4" || ! -s "$SHARED_DIR/diffusers_stage1_video_initial.pt" || ! -s "$SHARED_DIR/diffusers_stage2_video_noise.pt" ]]; then
  echo "[run] Diffusers corrected old-LoRA noise source -> $DIFF_DIR"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/outputs/python_deps/ltx23_diffusers:$PYTHONPATH" \
    .conda/ltx23/bin/python scripts/benchmark_ltx23_diffusers_twostage.py "${DIFFUSER_ARGS[@]}"
else
  echo "[skip] Diffusers corrected old-LoRA outputs already exist"
fi

COMMON_ARGS=(
  --model-path "$LTX_MODEL_PATH"
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

apply_same_noise_env() {
  export SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH="$SHARED_DIR/diffusers_stage1_video_initial.pt"
  export SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH="$SHARED_DIR/diffusers_stage1_audio_initial.pt"
  export SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH="$SHARED_DIR/diffusers_stage2_video_noise.pt"
  export SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH="$SHARED_DIR/diffusers_stage2_audio_noise.pt"
  export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$1/latents"
  export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$1/latents"
}

apply_kwl_env() {
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
}

apply_sparse_env() {
  export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
  export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
  export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
  export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"
}

apply_fp4_env() {
  export SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND="${SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND:-cudnn}"
  export SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND="${SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND:-flashinfer}"
  export SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU="${SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU:-1}"
  export SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE="${SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE:-1}"
  export SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE="${SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE:-1}"
}

run_variant() {
  local variant="$1"
  local gpu="$2"
  local port_offset="$3"
  local mode="$4"
  shift 4
  local out_dir="$ROOT/$variant"
  mkdir -p "$out_dir"
  if [[ "$FORCE" != "1" && -s "$out_dir/out.mp4" && -s "$out_dir/perf.json" ]]; then
    echo "[skip] $variant already exists at $out_dir"
    return 0
  fi
  echo "[run] $variant mode=$mode gpu=$gpu -> $out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    apply_same_noise_env "$out_dir"
    case "$mode" in
      dense) ;;
      kwl) apply_kwl_env ;;
      sparse) apply_sparse_env ;;
      nvfp4_piecewise) apply_kwl_env; apply_sparse_env; apply_fp4_env ;;
      *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    .conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
      "${COMMON_ARGS[@]}" \
      --master-port "$((30105 + port_offset))" \
      --scheduler-port "$((5678 + port_offset))" \
      --port "$((30100 + port_offset))" \
      "$@" \
      --output-file-path "$out_dir/out.mp4" \
      --perf-dump-path "$out_dir/perf.json"
    VARIANT="$variant" OUT_DIR="$out_dir" .conda/ltx23/bin/python scripts/summarize_ltx23_sglang_perf.py \
      --out-dir "$out_dir" \
      --variant "$variant"
  ) >"$out_dir/run.log" 2>"$out_dir/run.err"
}

pids=()
run_variant "sglang_dense_main" 0 0 dense &
pids+=("$!")
run_variant "kwl_fusion_report" 1 10 kwl &
pids+=("$!")
run_variant "sparse_bringup_piecewise" 2 20 sparse \
  --component-attention-backends.transformer piecewise_attn \
  --component-attention-backends.transformer_2 piecewise_attn &
pids+=("$!")
run_variant "nvfp4_piecewise" 3 30 nvfp4_piecewise \
  --attention-backend piecewise_attn \
  --component-paths.transformer outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat \
  --component-paths.transformer_2 outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat &
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

.conda/ltx23/bin/python - "$ROOT" <<'PY2'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
diff_perf = root / 'diffusers_corrected_oldlora' / 'perf_diffusers.json'
if diff_perf.exists():
    d = json.loads(diff_perf.read_text())
    timings = d.get('timings_s', {})
    rows.append({
        'variant': 'diffusers_corrected_oldlora',
        'branch_source': 'official Diffusers + local scheduler reset',
        'total_s': d.get('strict_pipeline_s'),
        'stage1_s': timings.get('actual.stage1_pipeline_s'),
        'stage2_s': timings.get('actual.stage2_pipeline_s'),
        'decode_s': timings.get('actual.video_vae_decode_s'),
        'output_video': str(root / 'diffusers_corrected_oldlora' / 'out.mp4'),
        'notes': 'Stage2 scheduler reset; old local distilled LoRA; source of shared noise dumps.',
    })
for variant, source, notes in [
    ('sglang_dense_main', 'origin/main dense SGLang setting', 'No KWL, no sparse, no NVFP4.'),
    ('kwl_fusion_report', 'local/ltx2-dit-fusion-report KWL setting', 'Kernel-wise lossless fusions enabled.'),
    ('sparse_bringup_piecewise', 'origin/ltx-sparse-attn-bringup setting', 'Piecewise sparse attention, sparsity=0.9, block=64, video self only.'),
    ('nvfp4_piecewise', 'origin/ltx2-nvfp4-two-stage-cleanup + local fused FP4 setting', 'Selective NVFP4 video attn/ffn transformers plus piecewise attention.'),
]:
    p = root / variant / 'summary.json'
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    rows.append({
        'variant': variant,
        'branch_source': source,
        'total_s': s.get('total_s'),
        'stage1_s': s.get('denoise_s'),
        'stage2_s': s.get('refine_s'),
        'decode_s': s.get('decode_s'),
        'output_video': str(root / variant / 'out.mp4'),
        'notes': notes,
    })
summary = {
    'root': str(root),
    'prompt': 'A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail',
    'resolution': {'width': 1920, 'height': 1088, 'num_frames': 241, 'fps': 24},
    'same_noise_source': str(root / 'shared_noise'),
    'rows': rows,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY2

echo "[done] branch baseline same-noise outputs under $ROOT"
