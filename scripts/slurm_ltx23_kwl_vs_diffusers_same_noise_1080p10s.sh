#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH -J ltx23-same-noise
#SBATCH -o outputs/slurm/ltx23-same-noise-%j.out
#SBATCH -e outputs/slurm/ltx23-same-noise-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

ROOT="${ROOT:-outputs/ltx23-kwl-vs-diffusers-same-noise-1080p10s}"
MODEL_DIR="${MODEL_DIR:-/home/yitongl/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493}"
DIFFUSERS_PRETRAINED="${DIFFUSERS_PRETRAINED:-diffusers/LTX-2.3-Diffusers}"
PROMPT="${PROMPT:-A cinematic aerial shot of clouds moving across a mountain ridge at sunrise}"
FORCE="${FORCE:-0}"
WARMUP="${WARMUP:-0}"

NEGATIVE_PROMPT="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."

DIFF_DIR="$ROOT/diffusers"
KWL_DIR="$ROOT/kwl_same_noise"
SHARED_DIR="$ROOT/shared_noise"
COMPARE_MP4="$ROOT/diffusers-vs-kwl-same-noise-side-by-side.mp4"
mkdir -p outputs/slurm "$DIFF_DIR" "$KWL_DIR" "$SHARED_DIR"

DIFFUSER_WARMUP_ARGS=()
SGLANG_WARMUP_ARGS=()
if [[ "$WARMUP" == "1" ]]; then
  DIFFUSER_WARMUP_ARGS+=(--warmup)
  SGLANG_WARMUP_ARGS+=(--warmup true --warmup-steps 30)
fi

if [[ "$FORCE" == "1" || ! -s "$DIFF_DIR/out.mp4" || ! -s "$SHARED_DIR/diffusers_stage1_video_initial.pt" || ! -s "$SHARED_DIR/diffusers_stage2_video_noise.pt" ]]; then
  echo "[run] Diffusers official path -> $DIFF_DIR"
  PYTHONPATH="$PWD/outputs/python_deps/ltx23_diffusers:$PYTHONPATH" .conda/ltx23/bin/python scripts/benchmark_ltx23_diffusers_twostage.py \
    --pretrained-model-id "$DIFFUSERS_PRETRAINED" \
    --model-dir "$MODEL_DIR" \
    --local-files-only \
    --output-dir "$DIFF_DIR" \
    --output-video-path "$DIFF_DIR/out.mp4" \
    --dump-stage1-initial-latents-dir "$SHARED_DIR" \
    --dump-stage2-renoise-dir "$SHARED_DIR" \
    --prompt "$PROMPT" \
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
    --actual-runs 1 \
    "${DIFFUSER_WARMUP_ARGS[@]}"
else
  echo "[skip] Diffusers outputs already exist in $DIFF_DIR"
fi

# kwl = kernel-wise lossless path: kernel/runtime-equivalent fusions only.
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

export SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH="$SHARED_DIR/diffusers_stage1_video_initial.pt"
export SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH="$SHARED_DIR/diffusers_stage1_audio_initial.pt"
export SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH="$SHARED_DIR/diffusers_stage2_video_noise.pt"
export SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH="$SHARED_DIR/diffusers_stage2_audio_noise.pt"
export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$KWL_DIR/latents"
export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$KWL_DIR/latents"

if [[ "$FORCE" == "1" || ! -s "$KWL_DIR/out.mp4" || ! -s "$KWL_DIR/perf.json" ]]; then
  echo "[run] KWL with Diffusers initial noise -> $KWL_DIR"
  .conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path Lightricks/LTX-2.3 \
    --backend auto \
    --pipeline-class-name LTX2TwoStagePipeline \
    --num-gpus 1 \
    --performance-mode speed \
    --ltx2-two-stage-device-mode resident \
    "${SGLANG_WARMUP_ARGS[@]}" \
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
    --output-file-path "$KWL_DIR/out.mp4" \
    --perf-dump-path "$KWL_DIR/perf.json"
else
  echo "[skip] KWL outputs already exist in $KWL_DIR"
fi

.conda/ltx23/bin/python - "$SHARED_DIR" "$KWL_DIR/latents" "$ROOT/latent_alignment.json" <<PY
import json
import sys
from pathlib import Path

import torch

shared = Path(sys.argv[1])
kwl = Path(sys.argv[2])
out = Path(sys.argv[3])

def load(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    return payload["latents"]

def compare(diff_path, sgl_path):
    a = load(diff_path)
    b = load(sgl_path)
    a_cast = a.to(dtype=b.dtype)
    delta = (a_cast.float() - b.float()).abs()
    raw_delta = (a.float() - b.float()).abs()
    return {
        "diffusers_path": str(diff_path),
        "sglang_path": str(sgl_path),
        "shape_equal": list(a.shape) == list(b.shape),
        "diffusers_dtype": str(a.dtype),
        "sglang_dtype": str(b.dtype),
        "max_abs_after_cast": float(delta.max().item()),
        "mean_abs_after_cast": float(delta.mean().item()),
        "max_abs_raw": float(raw_delta.max().item()),
        "mean_abs_raw": float(raw_delta.mean().item()),
    }

pairs = {
    "stage1_video_initial": (
        shared / "diffusers_stage1_video_initial.pt",
        kwl / "sglang_stage1_video_initial.pt",
    ),
    "stage1_audio_initial": (
        shared / "diffusers_stage1_audio_initial.pt",
        kwl / "sglang_stage1_audio_initial.pt",
    ),
    "stage2_video_noise": (
        shared / "diffusers_stage2_video_noise.pt",
        kwl / "sglang_stage2_video_noise.pt",
    ),
    "stage2_audio_noise": (
        shared / "diffusers_stage2_audio_noise.pt",
        kwl / "sglang_stage2_audio_noise.pt",
    ),
}
result = {}
for name, (diff_path, sgl_path) in pairs.items():
    result[name] = compare(diff_path, sgl_path)
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

.conda/ltx23/bin/python scripts/make_side_by_side_video.py \
  --left "$DIFF_DIR/out.mp4" \
  --right "$KWL_DIR/out.mp4" \
  --left-label "Diffusers official" \
  --right-label "KWL same noise" \
  --out "$COMPARE_MP4"

.conda/ltx23/bin/python - "$ROOT" <<PY
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {"root": str(root)}
for name, rel in {
    "diffusers": "diffusers/perf_diffusers.json",
    "kwl_same_noise": "kwl_same_noise/perf.json",
    "latent_alignment": "latent_alignment.json",
}.items():
    path = root / rel
    if path.exists():
        data = json.load(open(path))
        if name == "diffusers":
            summary[name] = {
                "strict_pipeline_s": data.get("strict_pipeline_s"),
                "stage1_pipeline_s": data.get("timings_s", {}).get("actual.stage1_pipeline_s"),
                "stage2_pipeline_s": data.get("timings_s", {}).get("actual.stage2_pipeline_s"),
                "decode_s": data.get("timings_s", {}).get("actual.video_vae_decode_s"),
            }
        elif name == "kwl_same_noise":
            steps = {x.get("name"): x.get("duration_ms", 0) / 1000 for x in data.get("steps", [])}
            summary[name] = {
                "total_s": data.get("total_duration_ms", 0) / 1000,
                "stage1_denoise_s": steps.get("LTX2AVDenoisingStage"),
                "stage2_refine_s": steps.get("LTX2RefinementStage"),
                "decode_s": steps.get("LTX2AVDecodingStage"),
            }
        else:
            summary[name] = data
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[done] side-by-side: $COMPARE_MP4"
echo "[done] summary: $ROOT/summary.json"
