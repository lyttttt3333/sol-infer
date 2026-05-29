#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH -J ltx23-train-diffusers
#SBATCH -o outputs/slurm/ltx23-train-diffusers-%j.out
#SBATCH -e outputs/slurm/ltx23-train-diffusers-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

ROOT="${ROOT:-outputs/ltx23-train-valley-fiveway-1080p10s}"
MODEL_DIR="${MODEL_DIR:-/home/yitongl/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493}"
DIFFUSERS_PRETRAINED="${DIFFUSERS_PRETRAINED:-diffusers/LTX-2.3-Diffusers}"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"
FORCE="${FORCE:-0}"
OUT_DIR="$ROOT/diffusers"
mkdir -p outputs/slurm "$OUT_DIR"

if [[ "$FORCE" != "1" && -s "$OUT_DIR/out.mp4" && -s "$OUT_DIR/perf_diffusers.json" ]]; then
  echo "[skip] Diffusers output already exists at $OUT_DIR"
  exit 0
fi

echo "[run] Diffusers official path -> $OUT_DIR"
PYTHONPATH="$PWD/outputs/python_deps/ltx23_diffusers:$PYTHONPATH" .conda/ltx23/bin/python scripts/benchmark_ltx23_diffusers_twostage.py \
  --pretrained-model-id "$DIFFUSERS_PRETRAINED" \
  --model-dir "$MODEL_DIR" \
  --runtime-model-dir "$ROOT/diffusers_runtime" \
  --local-files-only \
  --output-dir "$OUT_DIR" \
  --output-video-path "$OUT_DIR/out.mp4" \
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
  --warmup \
  --actual-runs 1

.conda/ltx23/bin/python - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
perf = json.loads((out_dir / "perf_diffusers.json").read_text())
timings = perf.get("timings_s", {})
summary = {
    "variant": "diffusers",
    "output_dir": str(out_dir),
    "total_s": perf.get("strict_pipeline_s"),
    "stage1_pipeline_s": timings.get("actual.stage1_pipeline_s"),
    "stage2_pipeline_s": timings.get("actual.stage2_pipeline_s"),
    "decode_s": timings.get("actual.video_vae_decode_s"),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[done] Diffusers output: $OUT_DIR/out.mp4"
