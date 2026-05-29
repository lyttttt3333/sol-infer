#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH -J ltx23-official-hq-kwl
#SBATCH -o outputs/slurm/ltx23-official-hq-kwl-%j.out
#SBATCH -e outputs/slurm/ltx23-official-hq-kwl-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/outputs/.cache/huggingface"
export HF_HUB_CACHE="$PWD/outputs/.cache/huggingface/hub"
export XDG_CACHE_HOME="$PWD/outputs/.cache/xdg"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export TRITON_CACHE_DIR="$PWD/outputs/.cache/triton"
export TMPDIR="$PWD/outputs/.tmp"
export SGLANG_DIFFUSION_CACHE_ROOT="$PWD/outputs/.cache/sgl_diffusion"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export LTX23_OFFICIAL_KWL_ADALN="${LTX23_OFFICIAL_KWL_ADALN:-1}"
export LTX23_OFFICIAL_KWL_QKNORM="${LTX23_OFFICIAL_KWL_QKNORM:-1}"
export LTX23_OFFICIAL_KWL_QKNORM_ROPE="${LTX23_OFFICIAL_KWL_QKNORM_ROPE:-1}"
export LTX23_OFFICIAL_KWL_FFN_PROJ_IN_GELU="${LTX23_OFFICIAL_KWL_FFN_PROJ_IN_GELU:-1}"

export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

OFFICIAL_SRC="${OFFICIAL_SRC:-outputs/LTX-2-official-main}"
OFFICIAL_DEPS="$PWD/outputs/python_deps/ltx23_official"
DIFFUSERS_DEPS="$PWD/outputs/python_deps/ltx23_diffusers"
export PYTHONPATH="$PWD/$OFFICIAL_SRC/packages/ltx-core/src:$PWD/$OFFICIAL_SRC/packages/ltx-pipelines/src:$OFFICIAL_DEPS:$DIFFUSERS_DEPS:$PWD/python:${PYTHONPATH:-}"

MODEL_COMPONENT_DIR="${MODEL_COMPONENT_DIR:-outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493}"
OFFICIAL_MODEL_DIR="${OFFICIAL_MODEL_DIR:-outputs/LTX-2.3-official-files}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OFFICIAL_MODEL_DIR/ltx-2.3-22b-dev.safetensors}"
DISTILLED_LORA="${DISTILLED_LORA:-$OFFICIAL_MODEL_DIR/ltx-2.3-22b-distilled-lora-384-1.1.safetensors}"
SPATIAL_UPSAMPLER="${SPATIAL_UPSAMPLER:-$MODEL_COMPONENT_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
GEMMA_ROOT="${GEMMA_ROOT:-$MODEL_COMPONENT_DIR}"
ROOT="${ROOT:-outputs/ltx23-official-hq-kwl-pipeline-1080p10s}"
OUT_DIR="$ROOT/official_hq_kwl"
OUT_VIDEO="$OUT_DIR/out.mp4"
PROMPT="${PROMPT:-A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.}"
FORCE="${FORCE:-1}"

mkdir -p outputs/slurm outputs/.cache/huggingface outputs/.cache/xdg outputs/.cache/torch outputs/.cache/triton outputs/.tmp "$OFFICIAL_MODEL_DIR" "$OUT_DIR"

if [[ ! -d "$OFFICIAL_SRC/packages/ltx-pipelines/src/ltx_pipelines" ]]; then
  echo "[error] missing official source at $OFFICIAL_SRC"
  echo "        clone it under current repo: git clone --depth 1 https://github.com/Lightricks/LTX-2.git $OFFICIAL_SRC"
  exit 1
fi

# The upstream package __init__ eagerly imports every pipeline. For direct module execution here,
# keep the official HQ module unchanged but avoid unrelated import side effects.
INIT_FILE="$OFFICIAL_SRC/packages/ltx-pipelines/src/ltx_pipelines/__init__.py"
if grep -q "from ltx_pipelines.a2vid_two_stage" "$INIT_FILE"; then
  cp "$INIT_FILE" "$INIT_FILE.official_bak"
  printf '%s\n' '"""LTX-2 Pipelines package, local lightweight init for direct module execution."""' '' '__all__ = []' > "$INIT_FILE"
fi

if [[ ! -s "$CHECKPOINT_PATH" || ! -s "$DISTILLED_LORA" ]]; then
  echo "[download] official LTX-2.3 monolithic checkpoint / HQ LoRA into $OFFICIAL_MODEL_DIR"
  .conda/ltx23/bin/python - "$OFFICIAL_MODEL_DIR" <<'PYDL'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

local_dir = Path(sys.argv[1])
local_dir.mkdir(parents=True, exist_ok=True)
for filename in [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]:
    target = local_dir / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"[download skip] {target}", flush=True)
        continue
    path = hf_hub_download(
        repo_id="Lightricks/LTX-2.3",
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[download done] {path}", flush=True)
PYDL
fi

for required in "$CHECKPOINT_PATH" "$DISTILLED_LORA" "$SPATIAL_UPSAMPLER" "$GEMMA_ROOT/tokenizer/tokenizer.model" "$GEMMA_ROOT/tokenizer/preprocessor_config.json"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing required official pipeline asset: $required"
    exit 1
  fi
done

if [[ "$FORCE" == "1" ]]; then
  rm -f "$OUT_VIDEO" "$OUT_DIR/summary.json" "$OUT_DIR/run_command.txt"
elif [[ -s "$OUT_VIDEO" && -s "$OUT_DIR/summary.json" ]]; then
  echo "[skip] official HQ output already exists at $OUT_VIDEO"
  exit 0
fi

cat > "$OUT_DIR/run_command.txt" <<EOF
python -m scripts.ltx23_official_kwl_ops \\
  --checkpoint-path "$CHECKPOINT_PATH" \\
  --distilled-lora "$DISTILLED_LORA" \\
  --distilled-lora-strength-stage-1 0.25 \\
  --distilled-lora-strength-stage-2 0.5 \\
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \\
  --gemma-root "$GEMMA_ROOT" \\
  --prompt "$PROMPT" \\
  --negative-prompt "$NEGATIVE_PROMPT" \\
  --seed 42 --height 1088 --width 1920 --num-frames 241 --frame-rate 24 \\
  --num-inference-steps 15 \\
  --video-cfg-guidance-scale 3.0 --video-stg-guidance-scale 0.0 --video-rescale-scale 0.45 --a2v-guidance-scale 3.0 \\
  --audio-cfg-guidance-scale 7.0 --audio-stg-guidance-scale 0.0 --audio-rescale-scale 1.0 --v2a-guidance-scale 3.0 \\
  --video-stg-blocks --audio-stg-blocks --max-batch-size 1 \\
  --output-path "$OUT_VIDEO"
EOF

START_NS=$(date +%s%N)
echo "[run] official TI2VidTwoStagesHQPipeline -> $OUT_VIDEO"
echo "[run] checkpoint: $CHECKPOINT_PATH"
echo "[run] distilled lora: $DISTILLED_LORA"
echo "[run] spatial upsampler: $SPATIAL_UPSAMPLER"
.conda/ltx23/bin/python -m scripts.ltx23_official_kwl_ops \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --distilled-lora "$DISTILLED_LORA" \
  --distilled-lora-strength-stage-1 0.25 \
  --distilled-lora-strength-stage-2 0.5 \
  --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \
  --gemma-root "$GEMMA_ROOT" \
  --prompt "$PROMPT" \
  --negative-prompt "$NEGATIVE_PROMPT" \
  --seed 42 \
  --height 1088 \
  --width 1920 \
  --num-frames 241 \
  --frame-rate 24 \
  --num-inference-steps 15 \
  --video-cfg-guidance-scale 3.0 \
  --video-stg-guidance-scale 0.0 \
  --video-rescale-scale 0.45 \
  --a2v-guidance-scale 3.0 \
  --audio-cfg-guidance-scale 7.0 \
  --audio-stg-guidance-scale 0.0 \
  --audio-rescale-scale 1.0 \
  --v2a-guidance-scale 3.0 \
  --video-stg-blocks \
  --audio-stg-blocks \
  --max-batch-size 1 \
  --output-path "$OUT_VIDEO"
END_NS=$(date +%s%N)

.conda/ltx23/bin/python - "$OUT_DIR" "$OUT_VIDEO" "$START_NS" "$END_NS" <<'PYSUM'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
out_video = Path(sys.argv[2])
start_ns = int(sys.argv[3])
end_ns = int(sys.argv[4])
summary = {
    "variant": "official_ti2vid_two_stages_hq_kwl",
    "pipeline_source": "https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py",
    "output_video": str(out_video),
    "total_wall_s": (end_ns - start_ns) / 1e9,
    "prompt": "A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail",
    "seed": 42,
    "height": 1088,
    "width": 1920,
    "num_frames": 241,
    "fps": 24,
    "num_inference_steps_stage1": 15,
    "stage2_sigmas": [0.909375, 0.725, 0.421875, 0.0],
    "distilled_lora_strength_stage1": 0.25,
    "distilled_lora_strength_stage2": 0.5,
    "video_cfg_guidance_scale": 3.0,
    "video_stg_guidance_scale": 0.0,
    "video_rescale_scale": 0.45,
    "audio_cfg_guidance_scale": 7.0,
    "audio_stg_guidance_scale": 0.0,
    "audio_rescale_scale": 1.0,
}
out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PYSUM

echo "[done] official HQ output: $OUT_VIDEO"
