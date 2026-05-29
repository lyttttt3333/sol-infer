#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH -J ltx23-kwl-sparse-ablate
#SBATCH -o outputs/slurm/ltx23-kwl-sparse-ablate-%j.out
#SBATCH -e outputs/slurm/ltx23-kwl-sparse-ablate-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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

ROOT="${ROOT:-outputs/ltx23-kwl-sparse-stage-ablation-color-1080p10s}"
PROMPT="${PROMPT:-A vibrant neon street festival at night with saturated lanterns, holographic signs, dancers in bright silk costumes, confetti in the air, and reflections on rain-slick pavement, cinematic camera glide}"
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
  shift
  local out_dir="$ROOT/$variant"
  mkdir -p "$out_dir"

  if [[ "$FORCE" != "1" && -s "$out_dir/out.mp4" && -s "$out_dir/perf.json" ]]; then
    echo "[skip] $variant already exists at $out_dir"
    return
  fi

  echo "[run] $variant -> $out_dir"
  .conda/ltx23/bin/python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --output-file-path "$out_dir/out.mp4" \
    --perf-dump-path "$out_dir/perf.json"

  VARIANT="$variant" OUT_DIR="$out_dir" .conda/ltx23/bin/python scripts/summarize_ltx23_sglang_perf.py \
    --out-dir "$out_dir" \
    --variant "$variant"
}

run_variant "kwl"
run_variant "kwl_sparse_stage1" \
  --component-attention-backends.transformer piecewise_attn
run_variant "kwl_sparse_stage2" \
  --component-attention-backends.transformer_2 piecewise_attn
run_variant "kwl_sparse_stage1_stage2" \
  --component-attention-backends.transformer piecewise_attn \
  --component-attention-backends.transformer_2 piecewise_attn

.conda/ltx23/bin/python - "$ROOT" "$PROMPT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
prompt = sys.argv[2]
variants = [
    ("kwl", "KWL"),
    ("kwl_sparse_stage1", "KWL + sparse stage 1"),
    ("kwl_sparse_stage2", "KWL + sparse stage 2"),
    ("kwl_sparse_stage1_stage2", "KWL + sparse stage 1+2"),
]

rows = []
for key, label in variants:
    summary_path = root / key / "summary.json"
    data = json.loads(summary_path.read_text())
    rows.append(
        {
            "variant": key,
            "label": label,
            "total_s": data.get("total_s"),
            "denoise_s": data.get("denoise_s"),
            "refine_s": data.get("refine_s"),
            "dit_s": data.get("dit_s"),
            "decode_s": data.get("decode_s"),
            "output_video": str(root / key / "out.mp4"),
            "perf_json": str(root / key / "perf.json"),
        }
    )

baseline = rows[0]["total_s"]
for row in rows:
    row["speedup_vs_kwl"] = baseline / row["total_s"] if baseline and row["total_s"] else None

aggregate = {
    "root": str(root),
    "prompt": prompt,
    "warmup": {"enabled": True, "steps": 30},
    "resolution": {"width": 1920, "height": 1088, "num_frames": 241, "fps": 24},
    "piecewise": {
        "sparsity": "0.9",
        "block_size": 64,
        "only_video_self": True,
        "approx_remainder": True,
        "route_mode": "score",
    },
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")

lines = [
    "# LTX2.3 KWL Sparse Stage Ablation",
    "",
    f"Prompt: {prompt}",
    "",
    "Warmup: enabled, 30 steps before measured run.",
    "",
    "| Variant | E2E s | DiT s | Stage 1 s | Stage 2 s | Decode s | Speedup vs KWL |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    speedup = row["speedup_vs_kwl"]
    speedup_text = f"{speedup:.3f}x" if speedup else "n/a"
    lines.append(
        f"| {row['label']} | {row['total_s']:.3f} | {row['dit_s']:.3f} | "
        f"{row['denoise_s']:.3f} | {row['refine_s']:.3f} | {row['decode_s']:.3f} | {speedup_text} |"
    )
(root / "table.md").write_text("\n".join(lines) + "\n")
print(json.dumps(aggregate, indent=2))
PY

echo "[done] root: $ROOT"
echo "[done] summary: $ROOT/summary.json"
echo "[done] table: $ROOT/table.md"
