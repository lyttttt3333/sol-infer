#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -J ltx23-kwl-alloff
#SBATCH -o outputs/slurm/ltx23-kwl-alloff-%j.out
#SBATCH -e outputs/slurm/ltx23-kwl-alloff-%j.err

set -euo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
ROOT="${ROOT:-outputs/ltx23-sglang-hq-kwl-lossless-ablation-1080p10s}"
SHARED_DIR="$ROOT/shared_noise"
OUT_DIR="$ROOT/kwl_all_off_no_compile"
mkdir -p outputs/slurm "$OUT_DIR"

for required in \
  "$ROOT/dense/out.mp4" \
  "$SHARED_DIR/sglang_stage1_video_initial.pt" \
  "$SHARED_DIR/sglang_stage1_audio_initial.pt" \
  "$SHARED_DIR/sglang_stage2_video_noise.pt" \
  "$SHARED_DIR/sglang_stage2_audio_noise.pt"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing required artifact: $required" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES=0
export SGLANG_HQ_VARIANT=kwl
export ROOT="$ROOT"
export OUT_DIR="$OUT_DIR"
export FORCE=1
export WARMUP=true
export WARMUP_STEPS=15
export MASTER_PORT=30150
export SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN=0
export SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX=0
export SGLANG_HQ_KWL_FUSED_QK_ROPE=0
export SGLANG_HQ_KWL_FUSED_RMS_ADALN=0
export SGLANG_HQ_KWL_FUSED_ADALN=0
export SGLANG_HQ_KWL_FUSED_QKNORM_ROPE=0
export SGLANG_HQ_KWL_FUSED_DUAL_MODULATE=0
export SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL=0
export SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE=0
export SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU=0
export SGLANG_HQ_KWL_FUSED_AUDIO_QKVG=0
export SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE=0
export SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT=0
export SGLANG_HQ_KWL_COMPILE_TILED_VAE=0
export SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH="$SHARED_DIR/sglang_stage1_video_initial.pt"
export SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH="$SHARED_DIR/sglang_stage1_audio_initial.pt"
export SGLANG_LTX2_STAGE2_VIDEO_NOISE_PATH="$SHARED_DIR/sglang_stage2_video_noise.pt"
export SGLANG_LTX2_STAGE2_AUDIO_NOISE_PATH="$SHARED_DIR/sglang_stage2_audio_noise.pt"
export SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR="$OUT_DIR/latents"
export SGLANG_LTX2_DUMP_STAGE2_RENOISE_DIR="$OUT_DIR/latents"

echo "[run] KWL all switches off/no compile control -> $OUT_DIR"
bash scripts/run_ltx23_sglang_hq_1080p10s.sh > "$ROOT/kwl_all_off_no_compile.log" 2>&1

OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .conda/ltx23/bin/python - "$ROOT" <<'PYSUM'
import json, math, sys
from pathlib import Path
import cv2
root=Path(sys.argv[1])
left=root/'dense/out.mp4'; right=root/'kwl_all_off_no_compile/out.mp4'
cap_l=cv2.VideoCapture(str(left)); cap_r=cv2.VideoCapture(str(right))
frames=sum_abs=sum_sq=sum_pix=0
while True:
    ok_l, fl=cap_l.read(); ok_r, fr=cap_r.read()
    if not ok_l or not ok_r: break
    if fl.shape != fr.shape:
        fr=cv2.resize(fr,(fl.shape[1],fl.shape[0]))
    d=fl.astype('float32')-fr.astype('float32')
    sum_abs += float(abs(d).sum()); sum_sq += float((d*d).sum()); sum_pix += int(d.size); frames += 1
cap_l.release(); cap_r.release()
mse=sum_sq/sum_pix if sum_pix else 0.0
mad=sum_abs/sum_pix if sum_pix else 0.0
psnr=10*math.log10((255*255)/mse) if mse>0 else float('inf')
result={'frames':frames,'mean_abs_diff':mad,'mse':mse,'psnr_db':psnr}
(root/'kwl_all_off_no_compile_diff.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(result, indent=2, sort_keys=True))
PYSUM

echo "[done] $ROOT/kwl_all_off_no_compile_diff.json"
