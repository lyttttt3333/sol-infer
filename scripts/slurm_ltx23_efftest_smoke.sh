#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 00:50:00
#SBATCH -J ltx23-efftest
#SBATCH -o outputs/slurm/ltx23-efftest-%j.out
#SBATCH -e outputs/slurm/ltx23-efftest-%j.err
# Smoke-test the efficiency-framework stage-2 midpoint prune wiring:
#   gpu0 = full-opt WITH framework-scored prune (RATIO=0.5 feat_norm steps 1,2)
#   gpu1 = full-opt WITHOUT prune (prune env unset -> guarded off == baseline)
# Validates: prune-on runs end-to-end + logs the "stage2 midpoint prune" line
# (proves the framework keep_indices path executed) + produces a video.
set -euo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm
export MODEL_PATH="outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493"
ROOT="${ROOT:-outputs/ltx23-efftest}"; export ROOT FORCE=1 WARMUP=false
PROMPT_TEXT="A cinematic aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement"

apply_kwl(){ export SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN=1 SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX=1 SGLANG_HQ_KWL_FUSED_QK_ROPE=1 SGLANG_HQ_KWL_FUSED_RMS_ADALN=1 SGLANG_HQ_KWL_FUSED_ADALN=1 SGLANG_HQ_KWL_FUSED_QKNORM_ROPE=1 SGLANG_HQ_KWL_FUSED_DUAL_MODULATE=1 SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE=1 SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL=1 SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE=1 SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU=1 SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT=1 SGLANG_HQ_KWL_FUSED_AUDIO_QKVG=1 SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE=1 SGLANG_HQ_KWL_COMPILE_TILED_VAE=1; }
PISA="piecewise_sparsity=0.9,piecewise_block_size=64,piecewise_only_video_self_attention=true,piecewise_stage1_schedule=false,piecewise_stage1_dense_steps=0,piecewise_stage1_start_sparsity=0.9,piecewise_stage1_end_sparsity=0.9,piecewise_dense_layers=none,piecewise_stage1_dense_layers=none,piecewise_stage2_dense_layers=0-1,piecewise_approx_remainder=true,piecewise_route_mode=score,piecewise_dense_fallback=fa"

run_cell(){ local gpu="$1" cfg="$2"; local od="$ROOT/$cfg"; mkdir -p "$od"
  ( set -e
    export CUDA_VISIBLE_DEVICES="$gpu" MASTER_PORT=$((31500+gpu)) OUT_DIR="$od" PROMPT="$PROMPT_TEXT"
    export SGLANG_HQ_TMPDIR="$PWD/outputs/.tmp/eff_gpu$gpu"; mkdir -p "$SGLANG_HQ_TMPDIR"
    export SGLANG_LTX2_ENABLE_TEMPORAL_UPSAMPLE=0
    apply_kwl
    export SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1 SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET=8of15_last_29calls SGLANG_LTX2_PREPROJECT_PROMPTS=1 SGLANG_LTX2_CACHE_ROPE_EMB=1
    export SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS="transformer=fa,transformer_2=piecewise_attn" SGLANG_HQ_ATTENTION_BACKEND_CONFIG="$PISA"
    if [ "$cfg" = prune_on ]; then
      export SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_RATIO=0.5 SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_METHOD=feat_norm SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_STEPS=1,2
    fi
    bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl_stage1_cache_core
  ) > "$ROOT/$cfg.log" 2>&1 || echo "[error] $cfg"; }

run_cell 0 prune_on & run_cell 1 prune_off &
wait
echo "[job done] eff smoke -> $ROOT"
echo "=== prune log line present? ==="
grep -h "stage2 midpoint prune" "$ROOT/prune_on.log" 2>/dev/null | head -5 || echo "(no prune log line!)"
echo "=== outputs ==="
ls -la "$ROOT"/prune_on/out.mp4 "$ROOT"/prune_off/out.mp4 2>/dev/null
