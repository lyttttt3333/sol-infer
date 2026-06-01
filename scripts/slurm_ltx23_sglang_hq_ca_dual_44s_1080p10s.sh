#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH -J ltx23-hq-ca-dual
#SBATCH -o outputs/slurm/ltx23-hq-ca-dual-%j.out
#SBATCH -e outputs/slurm/ltx23-hq-ca-dual-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

export CUDA_VISIBLE_DEVICES="${LTX23_CUDA_VISIBLE_DEVICES:-0}"
export ROOT="${ROOT:-outputs/ltx23-hq-current-best-ca-dual-1080p10s}"
export OUT_DIR="${OUT_DIR:-$ROOT/ca_dual_only}"
export FORCE="${FORCE:-1}"
export WARMUP="${WARMUP:-true}"
export WARMUP_STEPS="${WARMUP_STEPS:-15}"
export MASTER_PORT="${MASTER_PORT:-30401}"

export SGLANG_HQ_VARIANT=kwl_stage1_cache_core_stage2_sparse
export SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1
export SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET=8of15_last_29calls

export SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN=1
export SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX=1
export SGLANG_HQ_KWL_FUSED_QK_ROPE=1
export SGLANG_HQ_KWL_FUSED_RMS_ADALN=1
export SGLANG_HQ_KWL_FUSED_ADALN=1
export SGLANG_HQ_KWL_FUSED_QKNORM_ROPE=1
export SGLANG_HQ_KWL_FUSED_DUAL_MODULATE=1
export SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE=1
export SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL=1
export SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE=1
export SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU=1
export SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT=1
export SGLANG_HQ_KWL_FUSED_AUDIO_QKVG=1
export SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE=1
export SGLANG_HQ_KWL_COMPILE_TILED_VAE=1

export SGLANG_LTX2_PREPROJECT_PROMPTS=1
export SGLANG_LTX2_CACHE_ROPE_EMB=1

# Deliberately leave these disabled: same-node ablation showed no additional
# speedup, and TE proj_in+GELU falls back on B200 for this shape.
unset SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT_RESIDUAL
unset SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU
unset SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE

bash scripts/run_ltx23_sglang_hq_1080p10s.sh "$SGLANG_HQ_VARIANT"

python - <<'PY2'
import json
import os
from pathlib import Path
out = Path(os.environ['OUT_DIR'])
perf = json.loads((out / 'perf.json').read_text())
sem = json.loads((out / 'hq_semantics.json').read_text())
steps = {item['name']: item['duration_ms'] / 1000 for item in perf.get('steps', [])}
summary = {
    'artifact_dir': str(out),
    'video': str(out / 'out.mp4'),
    'perf_json': str(out / 'perf.json'),
    'semantics_json': str(out / 'hq_semantics.json'),
    'total_s': perf.get('total_duration_ms', 0) / 1000,
    'stage1_s': steps.get('LTX2AVDenoisingStage', 0),
    'stage2_s': steps.get('LTX2RefinementStage', 0),
    'decode_s': steps.get('LTX2AVDecodingStage', 0),
    'stage1_cache_core_preset': sem.get('stage1_cache_core_preset'),
    'attention_backend_config': sem.get('attention_backend_config'),
    'kwl_flags': sem.get('kwl_flags'),
    'te_nvfp4_recipe': sem.get('te_nvfp4_recipe'),
}
(out / 'summary_ca_dual_best.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY2
