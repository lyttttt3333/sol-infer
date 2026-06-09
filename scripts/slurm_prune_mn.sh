#!/bin/bash
#SBATCH --job-name=prune-mn
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/prune-mn-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/prune-mn-%j.out
set -uo pipefail
REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python; RUN_BASE=/home/yitongl/cosmos3-run; CACHE=$RUN_BASE/.cache
cd "$REPO"; echo "[$(date)] $(hostname)"
export HF_HOME=/home/yitongl/.hf_cache/huggingface HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/python:${PYTHONPATH:-}"
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export XDG_CACHE_HOME=$CACHE/xdg TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion TMPDIR=$RUN_BASE/.tmp
# Isolate pruning: FP4 / TeaCache / compile OFF. Fixed r=0.5, feat_norm.
export SGLANG_COSMOS3_FP4_LINEAR=0
unset SGLANG_COSMOS3_TEACACHE_ENABLED SGLANG_COSMOS3_TEACACHE_THRESH SGLANG_COSMOS3_TEACACHE_START SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS
export SGLANG_COSMOS3_PRUNE_RATIO=0.5 SGLANG_COSMOS3_PRUNE_METHOD=feat_norm SGLANG_COSMOS3_PRUNE_COMPENSATION=prev
L="$HF_HUB_CACHE/models--nvidia--Cosmos3-Super/snapshots/$(cat $HF_HUB_CACHE/models--nvidia--Cosmos3-Super/refs/main)"
PR=/home/yitongl/cosmos3-run/official_prompts
gen(){ # tag prune_steps(m-(34-n)) sched master   [first m full, last n full]
  local out=$RUN_BASE/prune-mn/$1; mkdir -p "$out"
  export SGLANG_COSMOS3_PRUNE_STEPS="$2"
  echo "[$(date)] === $1 (PRUNE_STEPS=$2) ==="
  stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path "$L" --num-gpus 4 \
    --prompt "$(cat $PR/t2v_prompt.txt)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
    --height 720 --width 1280 --num-frames 189 --fps 24 \
    --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
    --use-guardrails false --seed 42 --warmup true --warmup-steps 1 \
    --dit-cpu-offload false --enable-sequence-shard true \
    --scheduler-port $3 --master-port $4 \
    --output-file-path "$out/out.mp4" --perf-dump-path "$out/perf.json" 2>&1 | grep -E "token-prune\]|finished in [0-9]|Error executing" | tail -4
  $PYTHON -c "import json;d=json.load(open('$out/perf.json'));s={x['name']:x['duration_ms'] for x in d['steps']};print('  >>> $1: denoise %.1fs'%(s['Cosmos3DenoisingStage']/1000))" 2>/dev/null || echo "  >>> $1: FAILED"
}
gen m3n3  "3-31"  5780 31180
gen m5n5  "5-29"  5782 31182
gen m0n5  "0-29"  5784 31184
gen m5n0  "5-34"  5786 31186
gen m3n8  "3-26"  5788 31188
echo "[$(date)] prune-mn done"
