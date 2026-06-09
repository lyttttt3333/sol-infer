#!/bin/bash
#SBATCH --job-name=fp4-demo
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/fp4-demo-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/fp4-demo-%j.out
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
L="$HF_HUB_CACHE/models--nvidia--Cosmos3-Super/snapshots/$(cat $HF_HUB_CACHE/models--nvidia--Cosmos3-Super/refs/main)"
PR=/home/yitongl/cosmos3-run/official_prompts
clrtc(){ unset SGLANG_COSMOS3_TEACACHE_ENABLED SGLANG_COSMOS3_TEACACHE_THRESH SGLANG_COSMOS3_TEACACHE_START SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS; }
gen(){ # tag promptfile mode(base/accel) sched master
  local out=$RUN_BASE/fp4-demo/$1; mkdir -p "$out"
  if [ "$3" = "accel" ]; then
    export SGLANG_COSMOS3_FP4_LINEAR=1
    export SGLANG_COSMOS3_TEACACHE_ENABLED=1 SGLANG_COSMOS3_TEACACHE_THRESH=1.15 SGLANG_COSMOS3_TEACACHE_START=10 SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=3
    local compile=true
  else
    export SGLANG_COSMOS3_FP4_LINEAR=0; clrtc; local compile=false
  fi
  echo "[$(date)] === $1 (mode=$3) ==="
  stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path "$L" --num-gpus 4 \
    --prompt "$(cat $2)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
    --height 720 --width 1280 --num-frames 189 --fps 24 \
    --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
    --use-guardrails false --seed 42 --warmup true --warmup-steps 1 \
    --dit-cpu-offload false --enable-sequence-shard true --enable-torch-compile $compile \
    --scheduler-port $4 --master-port $5 \
    --output-file-path "$out/out.mp4" --perf-dump-path "$out/perf.json" 2>&1 | grep -E "fp4-linear\]|skip=\[|finished in [0-9]|Error executing|Generation failed" | tail -3
  $PYTHON -c "import json;d=json.load(open('$out/perf.json'));s={x['name']:x['duration_ms'] for x in d['steps']};print('  >>> $1: denoise %.1fs'%(s['Cosmos3DenoisingStage']/1000))" 2>/dev/null || echo "  >>> $1: FAILED"
}
gen p0_robotarm_baseline $PR/t2v_prompt.txt    base  5730 31130
gen p0_robotarm_accel    $PR/t2v_prompt.txt    accel 5731 31131
gen p1_botanist_baseline $PR/t2v_prompt_p1.txt base  5732 31132
gen p1_botanist_accel    $PR/t2v_prompt_p1.txt accel 5733 31133
gen p2_fox_baseline      $PR/t2v_prompt_p2.txt base  5734 31134
gen p2_fox_accel         $PR/t2v_prompt_p2.txt accel 5735 31135
echo "[$(date)] fp4-demo done"
