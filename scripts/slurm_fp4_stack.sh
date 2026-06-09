#!/bin/bash
#SBATCH --job-name=fp4-stack
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/fp4-stack-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/fp4-stack-%j.out
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
tc(){ export SGLANG_COSMOS3_TEACACHE_ENABLED=1 SGLANG_COSMOS3_TEACACHE_THRESH=1.15 SGLANG_COSMOS3_TEACACHE_START=16 SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=2; }
gen(){ # tag fp4 compile teacache sched master
  local out=$RUN_BASE/fp4-stack/$1; mkdir -p "$out"
  export SGLANG_COSMOS3_FP4_LINEAR=$2
  if [ "$4" = "1" ]; then tc; else clrtc; fi
  echo "[$(date)] === $1 (fp4=$2 compile=$3 teacache=$4) ==="
  stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path "$L" --num-gpus 4 \
    --prompt "$(cat $PR/t2v_prompt.txt)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
    --height 720 --width 1280 --num-frames 189 --fps 24 \
    --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
    --use-guardrails false --seed 42 --warmup true --warmup-steps 1 \
    --dit-cpu-offload false --enable-sequence-shard true --enable-torch-compile $3 \
    --scheduler-port $5 --master-port $6 \
    --output-file-path "$out/out.mp4" --perf-dump-path "$out/perf.json" 2>&1 | grep -E "fp4-linear\]|skip=\[|finished in [0-9]|Error executing|Generation failed|SymNode" | tail -4
  $PYTHON -c "import json;d=json.load(open('$out/perf.json'));s={x['name']:x['duration_ms'] for x in d['steps']};print('  >>> $1: denoise %.1fs (%.0f ms/step)'%(s['Cosmos3DenoisingStage']/1000,s['Cosmos3DenoisingStage']/35))" 2>/dev/null || echo "  >>> $1: FAILED"
}
gen fp4_teacache       1 false 1 5710 31110
gen fp4_compile        1 true  0 5712 31112
gen fp4_compile_teacache 1 true 1 5714 31114
echo "[$(date)] fp4-stack done"
