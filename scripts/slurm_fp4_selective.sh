#!/bin/bash
#SBATCH --job-name=fp4-sel
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --output=/home/yitongl/cosmos3-run/fp4-sel-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/fp4-sel-%j.out
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
export SGLANG_COSMOS3_FP4_LINEAR=1
L="$HF_HUB_CACHE/models--nvidia--Cosmos3-Super/snapshots/$(cat $HF_HUB_CACHE/models--nvidia--Cosmos3-Super/refs/main)"
PR=/home/yitongl/cosmos3-run/official_prompts
gen(){ # tag  FL FLlast Sfirst Slast  sched master
  local out=$RUN_BASE/fp4-sel/$1; mkdir -p "$out"
  export SGLANG_COSMOS3_FP4_SKIP_FIRST_LAYERS=$2 SGLANG_COSMOS3_FP4_SKIP_LAST_LAYERS=$3
  export SGLANG_COSMOS3_FP4_SKIP_FIRST_STEPS=$4 SGLANG_COSMOS3_FP4_SKIP_LAST_STEPS=$5
  echo "[$(date)] === $1 (skipL_first=$2 last=$3 skipStep_first=$4 last=$5) ==="
  stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path "$L" --num-gpus 4 \
    --prompt "$(cat $PR/t2v_prompt.txt)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
    --height 720 --width 1280 --num-frames 189 --fps 24 \
    --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
    --use-guardrails false --seed 42 --warmup true --warmup-steps 1 \
    --dit-cpu-offload false --enable-sequence-shard true \
    --scheduler-port $6 --master-port $7 \
    --output-file-path "$out/out.mp4" --perf-dump-path "$out/perf.json" 2>&1 | grep -E "fp4-linear\]|finished in [0-9]|Error executing" | tail -3
  $PYTHON -c "import json;d=json.load(open('$out/perf.json'));s={x['name']:x['duration_ms'] for x in d['steps']};print('  >>> $1: denoise %.1fs (%.0f ms/step)'%(s['Cosmos3DenoisingStage']/1000,s['Cosmos3DenoisingStage']/35))" 2>/dev/null || echo "  >>> $1: FAILED"
}
#    tag                      Lf Ll Sf Sl
gen full_fp4                  0  0  0  0  5720 31120
gen first4layers_bf16         4  0  0  0  5721 31121
gen first3steps_bf16          0  0  3  0  5722 31122
gen last3steps_bf16           0  0  0  3  5723 31123
gen first3_last3steps_bf16    0  0  3  3  5724 31124
gen first8L_first3_last3S     8  0  3  3  5725 31125
echo "[$(date)] fp4-sel done"
