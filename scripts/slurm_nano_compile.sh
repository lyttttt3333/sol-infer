#!/bin/bash
#SBATCH --job-name=nano-compile
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/nano-compile-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/nano-compile-%j.out
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
L="$HF_HUB_CACHE/models--nvidia--Cosmos3-Nano/snapshots/$(cat $HF_HUB_CACHE/models--nvidia--Cosmos3-Nano/refs/main)"
PR=/home/yitongl/cosmos3-run/official_prompts
run(){ # tag compile mode port
  local out=$RUN_BASE/nano-compile/$1; mkdir -p "$out"
  export SGLANG_COSMOS3_COMPILE_MODE=$3
  echo "[$(date)] === nano $1 (compile=$2 mode=$3, sp=1, num-gpus=1) ==="
  stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path "$L" --num-gpus 1 \
    --prompt "$(cat $PR/t2v_prompt.txt)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
    --height 720 --width 1280 --num-frames 189 --fps 24 \
    --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
    --use-guardrails false --seed 42 --warmup true --warmup-steps 1 \
    --dit-cpu-offload false --enable-torch-compile $2 \
    --scheduler-port $4 --master-port $5 \
    --output-file-path "$out/out.mp4" --perf-dump-path "$out/perf.json" 2>&1 | grep -E "finished in [0-9]|Denoising complete|SymNode|graph break|Error executing|Generation failed" | tail -3
  $PYTHON -c "import json;d=json.load(open('$out/perf.json'));s={x['name']:x['duration_ms'] for x in d['steps']};print('  >>> nano $1: denoise %.1fs (%.0f ms/step)'%(s['Cosmos3DenoisingStage']/1000,s['Cosmos3DenoisingStage']/35))" 2>/dev/null || echo "  >>> nano $1: FAILED (no perf.json)"
}
run off          false default          5650 31050
run compile_def  true  default          5651 31051
run compile_cuda true  reduce-overhead  5652 31052
echo "[$(date)] nano compile bench done"
