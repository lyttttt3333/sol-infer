#!/bin/bash
#SBATCH --job-name=super-compile-ab
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/super-compile-ab-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/super-compile-ab-%j.out
set -uo pipefail
REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python
RUN_BASE=/home/yitongl/cosmos3-run; CACHE=$RUN_BASE/.cache
cd "$REPO"; echo "[$(date)] Node $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader|head -1||true
export HF_HOME=/home/yitongl/.hf_cache/huggingface HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/python:${PYTHONPATH:-}"
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export XDG_CACHE_HOME=$CACHE/xdg TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion TMPDIR=$RUN_BASE/.tmp
L="$HF_HUB_CACHE/models--nvidia--Cosmos3-Super/snapshots/$(cat $HF_HUB_CACHE/models--nvidia--Cosmos3-Super/refs/main)"
PR=/home/yitongl/cosmos3-run/official_prompts
run(){  # $1=tag $2=compile
  local out=$RUN_BASE/compile-ab/$1; mkdir -p "$out"
  echo "[$(date)] === RUN $1 (enable_torch_compile=$2) ==="
  stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path "$L" --num-gpus 4 \
    --prompt "$(cat $PR/t2v_prompt.txt)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
    --height 720 --width 1280 --num-frames 189 --fps 24 \
    --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
    --use-guardrails false --seed 42 --warmup true --warmup-steps 1 \
    --dit-cpu-offload false --enable-sequence-shard true --enable-torch-compile $2 \
    --scheduler-port $3 --master-port $4 \
    --output-file-path "$out/out.mp4" --perf-dump-path "$out/perf.json" 2>&1 | grep -vE "Denoising:.*it/s\]$" | tail -3
  $PYTHON -c "import json;d=json.load(open('$out/perf.json'));s={x['name']:x['duration_ms'] for x in d['steps']};print('  >>> $1: denoise %.1fs (%.0f ms/step), total %.1fs'%(s['Cosmos3DenoisingStage']/1000,s['Cosmos3DenoisingStage']/35,d['total_duration_ms']/1000))"
}
run compile_off false 5610 31010
run compile_on  true  5620 31020
echo "[$(date)] A/B done"
