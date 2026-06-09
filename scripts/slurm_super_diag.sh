#!/bin/bash
#SBATCH --job-name=super-diag
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/home/yitongl/cosmos3-run/super-diag-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/super-diag-%j.out

set -uo pipefail
REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python
RUN_BASE=/home/yitongl/cosmos3-run
CACHE=$RUN_BASE/.cache
cd "$REPO"
echo "[$(date)] Node $(hostname)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

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
mkdir -p "$RUN_BASE/super-diag"
echo "[$(date)] launching baseline Super-T2V, num-gpus=4 (line-buffered, direct to slurm out)"
stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path "$L" --num-gpus 4 \
  --prompt "A red fox running across a snowy forest trail at sunrise, cinematic" \
  --negative-prompt "blurry, low quality" \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
  --use-guardrails false --seed 42 --warmup false \
  --dit-cpu-offload false --enable-sequence-shard true --scheduler-port 5601 --master-port 31001 \
  --output-file-path "$RUN_BASE/super-diag/out.mp4" --perf-dump-path "$RUN_BASE/super-diag/perf.json"
echo "[$(date)] python exit=$?"