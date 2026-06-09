#!/bin/bash
#SBATCH --job-name=super-profile
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/home/yitongl/cosmos3-run/super-profile-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/super-profile-%j.out
set -uo pipefail
REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python
RUN_BASE=/home/yitongl/cosmos3-run; CACHE=$RUN_BASE/.cache
OUT=$RUN_BASE/super-profile; mkdir -p "$OUT"
cd "$REPO"
echo "[$(date)] Node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export HF_HOME=/home/yitongl/.hf_cache/huggingface HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/python:${PYTHONPATH:-}"
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export XDG_CACHE_HOME=$CACHE/xdg TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion TMPDIR=$RUN_BASE/.tmp
# --- module profiler ---
export SGLANG_COSMOS3_PROFILE_MODULES=1
export SGLANG_COSMOS3_PROFILE_WARMUP=5
export SGLANG_COSMOS3_PROFILE_DUMP=$OUT/module_profile.json
L="$HF_HUB_CACHE/models--nvidia--Cosmos3-Super/snapshots/$(cat $HF_HUB_CACHE/models--nvidia--Cosmos3-Super/refs/main)"
PR=/home/yitongl/cosmos3-run/official_prompts
echo "[$(date)] profiling Super-T2V baseline (35 steps, warmup 5 -> 30 steady steps)"
stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path "$L" --num-gpus 4 \
  --prompt "$(cat $PR/t2v_prompt.txt)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
  --use-guardrails false --seed 42 --warmup false \
  --dit-cpu-offload false --enable-sequence-shard true --scheduler-port 5602 --master-port 31002 \
  --output-file-path "$OUT/out.mp4" --perf-dump-path "$OUT/perf.json"
echo "[$(date)] done. module profile:"; cat "$OUT/module_profile.json" 2>/dev/null
