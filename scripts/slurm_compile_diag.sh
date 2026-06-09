#!/bin/bash
#SBATCH --job-name=compile-diag
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/home/yitongl/cosmos3-run/compile-diag-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/compile-diag-%j.out
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
export TORCH_LOGS="recompiles,graph_breaks"   # surface compile issues
L="$HF_HUB_CACHE/models--nvidia--Cosmos3-Super/snapshots/$(cat $HF_HUB_CACHE/models--nvidia--Cosmos3-Super/refs/main)"
PR=/home/yitongl/cosmos3-run/official_prompts; OUT=$RUN_BASE/compile-diag; mkdir -p "$OUT"
echo "[$(date)] compile_on, FULL output, 8 steps only (fast)"
stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path "$L" --num-gpus 4 \
  --prompt "$(cat $PR/t2v_prompt.txt)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 8 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
  --use-guardrails false --seed 42 --warmup false \
  --dit-cpu-offload false --enable-sequence-shard true --enable-torch-compile true \
  --scheduler-port 5630 --master-port 31030 \
  --output-file-path "$OUT/out.mp4" --perf-dump-path "$OUT/perf.json"
echo "[$(date)] exit=$? ; perf.json: $([ -f $OUT/perf.json ] && echo YES || echo NO)"
