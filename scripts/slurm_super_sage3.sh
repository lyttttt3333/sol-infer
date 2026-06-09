#!/bin/bash
#SBATCH --job-name=super-sage3
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/yitongl/cosmos3-run/super-sage3-%j.out
#SBATCH --error=/home/yitongl/cosmos3-run/super-sage3-%j.out
set -uo pipefail
REPO=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
PYTHON=$REPO/.conda/ltx23/bin/python; RUN_BASE=/home/yitongl/cosmos3-run; CACHE=$RUN_BASE/.cache
PYLIBS=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/pylibs
cd "$REPO"; echo "[$(date)] $(hostname)"
export HF_HOME=/home/yitongl/.hf_cache/huggingface HF_HUB_CACHE=/home/yitongl/.hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/python:$PYLIBS:/home/yitongl/.pylibs:${PYTHONPATH:-}"
export CUDA_HOME=$REPO/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export XDG_CACHE_HOME=$CACHE/xdg TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion TMPDIR=$RUN_BASE/.tmp
echo "sageattn3 import:"; $PYTHON -c "from sageattn3 import sageattn3_blackwell; print('  OK')" 2>&1 | grep -iE "OK|Error|No module" | head -1
L="$HF_HUB_CACHE/models--nvidia--Cosmos3-Super/snapshots/$(cat $HF_HUB_CACHE/models--nvidia--Cosmos3-Super/refs/main)"
PR=/home/yitongl/cosmos3-run/official_prompts
run(){ local out=$RUN_BASE/super-sage3/$1; mkdir -p "$out"
  echo "[$(date)] === SUPER $1 (attention-backend=$2, sp=4) ==="
  stdbuf -oL -eL "$PYTHON" -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --model-path "$L" --num-gpus 4 --attention-backend $2 \
    --prompt "$(cat $PR/t2v_prompt.txt)" --negative-prompt "$(cat $PR/negative_prompt.txt)" \
    --height 720 --width 1280 --num-frames 189 --fps 24 \
    --num-inference-steps 35 --guidance-scale 6.0 --flow-shift 10.0 --max-sequence-length 4096 \
    --use-guardrails false --seed 42 --warmup true --warmup-steps 1 \
    --dit-cpu-offload false --enable-sequence-shard true \
    --scheduler-port $3 --master-port $4 \
    --output-file-path "$out/out.mp4" --perf-dump-path "$out/perf.json" 2>&1 | grep -E "finished in [0-9]|Error executing|Generation failed|not supported|sage|head_dim|assert" | tail -4
  $PYTHON -c "import json;d=json.load(open('$out/perf.json'));s={x['name']:x['duration_ms'] for x in d['steps']};print('  >>> SUPER $1: denoise %.1fs (%.0f ms/step)'%(s['Cosmos3DenoisingStage']/1000,s['Cosmos3DenoisingStage']/35))" 2>/dev/null || echo "  >>> SUPER $1: FAILED (no perf.json)"
}
run fa_baseline fa          5690 31090
run sage3_fp4   sage_attn_3 5692 31092
echo "[$(date)] super-sage3 done"
