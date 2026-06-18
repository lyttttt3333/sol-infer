#!/bin/bash
#SBATCH --job-name=vbench-sana-gen
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/yitongl/sana_video/logs/vbench-gen-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/vbench-gen-%j.out

# Multi-node data-parallel SANA-Video generation over the 944 VBench prompts.
# srun launches nodes*4 tasks; each task = 1 GPU worker (global rank=SLURM_PROCID),
# generating prompts[rank::world]. Config (dense|fullopt) via $1.
set -uo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/home/yitongl/.cache/xdg
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/home/yitongl/.cache/torchinductor
export TRITON_CACHE_DIR=/home/yitongl/.cache/triton
# CUDA toolkit (pip nvidia cu13) for sglang JIT kernels.
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/home/yitongl/.cache/torch_extensions
export SGLANG_DIFFUSION_CACHE_ROOT=/home/yitongl/.cache/sgl_diffusion
export CUDA_CACHE_PATH=/home/yitongl/.cache/cuda_cache
# NOTE: do NOT set CUDA_VISIBLE_DEVICES here -- each task picks its GPU from
# SLURM_LOCALID inside gen_vbench_sana.py.

export GEN_CONFIG=${1:-dense}
export GEN_OUTDIR=/home/yitongl/code/vbench_sana/named/sana_${GEN_CONFIG}
export VBENCH_LIMIT=${VBENCH_LIMIT:-0}

mkdir -p /home/yitongl/sana_video/logs "$GEN_OUTDIR" "$XDG_CACHE_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$SGLANG_DIFFUSION_CACHE_ROOT" "$CUDA_CACHE_PATH"
PY=.conda/ltx23/bin/python
echo "[$(date)] config=$GEN_CONFIG nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS -> $GEN_OUTDIR"

srun --kill-on-bad-exit=0 $PY scripts/sana/gen_vbench_sana.py
rc=$?
echo "[$(date)] EXIT_RC=$rc  generated=$(ls "$GEN_OUTDIR"/*.mp4 2>/dev/null | wc -l)/944"
exit $rc
