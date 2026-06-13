#!/bin/bash
#SBATCH --job-name=sana-video-sglang
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/home/yitongl/sana_video/logs/sglang-%j.out
#SBATCH --error=/home/yitongl/sana_video/logs/sglang-%j.out

# Validate + run the SANA-Video port in the sglang multimodal_gen runtime.
# Single GPU (the model is single-GPU); 4-GPU exclusive node per cluster convention.

set -uo pipefail
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer

export HF_HOME=/home/yitongl/.hf_cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN=$(cat /home/yitongl/.cache/huggingface/token)
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/home/yitongl/.cache/xdg
export TMPDIR=/home/yitongl/sana_video/.tmp
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/home/yitongl/.cache/torchinductor
export TRITON_CACHE_DIR=/home/yitongl/.cache/triton

# CUDA toolkit (pip nvidia cu13) so sglang JIT kernels (tvm_ffi/timestep_embedding,
# deep_gemm, nvfp4) can compile. Same convention as the LTX-2 nvfp4 bench scripts.
export CUDA_HOME="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# Persist JIT-compiled kernels on /home so only the first run pays the compile cost.
export TORCH_EXTENSIONS_DIR=/home/yitongl/.cache/torch_extensions
export SGLANG_DIFFUSION_CACHE_ROOT=/home/yitongl/.cache/sgl_diffusion
export CUDA_CACHE_PATH=/home/yitongl/.cache/cuda_cache

mkdir -p /home/yitongl/sana_video/logs /home/yitongl/sana_video/outputs "$TMPDIR" "$XDG_CACHE_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$SGLANG_DIFFUSION_CACHE_ROOT" "$CUDA_CACHE_PATH"

PY=.conda/ltx23/bin/python
echo "[$(date)] node=$(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1

$PY scripts/sana_video_sglang_run.py "$@"
rc=$?
echo "[$(date)] EXIT_RC=$rc"
exit $rc
