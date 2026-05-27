#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CUDA_HOME="${CUDA_HOME:-${REPO_ROOT}/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13}"
CCCL_INCLUDE="${FLASHINFER_CCCL_INCLUDE:-${REPO_ROOT}/.conda/ltx23/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include}"
CCCL_MAP="${DEEPGEMM_CCCL_MAP:-/tmp/ltx2-cccl-include}"

mkdir -p "${CCCL_MAP}/cccl"
ln -sfn "${CCCL_INCLUDE}/cuda" "${CCCL_MAP}/cccl/cuda"
ln -sfn "${CCCL_INCLUDE}/nv" "${CCCL_MAP}/cccl/nv"

if [[ -n "${DEEPGEMM_NVCC_ARCH_OVERRIDE:-}" ]]; then
  rewritten_args=()
  for arg in "$@"; do
    if [[ "${arg}" == --gpu-architecture=* ]]; then
      rewritten_args+=("--gpu-architecture=${DEEPGEMM_NVCC_ARCH_OVERRIDE}")
    else
      rewritten_args+=("${arg}")
    fi
  done
  set -- "${rewritten_args[@]}"
fi

exec "${CUDA_HOME}/bin/nvcc" -I"${CCCL_MAP}" -I"${CCCL_INCLUDE}" "$@"
