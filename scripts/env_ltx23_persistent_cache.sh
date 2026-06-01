#!/usr/bin/env bash
# Shared persistent cache locations for LTX2/SGLang experiments.
# Source this from benchmark and generation scripts before invoking Python.

if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    _ltx23_cache_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$_ltx23_cache_script_dir/.." && pwd)"
  else
    REPO_ROOT="$(pwd)"
  fi
fi

export LTX23_CACHE_ROOT="${LTX23_CACHE_ROOT:-$REPO_ROOT/outputs/.cache}"
mkdir -p \
  "$LTX23_CACHE_ROOT/huggingface" \
  "$LTX23_CACHE_ROOT/huggingface/hub" \
  "$LTX23_CACHE_ROOT/xdg" \
  "$LTX23_CACHE_ROOT/torch" \
  "$LTX23_CACHE_ROOT/triton" \
  "$LTX23_CACHE_ROOT/torchinductor" \
  "$LTX23_CACHE_ROOT/torch_extensions" \
  "$LTX23_CACHE_ROOT/cuda" \
  "$LTX23_CACHE_ROOT/c_headers" \
  "$LTX23_CACHE_ROOT/sgl_diffusion" \
  "$LTX23_CACHE_ROOT/tmp"

export HF_HOME="${HF_HOME:-$LTX23_CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LTX23_CACHE_ROOT/xdg}"
export TORCH_HOME="${TORCH_HOME:-$LTX23_CACHE_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LTX23_CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$LTX23_CACHE_ROOT/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$LTX23_CACHE_ROOT/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$LTX23_CACHE_ROOT/cuda}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export SGLANG_DIFFUSION_CACHE_ROOT="${SGLANG_DIFFUSION_CACHE_ROOT:-$LTX23_CACHE_ROOT/sgl_diffusion}"
export TMPDIR="${TMPDIR:-$LTX23_CACHE_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"

# Some cluster images expose a conda Python built with HAVE_CRYPT_H while the
# compute node lacks the libxcrypt development header. Triton launcher JIT only
# needs Python.h, not crypt(), so a local empty header keeps those JIT kernels
# buildable without requiring system package changes.
if ! printf '#include <crypt.h>\n' | "${CC:-cc}" -E - >/dev/null 2>&1; then
  printf '#pragma once\n' > "$LTX23_CACHE_ROOT/c_headers/crypt.h"
  export CPATH="$LTX23_CACHE_ROOT/c_headers:${CPATH:-}"
fi
