#!/usr/bin/env bash
# Post-install fixups so the runtime CUDA JIT (tvm-ffi / sgl_kernel) can compile
# and link on a fresh env. `pip install -e python[diffusion]` only pulls the
# *runtime* CUDA libs (torch's deps); the JIT additionally needs the compiler
# toolchain + CCCL headers + dev symlinks, none of which are hard deps of any
# wheel. Run this once after the editable install, inside the activated env.
#
# Usage:  bash scripts/postinstall_cuda_jit.sh [--with-te]
#   --with-te : also install NVIDIA transformer_engine (needed for the NVFP4
#               'fullopt' path of Cosmos3-Super and LTX-2.3; Blackwell-only at run).
#
# Idempotent. Pins match the known-good aarch64 / CUDA-13 resolution; override
# any version via the env vars below if your stack differs.
set -euo pipefail

PY="${PYTHON_BIN:-python}"
NVCC_VER="${NVCC_VER:-13.2.78}"          # nvcc/crt/nvvm — JIT "nvcc: not found"
CCCL_VER="${CCCL_VER:-}"                  # cuda-cccl (nv/target headers); empty = latest cu13
CUBLAS_VER="${CUBLAS_VER:-13.2.2.2}"      # TE needs cublasLtGroupedMatrixLayoutInit_internal
WITH_TE=0
[[ "${1:-}" == "--with-te" ]] && WITH_TE=1

echo "[postinstall] python: $("$PY" -c 'import sys;print(sys.executable)')"

# 1. CUDA compiler toolchain (JIT compile): nvcc + crt + nvvm
echo "[postinstall] installing CUDA compiler toolchain (nvcc/crt/nvvm $NVCC_VER)"
"$PY" -m pip install --no-deps \
  "nvidia-cuda-nvcc==${NVCC_VER}" "nvidia-cuda-crt==${NVCC_VER}" "nvidia-nvvm==${NVCC_VER}"

# 2. CCCL headers (JIT needs <nv/target>, <cuda/...>, cub, thrust)
echo "[postinstall] installing CCCL headers (nvidia-cuda-cccl)"
if [[ -n "$CCCL_VER" ]]; then
  "$PY" -m pip install --no-deps "nvidia-cuda-cccl==${CCCL_VER}"
else
  "$PY" -m pip install --no-deps nvidia-cuda-cccl
fi

# 3. cublas with the symbol TransformerEngine links against (NVFP4 path)
echo "[postinstall] aligning nvidia-cublas to $CUBLAS_VER (TE / NVFP4)"
"$PY" -m pip install --no-deps "nvidia-cublas==${CUBLAS_VER}" || \
  echo "[postinstall] WARN: could not pin cublas==$CUBLAS_VER (only needed for NVFP4 fullopt)"

# 4. JIT linker expects lib64/ + unversioned dev symlinks (pip wheels ship only
#    versioned libs in lib/), else: `ld: cannot find -lcudart`.
NV_DIR="$("$PY" -c 'import importlib.util,os; s=importlib.util.find_spec("nvidia"); print(os.path.dirname(s.submodule_search_locations[0]) if s else "")')"
NV_DIR="${NV_DIR}/nvidia"
if [[ -d "$NV_DIR" ]]; then
  echo "[postinstall] creating lib64/ + unversioned .so dev symlinks under $NV_DIR"
  for libdir in "$NV_DIR"/*/lib; do
    [[ -d "$libdir" ]] || continue
    [[ -e "$(dirname "$libdir")/lib64" ]] || ln -s lib "$(dirname "$libdir")/lib64"
    for so in "$libdir"/lib*.so.*; do
      [[ -e "$so" ]] || continue
      base="$(basename "$so")"; unver="${base%%.so.*}.so"
      [[ -e "$libdir/$unver" ]] || ln -s "$base" "$libdir/$unver"
    done
  done
else
  echo "[postinstall] WARN: nvidia/ site-packages dir not found; skipped symlink step"
fi

# 5. transformer_engine (optional; NVFP4 fullopt). No aarch64 prebuilt wheel as of
#    cu13 — may build from source (needs nvcc + cmake) or be copied from a known-good env.
if [[ "$WITH_TE" == "1" ]]; then
  echo "[postinstall] installing transformer_engine (source build may take a while)"
  "$PY" -m pip install "transformer_engine[pytorch]" || \
    echo "[postinstall] WARN: TE install failed — NVFP4 fullopt will gracefully fall back to BF16"
fi

echo "[postinstall] done. Verify JIT: run any model's baseline once on a GPU node."
