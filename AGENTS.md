# Agent README — Deploy & Run (Sol-LTX-Infer)

Portable guide for deploying this video-diffusion inference stack (SANA-Video,
Cosmos3-Super, LTX-2.3) on a fresh machine. Three sections, in order:

1. [Set up the environment](#1-set-up-the-environment)
2. [Download the models](#2-download-the-models)
3. [Run inference](#3-run-inference)

## Prerequisites

- An NVIDIA GPU. The shipped stack is built for **CUDA 13** (torch 2.11+cu130);
  the driver must support it. Cosmos3-Super (64B) needs **4 GPUs**; SANA-Video and
  LTX-2.3 run on **1 GPU** (more is faster via sequence parallel).
- `conda` (or miniforge) and `git` on PATH.
- A HuggingFace account/token for gated model downloads (`huggingface-cli login`).
- GPU inference must run on a machine with a working GPU + driver. A CPU-only
  box can install deps and download models but cannot run the pipelines.

---

## 1. Set up the environment

Creates a self-contained conda env `.conda/ltx23` (Python 3.12, torch 2.11+cu130,
diffusers 0.38) and installs the package in editable mode.

```bash
git clone <repo-url> Sol-LTX-Infer && cd Sol-LTX-Infer

# create the env (Python 3.12); writes to ./.conda/ltx23
PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh
conda activate "$PWD/.conda/ltx23"

# install the diffusion stack (editable)
uv pip install -e "$PWD/python[diffusion]" --prerelease=allow

# CUDA JIT fixups: the editable install pulls only runtime CUDA libs; the runtime
# kernel JIT also needs the compiler toolchain (nvcc), CCCL headers, and dev
# symlinks. This one-shot makes them present. Add --with-te for the NVFP4 fullopt
# path (Cosmos / LTX); without TE, fullopt gracefully falls back to BF16.
PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh   # [--with-te]
```

Optional: `sgl-deep-gemm` (FP4 GEMM speedups) builds separately in a
CUDA-versioned container — see `scripts/build_sgl_deep_gemm.sh`. Not required to run.

Verify:

```bash
.conda/ltx23/bin/python -c "import torch, diffusers, sglang; \
print(torch.__version__, diffusers.__version__, torch.cuda.is_available())"
# expect: 2.11.0+cu130 0.38.0 True   (False on a GPU-less box)
```

---

## 2. Download the models

Each pipeline loads weights from the HuggingFace Hub. Set a cache dir and pull the
repos you need (download requires network; the actual runs can then go offline
with `HF_HUB_OFFLINE=1`).

```bash
export HF_HOME="$PWD/.hf_cache"          # or any persistent path
huggingface-cli login                    # once, for gated repos
```

| Model | HF repo |
|---|---|
| SANA-Video (2B, 480p) | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` |
| Cosmos3-Super (64B)   | `nvidia/Cosmos3-Super` |
| LTX-2.3               | `Lightricks/LTX-2.3` |

Download (only what you need), e.g.:

```bash
huggingface-cli download Efficient-Large-Model/SANA-Video_2B_480p_diffusers
huggingface-cli download nvidia/Cosmos3-Super
huggingface-cli download Lightricks/LTX-2.3
```

Convenience download scripts (resumable `snapshot_download`, e.g. for headless /
CPU-mover nodes) live per model: `scripts/sana/download_sana_video_cpu.sh`,
`scripts/cosmos/slurm_download_cosmos3.sh`,
`scripts/ltx/slurm_download_ltx23_official_{cpu,lora_cpu}.sh`.

---

## 3. Run inference

Each model has one launch entry, with a plain `baseline` (no acceleration) and a
`fullopt` (the full speedup stack).

### SANA-Video (1 GPU)

```bash
PY=.conda/ltx23/bin/python
# baseline (dense)
$PY scripts/sana/sana_video_sglang_run.py \
    --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
    --prompt "a corgi running on the beach" --output out/sana_baseline
# fullopt = EasyCache + fusion
$PY scripts/sana/sana_video_sglang_run.py \
    --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
    --prompt "a corgi running on the beach" --output out/sana_fullopt \
    --easycache 0.1 --linattn-bf16 --qkv-merge --compile
```

### LTX-2.3 1080p/10s (1 GPU)

```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh baseline   # dense reference
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt    # ~2.5x; self-contained
```

`fullopt` bakes in the whole recipe (KWL fusion + stage-1 step-skip + stage-2 PISA
sparse attention + NVFP4 video FFN + stage-2 token-prune) — no extra flags. Point
`MODEL_PATH` / `DISTILLED_LORA` / `SPATIAL_UPSAMPLER` at your downloaded LTX-2.3
files if they are not in the default cache location (see the top of the script).

### Cosmos3-Super 64B (4 GPUs)

```bash
MODEL_REPO=nvidia/Cosmos3-Super \
ROOT=out/cosmos3 PROMPT_FILE=prompt.txt PROMPT_TAG=demo \
bash scripts/cosmos/slurm_cosmos3_super.sh baseline   # or: fullopt
```

`fullopt` = TeaCache (1.15/start10/max3) + step-selective NVFP4 (first/last 3
steps dense). The entry has `#SBATCH` headers for a SLURM cluster; on a single box
run it as a normal `bash` script after adjusting the few site paths near the top.

### Notes

- Quote timing from the `... warmup excluded` line in the log (cold runs are
  compile-dominated).
- On a multi-GPU scheduler (e.g. SLURM), wrap the run in a job that requests the
  GPU count above; otherwise run directly on the GPU host.
- `scripts/killall_sglang.sh` cleans up stray server/worker processes.
