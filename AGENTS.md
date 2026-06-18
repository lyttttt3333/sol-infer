# Agent README — Environment Setup (Sol-LTX-Infer)

How to set up and run the video-diffusion inference environment (SANA-Video,
Cosmos3-Super, LTX-2.3) on the SLURM cluster. Written for an agent/new dev who
has never touched this checkout. Read all of section 0 before running anything.

## 0. Hard rules (violating these breaks the cluster or wastes hours)

1. **NO heavy work on the login node.** Installs (`uv pip install`), model
   downloads, `tar`, large `cp`, ffmpeg encodes, decode/metrics — all of it must
   run inside a SLURM job. The shared login node will crash under that load.
   - CPU-only heavy work (downloads, ffmpeg, metrics) → `cpu_datamover` partition.
   - GPU runs → `batch` partition. NOTE: `batch` QOS requires a **minimum of 4
     GPUs** (`--gpus-per-node=4 --exclusive`), even for a 1-GPU job.
2. **Compute nodes have NO network.** Anything that hits HuggingFace / the
   internet (model download, `huggingface_hub` upload) must run on a *networked*
   node (login / vscode node), never inside a `batch` job. Download models first
   (section 4), then run offline with `HF_HUB_OFFLINE=1`.
3. **Two-checkout model — keep CODE and ENV separate.**
   - Working **CODE** checkout: `~/code/cosmos3/Sol-LTX-Infer` — edit code here,
     point `PYTHONPATH`/`cd` here.
   - Reference **ENV** checkout: `~/code/Sol-LTX-Infer` — holds the shared
     `.conda/ltx23` environment, the materialized model caches, and official
     weights/LoRA. Treated as env+data only. The two share env/data but must have
     **zero CODE intersection** — never import sglang from the reference checkout
     when working in the cosmos3 one.
4. **Caches live on `$CODE_ROOT/.cache`, never `/tmp`.** `use_code_storage_env.sh`
   redirects `UV_CACHE_DIR`/`PIP_CACHE_DIR`/`HF_HOME`/`TMPDIR`/etc. off `/tmp`
   (which is small and node-local). Always source it before pip/uv/model runs.

## 1. Prerequisites

- `conda` on PATH.
- SLURM account `nvr_elm_llm`, partitions `batch` (GPU) and `cpu_datamover` (CPU).
- Target hardware: GB200 / B200 (aarch64). The shipped env is CUDA 13.0.

## 2. Create the conda environment

The env is `.conda/ltx23` (Python 3.12, torch 2.11+cu130, diffusers 0.38).

```bash
cd ~/code/cosmos3/Sol-LTX-Infer
PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh   # -> .conda/ltx23
```

This:
- sources `scripts/use_code_storage_env.sh` (cache redirection),
- creates `.conda/ltx23` with `python=3.12` + `pip` + `uv`,
- installs conda activate hooks so the cache-redirect env is applied on every
  `conda activate .conda/ltx23`.

If a working env already exists in the reference checkout
(`~/code/Sol-LTX-Infer/.conda/ltx23`), you can reuse it directly as the ENV side
(see section 6) instead of creating a new one.

## 3. Install dependencies

Run inside a SLURM job (rule 0.1), not on the login node:

```bash
# interactive on a CPU node, or wrap in an sbatch
source scripts/use_code_storage_env.sh
conda activate "$PWD/.conda/ltx23"
uv pip install -e "$PWD/python[diffusion]" --prerelease=allow
```

`sgl-deep-gemm` (optional, FP4 GEMM speedups) is built separately inside a
CUDA-versioned container — see `scripts/build_sgl_deep_gemm.sh`.

## 4. Download models (networked node, cpu_datamover)

Compute nodes are offline, so materialize models first on a node with network.
Per-model download scripts (run via `cpu_datamover`):

```bash
sbatch scripts/cosmos/slurm_download_cosmos3.sh
sbatch scripts/ltx/slurm_download_ltx23_official_cpu.sh
sbatch scripts/ltx/slurm_download_ltx23_official_lora_cpu.sh
bash   scripts/sana/download_sana_video_cpu.sh        # or via slurm_datamover_py.sh
```

Generic CPU runner for any download/processing python on the datamover node:
`sbatch scripts/slurm_datamover_py.sh <script.py> [args...]`.

Materialized models land under
`<checkout>/outputs/.cache/sgl_diffusion/materialized_models/`.

## 5. Run inference (GPU, batch partition)

Each model has ONE primary launch entry. All run single-GPU unless noted.

| Model | Launch | Modes |
|---|---|---|
| SANA-Video | `scripts/sana/sana_video_sglang_run.py` | EasyCache+fusion |
| Cosmos3-Super (64B) | `scripts/cosmos/slurm_cosmos3_super.sh` | `baseline` \| `fullopt` (TeaCache+NVFP4, 4 GPU) |
| LTX-2.3 1080p/10s | `scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh` | `baseline` \| `fullopt` |

LTX example (the canonical, unambiguous entry):

```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh baseline   # dense reference
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt    # full 2.47x stack, self-contained
```

`fullopt` bakes in the whole recipe (KWL fusion + stage-1 SCSP step-skip +
stage-2 PISA sparse + NVFP4 video FFN + stage-2 token-prune) — no extra env
needed. Perf: look for `... warmup excluded` in the log.

## 6. CODE-from-working / ENV-from-reference run pattern

To honor rule 0.3 — run with CODE from the working checkout but the env, model
cache, and weights shared from the reference checkout — set these before calling
a run script (example for LTX; see `~/cosmos3-run/run_ltx23_fullopt_vs_dense.sh`):

```bash
WORK=~/code/cosmos3/Sol-LTX-Infer        # CODE
REF=~/code/Sol-LTX-Infer                 # ENV + data (shared)
CU=$REF/.conda/ltx23/lib/python3.12/site-packages/nvidia
export PYTHON_BIN=$REF/.conda/ltx23/bin/python
export CUDA_HOME=$CU/cu13
export MODEL_PATH=$REF/outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$WORK"                               # PYTHONPATH resolves to $WORK/python
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt
```

## 7. SLURM cheat-sheet

```bash
# GPU run (batch requires >=4 GPUs)
#SBATCH -A nvr_elm_llm -p batch -N 1 --gpus-per-node=4 --exclusive -t 01:10:00

# CPU heavy work (download / ffmpeg / metrics) — has network
#SBATCH -A nvr_elm_llm -p cpu_datamover -N 1 --cpus-per-task=8 --mem=32G
```

`scripts/killall_sglang.sh` cleans up stray server/worker processes.

## 8. Verify a working setup

```bash
.conda/ltx23/bin/python -c "import torch,diffusers,sglang; print(torch.__version__, diffusers.__version__)"
# expect: 2.11.0+cu130 0.38.0
```
(A CUDA-driver-too-old warning on the login node is expected — GPUs only exist on
compute nodes.) Then run the LTX `baseline` smoke (section 5) inside a `batch` job.
