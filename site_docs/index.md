# Sol-LTX-Infer

**Accelerated video-diffusion inference — SANA-Video · Cosmos3-Super · LTX-2.3.**

[:material-home: Homepage](https://lyttttt3333.github.io/sol-infer/){ .md-button }
[:material-file-document: arXiv](https://arxiv.org/abs/XXXX.XXXXX){ .md-button }
[:material-github: GitHub](https://github.com/lyttttt3333/sol-infer){ .md-button }

Three production video-diffusion models, each reduced to **one clean acceleration
line** (plus a dense `baseline`), composed from **five reusable acceleration
methods**. All speedups are GB200, warmup-excluded, at each model's official spec.

## Contents

- **[Optimization pipelines](pipelines.md)** — how each model (SANA-Video,
  Cosmos3-Super, LTX-2.3) is assembled and what it achieves.
- **[Acceleration methods](acceleration.md)** — the five building blocks
  (cache, quant, kernel fusion, sparse attention, token pruning) the pipelines
  are made of.

## Models

| Model | Acceleration line | Speedup |
|---|---|---|
| **SANA-Video** (2B, 1 GPU) | EasyCache + fusion + compile | **2.1× / 2.56×** |
| **Cosmos3-Super** (64B, 4 GPU) | TeaCache + step-selective NVFP4 | **~2.26×** |
| **LTX-2.3** (1080p/10s, 1 GPU) | KWL fusion + cache + PISA + NVFP4 + token-prune | **~2.4×** |

## Run

```bash
PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh && conda activate ./.conda/ltx23
uv pip install -e "python[diffusion]" --prerelease=allow
PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt
```

See `QUICKSTART.md` (human) and `AGENTS.md` (agent/portable deploy) in the repo.
