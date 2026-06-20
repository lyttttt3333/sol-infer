<div align="center">

# Sol-LTX-Infer

### Accelerated video-diffusion inference — SANA-Video · Cosmos3-Super · LTX-2.3

<!-- TODO: fill in the real Homepage and arXiv URLs -->
[🏠 Homepage](https://lyttttt3333.github.io/sol-infer/) &nbsp;·&nbsp;
[📄 arXiv](https://arxiv.org/abs/XXXX.XXXXX) &nbsp;·&nbsp;
[📖 Docs — Pipelines](https://lyttttt3333.github.io/sol-infer/) &nbsp;·&nbsp;
[📖 Docs — Methods](https://lyttttt3333.github.io/sol-infer/) &nbsp;·&nbsp;
[↗ Sana Docs](https://nvlabs.github.io/Sana/docs/)

</div>

---

Three production video-diffusion models, each reduced to **one clean acceleration
line** (plus a dense `baseline`), composed from five reusable acceleration methods.
All speedups are GB200, warmup-excluded, at each model's official spec.

## Models

| Model | Entry | Acceleration line | Speedup |
|---|---|---|---|
| **SANA-Video** (2B, 1 GPU) | `scripts/sana/sana_video_sglang_run.py` | EasyCache + fusion + compile | **2.1× / 2.56×** |
| **Cosmos3-Super** (64B, 4 GPU) | `scripts/cosmos/slurm_cosmos3_super.sh` | TeaCache + step-selective NVFP4 | **~2.26×** |
| **LTX-2.3** (1080p/10s, 1 GPU) | `scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh` | KWL fusion + cache + PISA + NVFP4 + token-prune | **~2.4×** |

Each entry takes `baseline | fullopt` (SANA via flags). See
**[Docs — Pipelines](https://lyttttt3333.github.io/sol-infer/)**.

## The five acceleration methods

1. **Cache (step-skip)** — TeaCache / EasyCache / SCSP (reuse a denoise step's output).
2. **Quantization (NVFP4)** — TransformerEngine 4-bit, step-selective.
3. **Kernel fusion (KWL)** — lossless AdaLN / qknorm+RoPE / FFN / gate fusions + compile.
4. **Sparse attention (PISA)** — piecewise block-sparse video self-attention.
5. **Token pruning** — drop low-salience video tokens at mid refine steps.

Details and trade-offs: **[Docs — Methods](https://lyttttt3333.github.io/sol-infer/)**.

## Quickstart

```bash
PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh && conda activate ./.conda/ltx23
uv pip install -e "python[diffusion]" --prerelease=allow
PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh

bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt   # example
```

- Human copy-paste guide: [`QUICKSTART.md`](QUICKSTART.md)
- Agent / portable-deploy guide: [`AGENTS.md`](AGENTS.md)
- Prompts (versioned, per model): `prompts/{sana,ltx,cosmos}/`

## Repository layout

```
scripts/{sana,cosmos,ltx}/   per-model inference + download entries
scripts/                     env/deploy helpers (create_code_conda_env, postinstall_cuda_jit, …)
prompts/{sana,ltx,cosmos}/   versioned prompts
docs/acceleration/           pipeline + method docs (human / agents)
python/sglang/multimodal_gen/  the runtime (DiTs, cache, quant, attention backends, efficiency framework)
```
