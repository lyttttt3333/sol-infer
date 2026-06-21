<p align="center" style="border-radius: 10px">
  <img src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/asset/logo.png" width="35%" alt="Sol-LTX-Infer logo"/>
</p>

<h1 align="center">Sol-LTX-Infer</h1>

<h3 align="center">
  Accelerated video-diffusion inference —
  <a href="https://lyttttt3333.github.io/sol-infer/pipelines/sana/">SANA-Video</a> ·
  <a href="https://lyttttt3333.github.io/sol-infer/pipelines/cosmos3/">Cosmos3-Super</a> ·
  <a href="https://lyttttt3333.github.io/sol-infer/pipelines/ltx/">LTX-2.3</a>
</h3>

<h3 align="center">
  <a href="https://lyttttt3333.github.io/sol-infer/">📖 Docs</a> &nbsp;|&nbsp;
  <a href="https://lyttttt3333.github.io/sol-infer/pipelines/sana/">Pipelines</a> &nbsp;|&nbsp;
  <a href="https://lyttttt3333.github.io/sol-infer/techniques/cache/">Techniques</a> &nbsp;|&nbsp;
  <a href="https://lyttttt3333.github.io/sol-infer/installation/">Install</a> &nbsp;|&nbsp;
  <a href="https://lyttttt3333.github.io/sol-infer/model_zoo/">Model Zoo</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/Efficient-Large-Model">🤗 HuggingFace</a>
</h3>

<p align="center">
  <a href="https://lyttttt3333.github.io/sol-infer/"><img src="https://img.shields.io/badge/🏠_Homepage-Sol--LTX--Infer-76b900?style=flat-square" alt="Homepage"/></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/📄_arXiv-XXXX.XXXXX-b31b1b?style=flat-square" alt="arXiv"/></a>
  <a href="https://lyttttt3333.github.io/sol-infer/"><img src="https://img.shields.io/badge/📖_Docs-github.io-blue?style=flat-square" alt="Docs"/></a>
  <a href="https://github.com/lyttttt3333/sol-infer"><img src="https://img.shields.io/github/stars/lyttttt3333/sol-infer?style=flat-square&label=⭐_Stars&color=76b900" alt="Stars"/></a>
  <a href="#-license"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square" alt="License"/></a>
</p>

<h4 align="center">
  Three production video-diffusion models · one clean acceleration line each · five reusable methods · up to 2.56× on GB200
</h4>

---

**Sol-LTX-Infer** is an efficiency-oriented inference codebase for high-resolution video
diffusion, built on [SGLang](https://github.com/sgl-project/sglang)'s `multimodal_gen`
runtime. It takes three production models and reduces each to **one unambiguous
acceleration line** (plus a dense `baseline`), composed from **five reusable
acceleration methods** that each own a disjoint seam so they stack without interfering.
All speedups below are measured on **GB200, warmup-excluded**, at each model's official
spec.

## 📰 News

- **[2026/06]** 🔥 Full documentation site live: [3 pipeline designs + 5 acceleration techniques](https://lyttttt3333.github.io/sol-infer/), each technique with per-method literature surveys and paper links.
- **[2026/06]** 🔥 SANA-Video fast path solved — `--max-autotune` (subprocess autotune + persistent Inductor cache) reaches **2.56×** at 480p; cold-safe `default` mode gives **2.10×** anywhere.
- **[2026/06]** ✅ LTX-2.3 `fullopt` composes all five methods (KWL + cache + PISA + NVFP4 + token-prune) for **~2.4×**.
- **[2026/06]** ✅ Cosmos3-Super `fullopt` (TeaCache + step-selective NVFP4) reaches **~2.26×** on 4×GB200.
- **[2026/06]** ✅ Each model collapsed to a single `baseline | fullopt` entry; `scripts/` split per model.

## ⚡ Models & speedups

| Model | Size / GPUs | Acceleration line | Warm baseline → fullopt | Speedup |
|---|---|---|---|---|
| **[SANA-Video](https://lyttttt3333.github.io/sol-infer/pipelines/sana/)** | 2B / 1 | EasyCache + fusion + compile | 28.5 s → 13.5 s (11.0 s warm) | **2.10× / 2.56×** |
| **[Cosmos3-Super](https://lyttttt3333.github.io/sol-infer/pipelines/cosmos3/)** | 64B / 4 | TeaCache + step-selective NVFP4 | 97.2 s → 43.1 s | **~2.26×** |
| **[LTX-2.3](https://lyttttt3333.github.io/sol-infer/pipelines/ltx/)** | 1080p/10s / 1 | KWL fusion + cache + PISA + NVFP4 + token-prune | 95.7 s → 39.2 s | **~2.4×** |

<sub>GB200, warmup-excluded, official spec per model. SANA 480p (832×480, 81f, 50 steps); Cosmos3 1280×720, 189f, 35 steps; LTX 1088×1920, 241f.</sub>

## 🧩 The five acceleration methods

Each method owns a distinct seam — so the framework composes them and each stays
off==identity when disabled.

| # | Method | What it does | Type | Docs |
|---|---|---|---|---|
| 1 | **Cache (step-skip)** | reuse a denoise step's output (TeaCache / EasyCache / SCSP) | lossy | [cache](https://lyttttt3333.github.io/sol-infer/techniques/cache/) |
| 2 | **Quantization** | TransformerEngine NVFP4 4-bit, step-selective | lossy | [quant](https://lyttttt3333.github.io/sol-infer/techniques/quant/) |
| 3 | **Kernel fusion (KWL)** | fuse the memory-bound DiT glue (AdaLN, QK-norm+RoPE, gates, FFN) | lossless | [kernel](https://lyttttt3333.github.io/sol-infer/techniques/kernel/) |
| 4 | **Sparse attention (PISA)** | piecewise block-sparse video self-attention | lossy | [sparse](https://lyttttt3333.github.io/sol-infer/techniques/sparse/) |
| 5 | **Token pruning** | drop low-salience video tokens at mid refine steps | lossy | [token-prune](https://lyttttt3333.github.io/sol-infer/techniques/token_prune/) |

## 🚀 Quick start

```bash
# 1. environment
git clone https://github.com/lyttttt3333/sol-infer.git && cd sol-infer
PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh && conda activate "$PWD/.conda/ltx23"

# 2. dependencies (torch 2.11+cu130, diffusers 0.38) + CUDA-JIT fixups
uv pip install -e "$PWD/python[diffusion]" --prerelease=allow
PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh   # add --with-te for NVFP4

# 3. run (each entry takes `baseline | fullopt`)
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt
bash scripts/cosmos/slurm_cosmos3_super.sh fullopt
.conda/ltx23/bin/python scripts/sana/sana_video_sglang_run.py \
    --easycache 0.1 --linattn-bf16 --qkv-merge --compile   # add --max-autotune for peak
```

Full guide: **[Installation](https://lyttttt3333.github.io/sol-infer/installation/)** ·
**[Model Zoo](https://lyttttt3333.github.io/sol-infer/model_zoo/)** ·
copy-paste [`QUICKSTART.md`](QUICKSTART.md) · portable/agent [`AGENTS.md`](AGENTS.md).

## 📖 Getting started

- 📚 **[Full documentation](https://lyttttt3333.github.io/sol-infer/)** — one MkDocs site, everything below linked
- 🛠️ **[Installation](https://lyttttt3333.github.io/sol-infer/installation/)** — conda env, editable install, CUDA-JIT fixups
- 🗂️ **[Model Zoo](https://lyttttt3333.github.io/sol-infer/model_zoo/)** — HF repos + download helpers
- 🎬 **Optimized pipelines** — [SANA-Video](https://lyttttt3333.github.io/sol-infer/pipelines/sana/) · [Cosmos3-Super](https://lyttttt3333.github.io/sol-infer/pipelines/cosmos3/) · [LTX-2.3](https://lyttttt3333.github.io/sol-infer/pipelines/ltx/)
- ⚙️ **Acceleration techniques** — [Cache](https://lyttttt3333.github.io/sol-infer/techniques/cache/) · [Quantization](https://lyttttt3333.github.io/sol-infer/techniques/quant/) · [Kernel fusion](https://lyttttt3333.github.io/sol-infer/techniques/kernel/) · [Sparse attention](https://lyttttt3333.github.io/sol-infer/techniques/sparse/) · [Token pruning](https://lyttttt3333.github.io/sol-infer/techniques/token_prune/)

## 🔬 Key techniques

- **Step-skip caching** — TeaCache (accumulated rel-L1 of timestep-modulated input), EasyCache (calibration-free runtime adaptive), SCSP (stage-1 step-skip preset).
- **Step-selective NVFP4** — TransformerEngine 4-bit GEMMs on the heavy linears, first/last denoise steps kept in BF16 where quality is most sensitive; graceful BF16 fallback off Blackwell.
- **KWL kernel fusion** — hand-written Triton/CuTeDSL kernels collapse the memory-bound glue around attention/FFN (RMS+AdaLN, Q/K-norm+split-RoPE, dual modulation, all-9 Ada values, residual gate, FFN proj+GELU, audio QKVG, VAE GroupNorm+SiLU) into single launches; algorithm-lossless.
- **PISA sparse attention** — piecewise block-sparse video self-attention (exact selected blocks + centroid remainder).
- **Token pruning** — drop low-salience video tokens during the less-sensitive middle refine steps, then scatter back.
- **Branch sharing** — CFG/STG block-0 self-attention and guidance-prefix sharing where branches are provably equivalent.

## ✅ To-do

- [x] Single `baseline | fullopt` entry per model
- [x] Per-model `scripts/` + versioned `prompts/`
- [x] Full MkDocs documentation site (pipelines + techniques)
- [x] SANA-Video max-autotune fast path
- [ ] Fill the real Homepage / arXiv URLs
- [ ] Public model + recipe release notes
- [ ] More backends on the default lines (SVG2 / STA ablations)

## 🙏 Acknowledgements

Built on [SGLang](https://github.com/sgl-project/sglang) and
[🤗 Diffusers](https://github.com/huggingface/diffusers). Pipelines wrap
[SANA-Video](https://github.com/NVlabs/Sana), NVIDIA
[Cosmos](https://github.com/NVIDIA/Cosmos), and
[Lightricks LTX-Video](https://github.com/Lightricks/LTX-Video). Acceleration methods
draw on TeaCache, EasyCache, SVDQuant/Nunchaku, FlashAttention,
[TransformerEngine](https://github.com/NVIDIA/TransformerEngine), and the sparse-attention
/ token-reduction literature surveyed in the [docs](https://lyttttt3333.github.io/sol-infer/).

## 📌 Citation

```bibtex
@misc{sol-ltx-infer-2026,
  title  = {Sol-LTX-Infer: Accelerated Video-Diffusion Inference},
  author = {Sol-LTX-Infer Contributors},
  year   = {2026},
  howpublished = {\url{https://github.com/lyttttt3333/sol-infer}},
}
```

## 📄 License

Apache-2.0. Model weights follow their respective upstream licenses (SANA-Video,
NVIDIA Cosmos, Lightricks LTX) — see each model card.

## ⭐ Star history

<a href="https://star-history.com/#lyttttt3333/sol-infer&Date">
  <img src="https://api.star-history.com/svg?repos=lyttttt3333/sol-infer&type=Date" width="60%" alt="Star History Chart"/>
</a>
