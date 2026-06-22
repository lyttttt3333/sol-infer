<p align="center" style="border-radius: 10px">
  <img src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/asset/logo.png" width="35%" alt="Sol-Engine logo"/>
</p>

<h1 align="center">Sol-Engine</h1>

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
  <a href="https://lyttttt3333.github.io/sol-infer/installation/">Install</a>
</h3>

<p align="center">
  <a href="https://lyttttt3333.github.io/sol-infer/"><img src="https://img.shields.io/badge/🏠_Homepage-Sol--Engine-76b900?style=flat-square" alt="Homepage"/></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/📄_arXiv-XXXX.XXXXX-b31b1b?style=flat-square" alt="arXiv"/></a>
  <a href="https://lyttttt3333.github.io/sol-infer/"><img src="https://img.shields.io/badge/📖_Docs-github.io-blue?style=flat-square" alt="Docs"/></a>
  <a href="#-license"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square" alt="License"/></a>
</p>

<h4 align="center">
  Agent-native workflow · Full-stack acceleration techniques · Three T2V models, best implementations
</h4>

---

**Sol-Engine** is an efficiency-oriented inference codebase for high-resolution video
diffusion, built on [SGLang](https://github.com/sgl-project/sglang)'s `multimodal_gen`
runtime. It features an **agent-native inference workflow** and reduces three production
models into **one unambiguous acceleration line**. This is powered by a full-stack
solution composed of **five reusable acceleration techniques**, delivering a **2× to 3×
end-to-end speedup** across the three models. We are actively continuing development to
support a wider range of models.

## 📰 News

- **[2026/06]** 🔥 **SANA-Video** — EasyCache + kernel fusion + torch.compile → **2.77×** end-to-end (29.4 s → 10.6 s).
- **[2026/06]** 🔥 **Cosmos3-Super** — TeaCache + step-selective NVFP4 → **~2.26×** end-to-end (4×GB200).
- **[2026/06]** 🔥 **LTX-2.3** — KWL fusion + cache + PISA + NVFP4 + token-prune → **~2.4×** end-to-end.
- **[2026/06]** 📖 **Docs release** — full documentation site live: [3 pipeline designs + 5 acceleration techniques](https://lyttttt3333.github.io/sol-infer/), each technique with per-method literature surveys and paper links.

## ⚡ Models & speedups

<div align="center">

| Model | Params | Acceleration line | Speedup |
|---|---|---|---|
| **[SANA-Video](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers)** | 2B | EasyCache + fusion + compile | **2.77×** |
| **[Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super)** | 64B | TeaCache + step-selective NVFP4 | **~2.26×** |
| **[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)** | 22B | KWL fusion + cache + PISA + NVFP4 + token-prune | **~2.4×** |

</div>

<sub>GB200, warmup-excluded. SANA 480p (832×480, 81f, 50 steps); Cosmos3 1280×720, 189f, 35 steps; LTX 1088×1920, 241f.</sub>

## 🧩 The five acceleration methods

Each method owns a distinct seam — so the framework composes them and each stays
off==identity when disabled.

<div align="center">

| # | Method | What it does |
|---|---|---|
| 1 | **[Cache](https://lyttttt3333.github.io/sol-infer/techniques/cache/)** | reuse a denoise step's output (TeaCache / EasyCache / fix-step) |
| 2 | **[Quantization](https://lyttttt3333.github.io/sol-infer/techniques/quant/)** | TransformerEngine NVFP4 4-bit, step-selective |
| 3 | **[Kernel fusion](https://lyttttt3333.github.io/sol-infer/techniques/kernel/)** | fuse the memory-bound DiT glue (AdaLN, QK-norm+RoPE, gates, FFN) |
| 4 | **[Sparse attention](https://lyttttt3333.github.io/sol-infer/techniques/sparse/)** | piecewise block-sparse video self-attention |
| 5 | **[Token pruning](https://lyttttt3333.github.io/sol-infer/techniques/token_prune/)** | drop low-salience video tokens at mid refine steps |

</div>

## 🚀 Quick start (agent-native)

Sol-Engine is installed and launched the **agent-native** way. Rather than hand-running
the setup steps, you hand a coding agent — OpenAI **Codex** or **Claude Code** — a single
goal and let it create the environment, fetch the weights, and run all three models in
both `baseline` and `fullopt` settings, **troubleshooting and adapting the scripts to
your machine** as it goes.

From the repo root, give the agent this goal:

```text
/goal Execute the inference code for the three models using both baseline and full-opt
settings with the following requirements. Refer to AGENTS.md for the environment creation,
model download, and inference guides. For the environment, you need to create a new
environment. For model weights, you are allowed to reuse existing weights if they are
locally available; otherwise, you need to download them. Regarding adaptability, be aware
that the provided guides for environment creation, download scripts, and inference may
contain system incompatibilities, so you are expected to troubleshoot and adapt them to
your specific machine.
```

## 📖 Getting started

- 📚 **[Full documentation](https://lyttttt3333.github.io/sol-infer/)** — a comprehensive guidebook to the whole project: pipeline designs, acceleration techniques, setup, and model references in one place
- 🛠️ **[Installation](https://lyttttt3333.github.io/sol-infer/installation/)** — conda env, editable install, CUDA-JIT fixups, and the HF model repos + download helpers
- 🎬 **Optimized pipelines** — [SANA-Video](https://lyttttt3333.github.io/sol-infer/pipelines/sana/) · [Cosmos3-Super](https://lyttttt3333.github.io/sol-infer/pipelines/cosmos3/) · [LTX-2.3](https://lyttttt3333.github.io/sol-infer/pipelines/ltx/)
- ⚙️ **Acceleration techniques** — [Cache](https://lyttttt3333.github.io/sol-infer/techniques/cache/) · [Quantization](https://lyttttt3333.github.io/sol-infer/techniques/quant/) · [Kernel fusion](https://lyttttt3333.github.io/sol-infer/techniques/kernel/) · [Sparse attention](https://lyttttt3333.github.io/sol-infer/techniques/sparse/) · [Token pruning](https://lyttttt3333.github.io/sol-infer/techniques/token_prune/)

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
@misc{sol-engine-2026,
  title  = {Sol-Engine: Accelerated Video-Diffusion Inference},
  author = {Sol-Engine Contributors},
  year   = {2026},
  howpublished = {\url{https://github.com/lyttttt3333/sol-infer}},
}
```

## 📄 License

Apache-2.0. Model weights follow their respective upstream licenses (SANA-Video,
NVIDIA Cosmos, Lightricks LTX) — see each model card.
