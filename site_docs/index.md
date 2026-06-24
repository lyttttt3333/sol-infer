# Sol-Engine

Sol-Engine is an efficiency-oriented inference codebase for high-resolution video diffusion. It wraps SANA-Video, Cosmos3-Super, and LTX-2.3 with one explicit acceleration line per model.

[GitHub](https://github.com/lyttttt3333/sol-infer){ .md-button .md-button--primary }
[Paper](https://arxiv.org/abs/2606.23743){ .md-button }

## Models and speedups

| Model | Params | Acceleration line | Speedup |
|---|---:|---|---:|
| [SANA-Video](pipelines/sana.md) | 2B | EasyCache + fusion + compile | ~2.77x |
| [Cosmos3-Super](pipelines/cosmos3.md) | 64B | TeaCache + step-selective NVFP4 | ~2.26x |
| [LTX-2.3](pipelines/ltx.md) | 22B | KWL fusion + cache + PISA + NVFP4 + token-prune | ~2.4x |

GB200 timings, warmup excluded. SANA uses 480p, 81 frames, 50 steps; Cosmos3 uses 1280x720, 189 frames, 35 steps; LTX uses 1088x1920, 241 frames.

## Acceleration levels

Video diffusion inference exposes redundancy at three complementary levels:

- **Algorithm level**: adjacent denoising steps run structurally similar computation over slowly changing latents.
- **Model level**: long spatiotemporal sequences contain redundant tokens and attention interactions.
- **Kernel level**: DiT blocks repeatedly launch memory-bound work around GEMMs, layout movement, normalization, activation, and precision conversion.

Sol-Engine composes five reusable techniques across these levels: [cache](techniques/cache.md), [quantization](techniques/quant.md), [kernel fusion](techniques/kernel.md), [sparse attention](techniques/sparse.md), and [token pruning](techniques/token_prune.md).

## Start here

- [Installation](installation.md): environment creation, CUDA JIT fixups, and model downloads.
- [Pipelines](pipelines/sana.md): optimized launch paths for SANA-Video, Cosmos3-Super, and LTX-2.3.
- [Techniques](techniques/cache.md): the five acceleration methods and where they apply.

## Citation

```bibtex
@misc{li2026solvideoinferenceengine,
  title         = {Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation},
  author        = {Yitong Li and Junsong Chen and Haopeng Li and Haozhe Liu and Jincheng Yu and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
  year          = {2026},
  eprint        = {2606.23743},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2606.23743},
  url           = {https://arxiv.org/abs/2606.23743},
}
```
