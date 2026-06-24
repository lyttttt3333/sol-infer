# Sol-Engine

Sol-Engine is an efficiency-oriented inference codebase for high-resolution video
diffusion. It keeps the older three-page documentation layout: one homepage, one
pipeline page, and one acceleration-method page.

## Contents

- **[Optimization pipelines](pipelines.md)**: the concrete baseline and full-opt
  assembly for Cosmos3-Super, LTX-2.3, and SANA-Video.
- **[Acceleration methods](acceleration.md)**: the five reusable methods and the
  implementation files behind them.

## Models and speedups

| Model | Acceleration line | Speedup |
|---|---|---:|
| [Cosmos3-Super (4xB200)](pipelines.md) | TeaCache + NVFP4 | ~2.26x |
| [LTX-2.3 (1xB200)](pipelines.md) | kernel fusion + cache + PISA + NVFP4 + token-prune | ~2.4x |
| [SANA-Video (1xB200)](pipelines.md) | EasyCache + fusion + compile | 2.77x |

Timings are warmup-excluded. SANA uses 480p, 81 frames, 50 steps; Cosmos3 uses
1280x720, 189 frames, 35 steps; LTX uses 1088x1920, 241 frames.

## Quick start

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

Use this goal with Claude Code or Codex from the repository root.

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
