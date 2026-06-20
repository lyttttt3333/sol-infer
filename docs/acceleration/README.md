# Acceleration docs

Two categories, each written for two audiences (human prose vs. agent-actionable),
inspired by the layout of `NVlabs/Sana/docs`.

| Category | Human | Agents |
|---|---|---|
| **1. Optimization pipelines** (SANA-Video / Cosmos3-Super / LTX-2.3) | [human/pipelines.md](human/pipelines.md) | [agents/pipelines.md](agents/pipelines.md) |
| **2. Acceleration methods** (the 5 building blocks) | [human/methods.md](human/methods.md) | [agents/methods.md](agents/methods.md) |

- **Human** docs explain the *why* and *how* — design, trade-offs, when to use.
- **Agents** docs are terse and machine-actionable — exact flags / env vars / file
  paths / decision rules, no narration.

The five acceleration methods are: **Cache (step-skip)**, **Quantization (NVFP4)**,
**Kernel fusion (KWL)**, **Sparse attention (PISA)**, **Token pruning**. Each model
pipeline is one specific assembly of a subset of these five.

All speedups below are GB200, warmup-excluded, at each model's official spec.
