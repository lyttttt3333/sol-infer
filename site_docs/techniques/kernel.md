# Kernel fusion

Kernel fusion reduces memory-bound overhead inside DiT blocks. The target is the repeated glue around GEMMs: layout movement, normalization, activation, gate application, QK normalization, RoPE, and precision conversion.

## In Sol-Engine

| Pipeline | Fusion path | Role |
|---|---|---|
| SANA-Video | Linear attention BF16, QKV merge, compile | Part of the ~2.77x path |
| LTX-2.3 | KWL fusion | Lossless operator-level optimization before lossy methods |

## Scope

Fusion should not change scheduler settings, prompts, seeds, or the number of denoising steps. It is an implementation-level optimization that reduces launch and memory traffic overhead.
