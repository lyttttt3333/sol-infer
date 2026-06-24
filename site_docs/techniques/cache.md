# Cache

Cache methods exploit algorithm-level redundancy across adjacent diffusion denoising steps. The latent state changes gradually, so selected intermediate work can be reused when a step is sufficiently similar to a prior step.

## In Sol-Engine

| Pipeline | Cache path | Role |
|---|---|---|
| SANA-Video | EasyCache | Main SANA acceleration component |
| Cosmos3-Super | TeaCache | Residual replay across denoising steps |
| LTX-2.3 | Stage-specific cache | Part of the full LTX optimization stack |

## Practical notes

- Cache thresholds trade speed for visual fidelity.
- Timing should be read from warmup-excluded log lines.
- Quality checks should compare generated videos, not just total latency.
