# Caching Acceleration

## Overview

SGLang has several cache-style acceleration paths for diffusion transformer
models. They are not equivalent: some skip full transformer block-stack calls,
some skip individual blocks, and some reuse attention results inside a block.

| Strategy | Skip scope | How acceleration happens | Current LTX-2.3 status |
|----------|------------|--------------------------|------------------------|
| **TeaCache** | Timestep/block-stack residual replay | Compare consecutive modulated inputs. On a cache hit, skip the LTX2 transformer block stack for that denoising step and replay cached video/audio residuals. Output norm/projection/unpatchify still run. | Primary path for current HQ/non-HQ tests. Stage 1 enabled, stage 2 disabled by default. |
| **PAB** | Attention broadcast windows | Reuse attention outputs over configured spatial, temporal, and cross-attention windows. | Needs validation before relying on results. A previous 1:1 speed/identical-output run indicates the tested configuration did not produce effective skips. |
| **DBCache / Cache-DiT** | DiT block-level reuse | Skip selected DiT blocks from a residual-difference policy. | Implemented through Cache-DiT env flags, but the aggressive preset had visible quality issues in earlier LTX-2.3 samples. Revisit with fewer skips if targeting about 1.5x. |
| **LTX2 stage1 cache-core** | LTX2 stage-1 residual reuse | Handwritten LTX2 stage-1 cache path inside the block stack. | Experimental branch baseline. Use as a separate ablation from TeaCache. |
| **KWL baseline** | No cache skip | Keeps selected kernel/fusion optimizations while computing every denoising step. | Reference for speedup and visual comparisons. |

For benchmark reporting, always include both the runtime speedup and the
mechanism that produced it: skipped step indices, hit/compute counts, and the
pipeline section being accelerated. A total-pipeline speedup can hide useful
stage-1 acceleration when stage 2, VAE decode, LoRA switching, or CPU offload
dominates the wall clock.

## LTX-2.3 benchmark harness

The current LTX-2.3 cache benchmark scripts use 10 second prompts with concrete
human and animal scenes:

```bash
bash scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh
```

The runner covers:

- HQ: `LTX2TwoStageHQPipeline`, 15 stage-1 steps, res2s sampler.
- Non-HQ: `LTX2TwoStagePipeline`, 30 stage-1 steps, euler sampler.
- Variants: KWL baseline plus TeaCache `t=0.04/start=6`,
  `t=0.06/start=5`, and `t=0.08/start=5`.
- Outputs: per-case `out.mp4`, `perf.json`, `*_semantics.json`, logs, side by
  side compare videos, `benchmark_summary.json`, `benchmark_summary.md`, and
  `benchmark_report.html`.

Latest run (`ltx23-teacache-hq-nonhq-matrix-10s-full-4545670`, 2026-06-01):

- TeaCache did skip the intended LTX2 stage-1 transformer block stack. HQ skipped
  steps 6-13 for `0.04/start6` and 5-13 for `0.06/start5`/`0.08/start5`.
  Non-HQ skipped 8-11 of 30 stage-1 steps depending on threshold.
- Stage-1 speedup was meaningful: about 1.32-1.42x on HQ and 1.29-1.50x on
  non-HQ across the two prompts.
- End-to-end speedup was limited in this offloaded two-stage setup: HQ averaged
  0.93-0.99x for TeaCache variants, while non-HQ averaged 0.91-1.01x. The
  denoising savings were often offset by uncached stage 2 plus LoRA/offload
  overhead.
- For current LTX-2.3 experiments, use TeaCache numbers as evidence of stage-1
  acceleration and inspect compare videos before accepting visual quality. Do
  not treat total pipeline speedup from this offload configuration as the upper
  bound of the cache algorithm itself.

## Cache-DiT

[Cache-DiT](https://github.com/vipshop/cache-dit) provides block-level caching with
advanced strategies like DBCache and TaylorSeer. It can achieve up to **1.69x speedup**.

See [cache_dit.md](cache_dit.md) for detailed configuration.

### Quick Start

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

### Key Features

- **DBCache**: Dynamic block-level caching based on residual differences
- **TaylorSeer**: Taylor expansion-based calibration for optimized caching
- **SCM**: Step-level computation masking for additional speedup

## TeaCache

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

See [teacache.md](teacache.md) for detailed documentation.

### Quick Overview

- Tracks relative L1 distance between modulated inputs across timesteps.
- When accumulated distance is below threshold, reuses cached residual.
- For LTX-2.3, skips the transformer block stack and still runs output
  norm/projection/unpatchify/decode.
- Supports CFG with separate positive/negative caches where the model hook
  provides branch separation.

### Supported Models

- LTX-2.3
- Wan (wan2.1, wan2.2)
- Hunyuan (HunyuanVideo)
- Z-Image

For Flux and Qwen models, TeaCache is automatically disabled when CFG is enabled.

```{toctree}
:maxdepth: 1

cache_dit
teacache
```

## References

- [Cache-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
