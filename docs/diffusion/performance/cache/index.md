# Caching Acceleration

## Overview

SGLang has several cache-style acceleration paths for diffusion transformer
models. They are not equivalent: some skip full transformer block-stack calls,
some skip individual blocks, and some reuse attention results inside a block.

The current LTX-2.3 cache experiment README lives in this directory:
[README.md](README.md).

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
- Stage-1 inspection: set `STAGE1_ONLY_OUTPUT=1` to skip stage 2 and decode the
  upsampled stage-1 latents as the final video. Set `SAVE_STAGE1_OUTPUT=1` to
  additionally save `stage1_out.mp4` while still running the normal stage-2
  refine path.

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

### HQ stage-1 color probe

Prompt 1 in the TeaCache matrix showed an overexposed HQ KWL baseline while the
non-HQ KWL output looked normal. The stage-1-only probe was run to separate
stage 1, stage 2, LoRA, and KWL fused-ada effects:

```bash
outputs/ltx23-hq-stage1-lora-ablation-20260602-10/stage1_lora_ada_compare.mp4
outputs/ltx23-hq-stage1-lora-ablation-20260602-10/color_stats.json
```

First-frame color statistics from that run:

| Case | Luma mean | RGB mean | Clip >=250 RGB | Notes |
|---|---:|---|---|---|
| HQ stage1 KWL, fused-ada on | 200.10 | `[195.84, 220.49, 26.80]` | `[0.3579, 0.4730, 0.0351]` | Stage-1-only output, very bright. |
| HQ stage1 KWL, fused-ada off | 200.10 | `[195.84, 220.49, 26.80]` | `[0.3579, 0.4730, 0.0351]` | Same as fused-ada on in this probe. |
| Non-HQ stage1 KWL | 99.23 | `[116.77, 102.32, 16.74]` | `[0.0067, 0.0006, 0.0001]` | Same prompt/seed, normal exposure. |
| Old HQ final KWL | 214.15 | `[205.51, 238.31, 19.70]` | `[0.3923, 0.7791, 0.0314]` | Stage-2 refine worsens green clipping. |
| Old non-HQ final KWL | 75.05 | `[85.04, 78.97, 6.74]` | `[0.0006, 0.0000, 0.0000]` | Normal exposure. |

Interpretation:

- The HQ exposure issue is already visible in the decoded upsampled stage-1
  latents, before stage-2 refine.
- `SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL` did not change the stage-1-only output in
  this probe, so fused-ada is not the root cause of the HQ stage-1 brightness.
- Logs from this completed probe showed the requested `stage1_lora=0.0` cases
  were still applying the default stage-1 distilled LoRA strength `0.25`, because
  `LTX23HQSamplingParams.build_request_extra()` default values overrode the env
  override. The code now makes `SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_1` and
  `SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_2` take precedence for experiments.
  A follow-up full-video LoRA=0 rerun was attempted but canceled after repeated
  cluster startup stalls at text-encoder load, so LoRA=0 visual quality is still
  an open ablation rather than an accepted conclusion.

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
README
teacache
```

## References

- [Cache-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
