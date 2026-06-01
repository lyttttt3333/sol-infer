# TeaCache

> **Note**: This is one of two caching strategies available in SGLang.
> For an overview of all caching options, see [caching](../index.md).

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

## Overview

TeaCache works by:
1. Tracking the L1 distance between modulated inputs across consecutive timesteps
2. Accumulating the rescaled L1 distance over steps
3. When accumulated distance is below a threshold, reusing the cached residual
4. Supporting CFG (Classifier-Free Guidance) with separate positive/negative caches

## How It Works

### L1 Distance Tracking

At each denoising step, TeaCache computes the relative L1 distance between the current and previous modulated inputs:

```
rel_l1 = |current - previous|.mean() / |previous|.mean()
```

This distance is then rescaled using polynomial coefficients and accumulated:

```
accumulated += poly(coefficients)(rel_l1)
```

### Cache Decision

- If `accumulated >= threshold`: Force computation, reset accumulator
- If `accumulated < threshold`: Skip computation, use cached residual

### CFG Support

For models that support CFG cache separation (Wan, Hunyuan, Z-Image), TeaCache maintains separate caches for positive and negative branches:
- `previous_modulated_input` / `previous_residual` for positive branch
- `previous_modulated_input_negative` / `previous_residual_negative` for negative branch

For models that don't support CFG separation (Flux, Qwen), TeaCache is automatically disabled when CFG is enabled.

## Configuration

TeaCache is configured via `TeaCacheParams` in the sampling parameters:

```python
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams

params = TeaCacheParams(
    teacache_thresh=0.1,           # Threshold for accumulated L1 distance
    coefficients=[1.0, 0.0, 0.0],  # Polynomial coefficients for L1 rescaling
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `teacache_thresh` | float | Threshold for accumulated L1 distance. Higher = more caching, faster but potentially lower quality |
| `coefficients` | list[float] | Polynomial coefficients for L1 rescaling. Model-specific tuning |

### Model-Specific Configurations

Different models may have different optimal configurations. The coefficients are typically tuned per-model to balance speed and quality.

## Supported Models

TeaCache is built into the following model families:

| Model Family | CFG Cache Separation | Notes |
|--------------|---------------------|-------|
| LTX-2.3 | Stage/pass keyed residual replay | SGLang runtime hook for HQ and non-HQ benchmark variants |
| Wan (wan2.1, wan2.2) | Yes | Full support |
| Hunyuan (HunyuanVideo) | Yes | To be supported |
| Z-Image | Yes | To be supported |
| Flux | No | To be supported |
| Qwen | No | To be supported |

## LTX-2.3 Runtime Hook

For LTX-2.3, TeaCache is implemented as model-level residual replay in:

```text
python/sglang/multimodal_gen/runtime/cache/ltx2_teacache.py
python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py
```

The hook caches the residual from the transformer block stack for each
`stage/pass/shape` key. On a cache hit it skips the transformer blocks and
replays:

```text
hidden_states = hidden_states + cached_video_residual
audio_hidden_states = audio_hidden_states + cached_audio_residual
```

The output norm, projection, and unpatchify/decode path still runs every step.
By default only stage 1 is enabled because stage 2 has only a few refinement
steps and is more visually sensitive.

### Environment Variables

| Variable | Meaning | Default in benchmark variants |
|----------|---------|-------------------------------|
| `SGLANG_LTX2_TEACACHE_ENABLED` | Enable the LTX-2 TeaCache hook | `1` for TeaCache variants |
| `SGLANG_LTX2_TEACACHE_THRESH` | Accumulated relative L1 threshold; larger means more skips | `0.04`, `0.06`, or `0.08` |
| `SGLANG_LTX2_TEACACHE_START` | First denoising step eligible for replay | `6` for c04, `5` for c06/c08 |
| `SGLANG_LTX2_TEACACHE_STAGE2_DISABLE` | Disable TeaCache on stage 2 | `1` |
| `SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS` | Cap consecutive cache hits before recompute | `1` |

### LTX-2.3 Benchmark Variants

HQ, 15-step stage-1 `LTX2TwoStageHQPipeline`:

```bash
bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl
bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl_teacache_c04_s6
bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl_teacache_c06_s5
```

Non-HQ, 30-step stage-1 `LTX2TwoStagePipeline`:

```bash
bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh kwl
bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh kwl_cache_teacache_c04_s6
bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh kwl_cache_teacache_c06_s5
```

The combined matrix runner produces videos plus JSON/Markdown/HTML reports:

```bash
bash scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh
```

The report includes total, denoising, stage-1, and stage-2 timing plus TeaCache
hit/compute counts and skipped step indices parsed from runtime logs.


## References

- [TeaCache: Accelerating Diffusion Models with Temporal Similarity](https://arxiv.org/abs/2411.14324)
