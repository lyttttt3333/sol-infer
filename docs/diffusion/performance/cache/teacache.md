# TeaCache

> **Note**: This is one cache strategy available in SGLang.
> For an overview of all caching options, see [caching](index.md).

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

After each denoising stage, the runtime resets TeaCache state. This releases the
cached residual tensors before stage 2 starts and prevents stage-1 residuals
from remaining live on GPU memory during the refinement stage.

### Environment Variables

| Variable | Meaning | Default in benchmark variants |
|----------|---------|-------------------------------|
| `SGLANG_LTX2_TEACACHE_ENABLED` | Enable the LTX-2 TeaCache hook | `1` for TeaCache variants |
| `SGLANG_LTX2_TEACACHE_THRESH` | Accumulated relative L1 threshold; larger means more skips | `0.04`, `0.06`, or `0.08` |
| `SGLANG_LTX2_TEACACHE_START` | First denoising step eligible for replay | `6` for c04, `5` for c06/c08 |
| `SGLANG_LTX2_TEACACHE_STAGE2_DISABLE` | Disable TeaCache on stage 2 | `1` |
| `SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS` | Cap consecutive cache hits before recompute | `1` |
| `SGLANG_LTX2_TEACACHE_PERIODIC_RECOMPUTE_STEPS` | Optional fixed recompute cadence; `0` disables it | `0` |

### LTX-2.3 Benchmark Variants

HQ, 15-step stage-1 `LTX2TwoStageHQPipeline`:

```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh kwl
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh kwl_teacache_c04_s6
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh kwl_teacache_c06_s5
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh kwl_teacache_c08_s5
```

Non-HQ, 30-step stage-1 `LTX2TwoStagePipeline`:

```bash
bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh kwl
bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh kwl_cache_teacache_c04_s6
bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh kwl_cache_teacache_c06_s5
bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh kwl_cache_teacache_c08_s5
```

The combined matrix runner produces videos plus JSON/Markdown/HTML reports:

```bash
bash scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh
```

The report includes total, denoising, stage-1, and stage-2 timing plus TeaCache
hit/compute counts and skipped step indices parsed from runtime logs.

For visual debugging, the same runners support stage-1 output:

```bash
STAGE1_ONLY_OUTPUT=1 bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh kwl
SAVE_STAGE1_OUTPUT=1 bash scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh
```

`STAGE1_ONLY_OUTPUT=1` skips stage-2 refine and decodes the upsampled stage-1
latents as `stage1_out.mp4`. `SAVE_STAGE1_OUTPUT=1` keeps the normal final
`out.mp4` and additionally writes `stage1_out.mp4` for each case.

### Reading the Speedup

For LTX-2.3, TeaCache does not make every part of the request faster. The
reported hit count only applies to denoising steps where the transformer block
stack was skipped. Text encoding, VAE decode, audio/vocoder work, LoRA switching,
stage-2 refinement when disabled, and output writing still run. For this reason,
the report shows both total-pipeline speedup and stage-specific speedups.

Example interpretation:

- `skip_steps=[6, 7, 8, 9, 10, 11, 12, 13]` means the transformer block stack was
  skipped on those stage-1 denoising steps.
- `computes=27, hits=24` counts per stage/pass/cache key, not just human-visible
  denoising step numbers.
- A strong stage-1 speedup can still become a modest total speedup if stage 2 or
  decode dominates the wall clock.

### LTX-2.3 10s Benchmark, 2026-06-01

Run: `ltx23-teacache-hq-nonhq-matrix-10s-full-4545670`.

Main artifacts:

```bash
outputs/ltx23-teacache-hq-nonhq-matrix-10s-full-4545670/benchmark_summary.json
outputs/ltx23-teacache-hq-nonhq-matrix-10s-full-4545670/benchmark_summary.md
outputs/ltx23-teacache-hq-nonhq-matrix-10s-full-4545670/benchmark_report.html
outputs/ltx23-teacache-hq-nonhq-matrix-10s-full-4545670/hq/prompt_1/compare.mp4
outputs/ltx23-teacache-hq-nonhq-matrix-10s-full-4545670/nonhq/prompt_1/compare.mp4
```

Prompts:

- Prompt 0: elderly ceramic artist painting blue patterns on a porcelain vase.
- Prompt 1: red fox running through tall grass at sunrise.

Runtime placement:

- `performance_mode=speed`
- `SGLANG_LTX2_DIT_CPU_OFFLOAD=1`
- `SGLANG_LTX2_VAE_CPU_OFFLOAD=1`
- `SGLANG_LTX2_LAYERWISE_OFFLOAD_COMPONENTS=text_encoder`
- `SGLANG_LTX2_TWO_STAGE_DEVICE_MODE=snapshot`
- `SGLANG_LTX2_PIN_CPU_MEMORY=false`

Key result: TeaCache consistently accelerates stage-1 denoising, but the
end-to-end pipeline speedup is small or negative in this offloaded two-stage
setup because stage 2 is intentionally not cached and LoRA/offload/snapshot
stages dominate or fluctuate. The visual decision should be made from the
generated compare videos, not from speed alone.

| Pipeline | Prompt | Variant | Total s | Total x | Stage1 s | Stage1 x | Stage2 s | Stage-1 skipped steps | Hits/computes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| HQ 15-step | 0 | KWL baseline | 343.07 | 1.000 | 150.67 | 1.000 | 68.27 | - | - |
| HQ 15-step | 0 | TeaCache 0.04/start6 | 342.34 | 1.002 | 114.26 | 1.319 | 87.30 | 6-13 | 24/27 |
| HQ 15-step | 0 | TeaCache 0.06/start5 | 345.97 | 0.992 | 105.85 | 1.423 | 88.39 | 5-13 | 27/30 |
| HQ 15-step | 0 | TeaCache 0.08/start5 | 346.26 | 0.991 | 106.19 | 1.419 | 91.22 | 5-13 | 27/30 |
| Non-HQ 30-step | 0 | KWL baseline | 242.78 | 1.000 | 166.49 | 1.000 | 31.91 | - | - |
| Non-HQ 30-step | 0 | TeaCache 0.04/start6 | 238.66 | 1.017 | 128.81 | 1.293 | 38.10 | 6,8,11,13,16,18,21,24 | 8/16 |
| Non-HQ 30-step | 0 | TeaCache 0.06/start5 | 286.78 | 0.847 | 114.97 | 1.448 | 33.39 | 5,7,9,12,14,16,19,21,23,26 | 10/15 |
| Non-HQ 30-step | 0 | TeaCache 0.08/start5 | 286.78 | 0.847 | 110.68 | 1.504 | 33.31 | 5,7,9,11,14,16,18,20,23,25,28 | 11/14 |
| HQ 15-step | 1 | KWL baseline | 351.06 | 1.000 | 148.46 | 1.000 | 68.17 | - | - |
| HQ 15-step | 1 | TeaCache 0.04/start6 | 357.29 | 0.983 | 111.85 | 1.327 | 85.70 | 6-13 | 24/27 |
| HQ 15-step | 1 | TeaCache 0.06/start5 | 404.04 | 0.869 | 105.89 | 1.402 | 86.08 | 5-13 | 27/30 |
| HQ 15-step | 1 | TeaCache 0.08/start5 | 404.12 | 0.869 | 107.07 | 1.387 | 88.57 | 5-13 | 27/30 |
| Non-HQ 30-step | 1 | KWL baseline | 267.71 | 1.000 | 165.03 | 1.000 | 32.38 | - | - |
| Non-HQ 30-step | 1 | TeaCache 0.04/start6 | 267.71 | 1.000 | 127.15 | 1.298 | 33.76 | 6,8,11,13,16,18,21,24 | 8/16 |
| Non-HQ 30-step | 1 | TeaCache 0.06/start5 | 276.47 | 0.968 | 117.18 | 1.408 | 33.21 | 5,7,9,12,14,16,19,21,23,26 | 10/15 |
| Non-HQ 30-step | 1 | TeaCache 0.08/start5 | 276.47 | 0.968 | 112.59 | 1.466 | 33.10 | 5,7,9,11,14,16,18,20,23,25,28 | 11/14 |

Average over the two prompts:

| Pipeline | Variant | Avg total x | Avg stage1 x | Interpretation |
|---|---|---:|---:|---|
| HQ 15-step | TeaCache 0.04/start6 | 0.992 | 1.323 | Mildest TeaCache setting; stable stage-1 gain, no end-to-end gain here. |
| HQ 15-step | TeaCache 0.06/start5 | 0.930 | 1.413 | More stage-1 skip, but end-to-end slower because uncached stages dominate. |
| HQ 15-step | TeaCache 0.08/start5 | 0.930 | 1.403 | Similar to 0.06/start5; no extra HQ benefit in this run. |
| Non-HQ 30-step | TeaCache 0.04/start6 | 1.009 | 1.295 | Only setting with neutral/slightly positive end-to-end result. |
| Non-HQ 30-step | TeaCache 0.06/start5 | 0.907 | 1.428 | Better stage-1 gain, worse total due offload/LoRA overhead. |
| Non-HQ 30-step | TeaCache 0.08/start5 | 0.907 | 1.485 | Best stage-1 gain, but not best total in this setup. |


## References

- [TeaCache: Accelerating Diffusion Models with Temporal Similarity](https://arxiv.org/abs/2411.14324)
