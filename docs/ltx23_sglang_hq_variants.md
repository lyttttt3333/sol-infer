# LTX-2.3 SGLang HQ Variants

This repo provides two SGLang launch paths aligned to the official LTX-2.3 HQ
two-stage pipeline semantics:

- `sglang+hq`: `scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh`
- `sglang+hq+kwl`: `scripts/run_ltx23_sglang_hq_kwl_1080p10s.sh`

Both use `LTX2TwoStageHQPipeline`, not the older `LTX2TwoStagePipeline`.
The HQ pipeline fixes the semantic settings to the official HQ path:

- Stage 1: half-resolution generation, `15` Res2S steps.
- Stage 2: full-resolution refinement with sigmas
  `[0.909375, 0.725, 0.421875, 0.0]`, i.e. `3` denoising steps.
- Distilled LoRA strength: stage 1 `0.25`, stage 2 `0.5`.
- Video guider: CFG `3.0`, STG `0.0`, rescale `0.45`.
- Audio guider: CFG `7.0`, STG `0.0`, rescale `1.0`.
- Default prompt geometry: `1920x1088`, `241` frames, `24` fps, seed `42`.
- Official negative prompt is passed explicitly by the runner.

The runner passes the official local `1.1` distilled LoRA by default:

```bash
outputs/LTX-2.3-official-files/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
```

and the local spatial upsampler:

```bash
outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

`sglang+hq` explicitly disables KWL fusions, sparse attention, NVFP4/FP4, and
prefix/share shortcuts. `sglang+hq+kwl` keeps the same scheduler, noise,
guidance, LoRA, two-stage topology, prompt handling, and checkpoint paths, and
only enables operator-level fused/compiled kernels:

- fused QK RoPE
- fused AdaLN/RMS AdaLN
- fused QK norm + RoPE
- fused dual modulation
- fused Ada values
- fused residual gate
- fused FFN `proj_in + bias + GELU`
- compiled/fused gate-to-output path
- fused audio QKV/gate path
- optional tiled VAE decoder compile, controlled by
  `SGLANG_HQ_KWL_COMPILE_TILED_VAE` and enabled by default

These KWL switches are intended to change only kernel/operator numerics. Lossy
paths such as sparse attention and NVFP4 remain disabled in both scripts.

Run examples:

```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh
bash scripts/run_ltx23_sglang_hq_kwl_1080p10s.sh
```

Dry-run examples that validate paths and write `run_command.txt` without running
inference:

```bash
DRY_RUN=1 bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh
DRY_RUN=1 bash scripts/run_ltx23_sglang_hq_kwl_1080p10s.sh
```

Slurm examples:

```bash
sbatch scripts/slurm_ltx23_sglang_hq_1080p10s.sh
sbatch scripts/slurm_ltx23_sglang_hq_kwl_1080p10s.sh
```


## Sparse And Cache Variants

The unified runner also supports optimized lossy variants on top of KWL:

```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh kwl_sparse
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh kwl_cache
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh kwl_sparse_cache
```

Variant meanings:

- `dense`: faithful SGLang HQ path with KWL, sparse, cache, and FP4 disabled.
- `kwl`: kernel-wise lossless fused/operator optimized HQ path.
- `kwl_sparse`: KWL plus `piecewise_attn` sparse video self-attention.
- `kwl_cache`: KWL plus LTX-2 PAB attention-output cache.
- `kwl_sparse_cache`: KWL plus sparse attention plus PAB cache.

Sparse attention follows the `ltx-stage1-sparse-schedule` / `ltx-sparse-attn-bringup`
setting style:

- backend: `piecewise_attn` for `transformer` and `transformer_2`
- block size: `64`
- only video self-attention is approximated; other attention falls back dense
- stage 1 schedule: first `3` steps dense, then sparsity ramps from `0.8` to `0.9`
- final sparsity: `0.9` (`density=0.1`)
- layer selective guard: layer `0` remains dense by default via `piecewise_dense_layers=0`

PAB cache follows the cache branch README mechanism: attention module outputs are
cached and replayed with window `3`, A/V latent cross-attention reuse is disabled,
and `A2V` / `V2A` windows are `1`. For the HQ 1080p path, full-resolution stage 2
PAB is disabled by default because stage 2 attention outputs are too large for
single-card resident inference and caused OOM. The validated optimized default is:

```bash
SGLANG_LTX2_PAB_ENABLED=1
SGLANG_LTX2_PAB_SPATIAL_WINDOW=3
SGLANG_LTX2_PAB_TEMPORAL_WINDOW=3
SGLANG_LTX2_PAB_CROSS_WINDOW=3
SGLANG_LTX2_PAB_START_STEP=6
SGLANG_LTX2_PAB_END_STEP=-1
SGLANG_LTX2_PAB_STAGE2_ENABLED=0
SGLANG_LTX2_PAB_DISABLE_AUDIO_VIDEO_CROSS=1
SGLANG_LTX2_PAB_A2V_WINDOW=1
SGLANG_LTX2_PAB_V2A_WINDOW=1
```

The README-conservative cache point remains reproducible by overriding
`SGLANG_LTX2_PAB_START_STEP=12`.

The benchmark matrix script runs all four KWL-family variants in parallel:

```bash
sbatch scripts/slurm_ltx23_sglang_hq_kwl_sparse_cache_matrix_1080p10s.sh
```

Latest validated 1080p 10s matrix, prompt `antique brass clockwork train`,
`241` frames, seed `42`, warmup excluded from request runtime:

| Variant | Total s | Denoise s | Total speedup vs KWL | Denoise speedup vs KWL | Notes |
|---|---:|---:|---:|---:|---|
| `kwl` | 69.120 | 63.646 | ~1.000x | ~1.000x | lossless KWL baseline for this matrix |
| `kwl_cache` | 60.598 | 55.075 | ~1.141x | ~1.156x | PAB start=6, stage2 PAB off |
| `kwl_sparse` | 61.405 | 56.008 | ~1.126x | ~1.136x | sparse only |
| `kwl_sparse_cache` | 53.778 | 48.004 | ~1.285x | ~1.326x | best current combined setting |

Artifacts:

```bash
outputs/ltx23-sglang-hq-kwl-sparse-cache-matrix-pab6-stage2off-1080p10s/benchmark_summary.json
outputs/ltx23-sglang-hq-kwl-sparse-cache-matrix-pab6-stage2off-1080p10s/kwl-vs-kwl-sparse-cache-side-by-side.mp4
```

For comparison, the README-conservative `SGLANG_LTX2_PAB_START_STEP=12` point ran
successfully but added almost no speed on top of sparse for the 15-step HQ path:
`kwl_sparse_cache` total `59.512s` versus `kwl_sparse` total `59.527s`.

Default outputs:

```bash
outputs/ltx23-sglang-hq-1080p10s/dense/out.mp4
outputs/ltx23-sglang-hq-1080p10s/dense/perf.json
outputs/ltx23-sglang-hq-1080p10s/kwl/out.mp4
outputs/ltx23-sglang-hq-1080p10s/kwl/perf.json
```
