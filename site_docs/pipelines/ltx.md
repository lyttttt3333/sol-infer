# LTX-2.3

LTX-2.3 is the 22B high-resolution pipeline. The optimized path combines KWL fusion, cache, PISA sparse attention, NVFP4, and token pruning.

## Speed

| Setting | Acceleration line | Speedup |
|---|---|---:|
| Full optimization | KWL fusion + cache + PISA + NVFP4 + token-prune | ~2.4x |

Measured on GB200 with 1088x1920, 241 frames, warmup excluded.

## Launch

```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh baseline
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt
```

`fullopt` is self-contained. Override `MODEL_PATH`, `DISTILLED_LORA`, or `SPATIAL_UPSAMPLER` when weights live outside the default cache location.

## Techniques

- [Kernel fusion](../techniques/kernel.md): KWL fusion reduces DiT operator overhead.
- [Cache](../techniques/cache.md): stage-specific reuse avoids redundant work.
- [Sparse attention](../techniques/sparse.md): PISA targets redundant video self-attention.
- [Quantization](../techniques/quant.md): NVFP4 targets video FFN compute.
- [Token pruning](../techniques/token_prune.md): low-salience tokens are removed in selected refine steps.
