# Acceleration methods

Video diffusion inference exposes redundancy at three complementary levels. At
the **algorithm level**, adjacent denoising steps run structurally similar
computations over slowly changing latents, so cache can reuse prior work. At the
**model level**, long spatiotemporal sequences contain redundant tokens and
attention interactions, motivating sparse attention and token pruning. At the
**kernel level**, DiT blocks repeatedly launch memory-bound work around GEMMs,
layout movement, normalization, activation, and precision conversion, which
quantization and fusion reduce.

## Cache

Cache methods reuse a previous transformer output or residual when a denoising
step is close enough to recent steps.

- **TeaCache**: `python/sglang/multimodal_gen/runtime/cache/teacache.py`,
  `cosmos3_teacache.py`, and `ltx2_teacache.py`.
- **EasyCache**: implemented in the SANA-Video model path.
- **Stage-1 cache**: `python/sglang/multimodal_gen/runtime/cache/ltx2_stage1_cache_core.py`.
- **Additional cache hooks**: `ltx2_pab.py` and `cache_dit_integration.py`.

## Quantization

NVFP4 runs selected large linears in TransformerEngine 4-bit format on Blackwell
GPUs, with BF16 fallback when the runtime cannot enable the FP4 path.

- Cosmos3 linears: `python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py`.
- LTX video FFN: `python/sglang/multimodal_gen/runtime/efficiency/transforms/nvfp4_ffn.py`.
- Offline tools: `python/sglang/multimodal_gen/tools/quantize_ltx2_merged_transformer_nvfp4.py`,
  `quantize_ltx2_selective_nvfp4_transformer.py`, and
  `build_modelopt_nvfp4_transformer.py`.

## Kernel fusion

Fusion combines small DiT operations around attention, modulation, gates, FFN
projection, and VAE decode into fewer kernels.

- Wiring: `python/sglang/multimodal_gen/runtime/efficiency/transforms/kwl_fusions.py`.
- Kernels: `python/sglang/jit_kernel/diffusion/`.
- SANA launch flags: `--linattn-bf16`, `--qkv-merge`, and `--compile`.

## Sparse attention

Sparse attention restricts each video self-attention query to selected key/value
blocks.

Actual implementation files:

- `python/sglang/multimodal_gen/runtime/efficiency/transforms/sparse_attention.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/sparse_video_gen_2_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/sparse_linear_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/block_sparse_attn.py`

## Token pruning

Token pruning scores video tokens at selected refine steps, keeps the high-salience
subset through the heavy block stack, and scatters the result back.

Actual implementation files:

- `python/sglang/multimodal_gen/runtime/efficiency/techniques/token_prune.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`
- `python/sglang/multimodal_gen/runtime/efficiency/models/ltx2_spec.py`

## Composition

The `runtime/efficiency/` framework composes these methods by assigning each one a
separate pipeline surface: cache policy, attention backend, FFN precision, kernel
transform, or token-set owner.
