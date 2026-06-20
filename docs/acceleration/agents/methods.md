# Acceleration methods (agents)

Per method: files, knobs (env/flag), defaults, decision rules. Lossy unless noted.
See [../human/methods.md](../human/methods.md) for mechanism.

## 1. Cache (step-skip)
- TeaCache: `runtime/cache/cosmos3_teacache.py`, `runtime/cache/ltx2_teacache.py`
  - env: `SGLANG_COSMOS3_TEACACHE_{ENABLED,THRESH,START,MAX_CONTINUOUS_HITS,COEFFICIENTS}`
    (LTX: `SGLANG_LTX2_TEACACHE_*`). shipped Cosmos: thr 1.15 / start 10 / max 3.
  - rule: lower THRESH = safer/slower; COEFFICIENTS `(1,0)` = uncalibrated.
- EasyCache: `models/dits/sana_video.py`; flags `--easycache <thr> --ec-warmup --ec-subsample`
  (env `SGLANG_SANA_EASYCACHE_{THRESH,WARMUP,SUBSAMPLE}`). calibration-free.
- SCSP stage-1: `runtime/cache/ltx2_stage1_cache_core.py`;
  `SGLANG_LTX2_STAGE1_CACHE_CORE_{ENABLED,PRESET}` (preset `8of15_last_29calls`).
- also provided: PAB `runtime/cache/ltx2_pab.py`; cache-dit `cache_dit_integration.py`
  (DBCache, TaylorSeer) + `ltx2_block_adapter.py`.

## 2. Quantization (NVFP4)
- impl: `models/dits/cosmos3video.py::_maybe_init_fp4_linear`, `models/dits/ltx_2.py`,
  `efficiency/transforms/nvfp4_ffn.py`; kernels `jit_kernel/csrc/gemm/nvfp4`.
- env (Cosmos): `SGLANG_COSMOS3_FP4_LINEAR=1`,
  `SGLANG_COSMOS3_FP4_TARGETS=gate_up,down,qkv,out`,
  `SGLANG_COSMOS3_FP4_SKIP_FIRST_STEPS`, `_SKIP_LAST_STEPS`, `_SKIP_FIRST_LAYERS`, `_SKIP_LAST_LAYERS`.
- env (LTX): `SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1` → `SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=1`.
- requires: Blackwell sm_100+ + `transformer_engine`. else auto BF16 fallback (warn).
  install TE: `scripts/postinstall_cuda_jit.sh --with-te`.
- also provided: FP8 (`fp8.py`,`modelopt_fp8.py`), MXFP4/8 (`mxfp4*.py`,`modelslim_*`),
  Nunchaku/SVDQuant (`nunchaku_linear.py`), ModelOpt (`modelopt_quant.py`, SmoothQuant).
- rule: keep first/last N steps BF16 (quality); SmoothQuant calib for FP4.

## 3. Kernel fusion (KWL) — lossless
- kernels: `jit_kernel/diffusion/{triton,cutedsl}/` (ltx2_adaln, ltx2_qknorm,
  ltx2_dual_modulate, ltx2_ada_values, ltx2_gelu, scale_residual_norm_scale_shift, …).
- wiring: `efficiency/transforms/kwl_fusions.py`; LTX env `SGLANG_HQ_KWL_*` →
  `SGLANG_LTX2_{FUSED_*,SHARE_*,COMPILE_*}`.
- compile mode: `SGLANG_TORCH_COMPILE_MODE` (denoising.py default `max-autotune-no-cudagraphs`).
  - cold + max-autotune in-process = **deadlock** (grouped-conv Triton autotune hangs at cuda.sync).
  - cold-safe: `default`. fast: `max-autotune-no-cudagraphs` + `TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1`
    + persistent `TORCHINDUCTOR_CACHE_DIR`. or exclude conv: `TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN`.

## 4. Sparse attention (PISA)
- impl: `runtime/layers/attention/backends/piecewise_attn.py`.
- select: `--component-attention-backends transformer_2=piecewise_attn` +
  `--attention-backend-config` keys: `piecewise_sparsity`, `piecewise_block_size`,
  `piecewise_route_mode=score|local`, `piecewise_only_video_self_attention`,
  `piecewise_approx_remainder`, `piecewise_stage{1,2}_dense_layers`, `piecewise_dense_fallback`.
- LTX use: stage 2 only, sparsity 0.9, block 64, route score, dense layers 0-1.
- also provided backends: `video_sparse_attn.py`(VSA), `sparse_video_gen_2_attn.py`(SVG2),
  `sliding_tile_attn.py`(STA), `block_sparse_attn.py`, `vmoba.py`, `rain_fusion_attn.py`,
  `laser_attn.py`, `sparse_linear_attn.py`. dense: flash_attn, sdpa, sage_attn(3).

## 5. Token pruning
- impl: `efficiency/techniques/token_prune.py`; wiring `stages/ltx_2_denoising.py`
  (`_ltx2_prune_video_tokens_for_midpoint`); model seam `efficiency/models/ltx2_spec.py`
  (`PRUNABLE_TOKENS`).
- env (LTX): `SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_{RATIO=0.5,METHOD=feat_norm,STEPS=1,2}`.
- compensation `prev`. confined to stage-2 midpoint steps.

## composition rule
exclusive seams: 1 attention backend · 1 FFN precision · 1 token-set owner · 1
cache policy per stage. `runtime/efficiency/compose()` validates structurally; each
lossy method is off==identity when disabled.
