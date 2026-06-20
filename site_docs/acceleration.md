# Acceleration methods

The five building blocks the [pipelines](pipelines.md) are assembled from. Each
section: what it does, how it works, the knobs, the trade-off, and a reference
cheat-sheet (files / env / flags).

## 1. Cache (step-skip)

**Idea.** Adjacent denoising steps often change the latent very little; a cache
reuses a previous step's transformer output/residual instead of recomputing.

Three variants (different skip-decision policies):

- **TeaCache** — accumulate a per-step rel-L1 distance; recompute when it crosses a
  `threshold`, else replay the cached residual. Used by **Cosmos3**.
- **EasyCache** — calibration-free: estimate relative change on a subsampled tensor,
  reuse if below threshold. Used by **SANA**.
- **SCSP stage-1 cache-core** — fixed step-skip schedule for LTX stage 1. Used by **LTX**.

Also provided: **PAB**, and **cache-dit** (DBCache, TaylorSeer).

!!! info "Reference"
    - files: `runtime/cache/{cosmos3,ltx2}_teacache.py`, `ltx2_stage1_cache_core.py`,
      `ltx2_pab.py`, `cache_dit_integration.py`; SANA EasyCache in `models/dits/sana_video.py`
    - TeaCache env: `SGLANG_COSMOS3_TEACACHE_{ENABLED,THRESH,START,MAX_CONTINUOUS_HITS,COEFFICIENTS}`
    - EasyCache flags: `--easycache <thr> --ec-warmup --ec-subsample`
    - SCSP env: `SGLANG_LTX2_STAGE1_CACHE_CORE_{ENABLED,PRESET}` (`8of15_last_29calls`)
    - rule: lower threshold = safer/slower. **lossy.**

## 2. Quantization (NVFP4)

**Idea.** Run the big linears in 4-bit (NVFP4 block-scaling); GEMMs are 2.7–3.6×
faster on Blackwell. Applied **step-selectively** — first/last few steps stay BF16.

Also provided: FP8, MXFP4/MXFP8, Nunchaku/SVDQuant, ModelOpt (incl. SmoothQuant).

!!! info "Reference"
    - files: `models/dits/cosmos3video.py::_maybe_init_fp4_linear`, `models/dits/ltx_2.py`,
      `efficiency/transforms/nvfp4_ffn.py`; kernels `jit_kernel/csrc/gemm/nvfp4`
    - Cosmos env: `SGLANG_COSMOS3_FP4_LINEAR=1`, `FP4_TARGETS=gate_up,down,qkv,out`,
      `FP4_SKIP_FIRST_STEPS`, `FP4_SKIP_LAST_STEPS`
    - LTX env: `SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1`
    - requires Blackwell sm_100+ + `transformer_engine` (`postinstall_cuda_jit.sh --with-te`);
      else auto BF16 fallback. **lossy.**

## 3. Kernel fusion (KWL)

**Idea.** Fuse the many small ops around attention/FFN (RMSNorm, scale/shift,
residual gates, RoPE, modulation) into few kernels — cut launch overhead and
intermediate read/write. **Algorithm-lossless.**

Fusions: AdaLN, Q/K-norm+split-RoPE, dual modulation, 9-way Ada values,
FFN `proj_in+GELU`, audio QKVG concat, gate-to-out compile, tiled-VAE compile;
plus CFG/STG block-0 / guidance-prefix sharing.

!!! info "Reference"
    - kernels: `jit_kernel/diffusion/{triton,cutedsl}/` (ltx2_adaln, ltx2_qknorm,
      ltx2_dual_modulate, ltx2_ada_values, …); wiring `efficiency/transforms/kwl_fusions.py`
    - env: `SGLANG_HQ_KWL_*` → `SGLANG_LTX2_{FUSED_*,SHARE_*,COMPILE_*}`
    - compile mode (`SGLANG_TORCH_COMPILE_MODE`): cold + `max-autotune` in-process =
      deadlock; cold-safe = `default`; fast = `max-autotune-no-cudagraphs` +
      `TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1` + persistent `TORCHINDUCTOR_CACHE_DIR`.

## 4. Sparse attention (PISA)

**Idea.** Video self-attention over tens of thousands of tokens is quadratic.
Restrict each query to a subset of K/V blocks via block-level top-k routing +
centroid approximation of the remainder. LTX uses it on **stage 2 only**.

Also provided: VSA, SVG2, STA, block-sparse, VMoBA, RainFusion, LASER, sparse-linear.

!!! info "Reference"
    - file: `runtime/layers/attention/backends/piecewise_attn.py`
    - select: `--component-attention-backends transformer_2=piecewise_attn` +
      `--attention-backend-config piecewise_sparsity=…,piecewise_block_size=…,piecewise_route_mode=score|local,…`
    - LTX use: stage 2, sparsity 0.9, block 64, route score, dense layers 0-1. **lossy.**

## 5. Token pruning

**Idea.** Drop a fraction of video tokens during the (less sensitive) middle refine
steps, then scatter back — fewer tokens through the heavy block stack.

!!! info "Reference"
    - files: `efficiency/techniques/token_prune.py`, `stages/ltx_2_denoising.py`,
      seam `efficiency/models/ltx2_spec.py` (`PRUNABLE_TOKENS`)
    - LTX env: `SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_{RATIO=0.5,METHOD=feat_norm,STEPS=1,2}`
    - compensation `prev`; confined to stage-2 midpoint steps. **lossy.**

## How they compose

A [pipeline](pipelines.md) is a subset of these five. They are structurally
non-interfering when each owns a distinct seam: one attention backend, one FFN
precision, one token-set owner, one cache policy per stage. The
`runtime/efficiency/` framework validates this (`compose()`); each lossy method is
off==identity when disabled.
