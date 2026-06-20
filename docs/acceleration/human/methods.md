# Acceleration methods (human)

The five building blocks the [pipelines](pipelines.md) are assembled from. Each
section: what it does, how it works, the knobs, and the trade-off.

---

## 1. Cache (step-skip)

**Idea.** Diffusion denoising is iterative; adjacent steps often change the latent
very little. A cache reuses a previous step's transformer output (or residual)
instead of recomputing, skipping the heavy DiT forward on "easy" steps.

Three variants are shipped (different *skip-decision* policies):

- **TeaCache** (`runtime/cache/{cosmos3,ltx2}_teacache.py`) — accumulate a per-step
  relative-L1 distance; recompute when it crosses a `threshold`, otherwise replay the
  cached residual. Knobs: `threshold`, `start_step`, `max_continuous_hits`, optional
  polynomial coefficient calibration. Used by **Cosmos3**.
- **EasyCache** (in `models/dits/sana_video.py`) — calibration-free: estimate the
  relative change on a spatially-subsampled tensor and reuse if below threshold.
  Knobs: `--easycache <thr>`, `--ec-warmup`, `--ec-subsample`. Used by **SANA**.
- **SCSP stage-1 cache-core** (`runtime/cache/ltx2_stage1_cache_core.py`) — a fixed
  step-skip schedule for LTX stage 1 (preset `8of15_last_29calls`). Used by **LTX**.

Also provided (not on a default line): **PAB** (`ltx2_pab.py`), and the **cache-dit**
integration (`cache_dit_integration.py`) exposing **DBCache** and **TaylorSeer**.

**Trade-off.** Pure runtime win, but lossy — too-aggressive skipping drops detail.
TeaCache `thr` is the quality dial (lower = safer).

---

## 2. Quantization (NVFP4)

**Idea.** Run the big linear layers in 4-bit (NVFP4 block-scaling) instead of BF16;
the GEMMs are 2.7–3.6× faster on Blackwell.

**How.** `transformer_engine` `te.Linear` with `NVFP4BlockScaling`; BF16 weights are
copied in and quantized per-forward. Applied **step-selectively**: the first/last few
denoise steps stay BF16 (most quality-sensitive), middle steps run FP4.

- Cosmos3: `SGLANG_COSMOS3_FP4_LINEAR=1`, `FP4_TARGETS=gate_up,down,qkv,out`,
  `FP4_SKIP_FIRST_STEPS`/`_LAST_STEPS`.
- LTX: NVFP4 on the video FFN linears (`efficiency/transforms/nvfp4_ffn.py`).

**Requirements / fallback.** Needs Blackwell (sm_100+) + transformer_engine. On
older GPUs or without TE it **auto-falls back to BF16** with a warning (no crash).

Also provided: **FP8** (`fp8.py`, `modelopt_fp8.py`), **MXFP4/MXFP8**
(`mxfp4*.py`, `modelslim_*`), **Nunchaku / SVDQuant** (`nunchaku_linear.py`),
**ModelOpt** recipes incl. SmoothQuant calibration.

**Trade-off.** Lossy (4-bit). Step-selective + SmoothQuant calibration keep it close
to BF16; adds ~1.2× on top of cache on Cosmos3.

---

## 3. Kernel fusion (KWL)

**Idea.** The DiT spends a lot of time in many small ops around attention/FFN
(RMSNorm, scale/shift, residual gates, RoPE, modulation). Fuse them into few kernels
to cut launch overhead and intermediate read/write. **Algorithm-lossless.**

**How.** Per-fusion Triton / CuTeDSL kernels in `jit_kernel/diffusion/`:
AdaLN (norm+scale+shift+residual-gate), Q/K-norm+split-RoPE, dual modulation,
9-way Ada values, FFN `proj_in+GELU` (fused addmm-activation), audio QKVG concat,
attention gate-to-out compile, tiled-VAE-decoder compile; plus CFG/STG
block-0 / guidance-prefix sharing. Declarative wiring: `efficiency/transforms/kwl_fusions.py`.

**torch.compile caveat.** The full-model `max-autotune` compile path deadlocks on a
cold inductor cache (a grouped-conv Triton autotune hangs at `cuda.synchronize`).
Use inductor `default` mode for cold-safe runs, or warm the cache via subprocess
autotune (`TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1`) for the faster max-autotune path.

**Trade-off.** Lossless (bf16 rounding only), but each fusion is shape-specialized —
only retained where it actually helps the target workload.

---

## 4. Sparse attention (PISA)

**Idea.** Video self-attention over tens of thousands of tokens is quadratic. Sparse
attention restricts each query to a subset of key/value blocks.

**How (PISA / piecewise, `backends/piecewise_attn.py`).** Chunk Q/K/V into blocks;
score block pairs (top-k routing) plus a K-variance proxy; run exact attention on the
selected blocks, and either approximate the rest with block centroids
(`approx_remainder=true`) or drop them. Knobs: `sparsity`, `block_size`,
`route_mode` (score|local), `only_video_self_attention`, per-stage dense layers.
LTX applies it to **stage 2 only** (`transformer_2`), keeping dense layers 0-1.

Also provided: **VSA** (`video_sparse_attn.py`), **SVG2** (`sparse_video_gen_2_attn.py`),
**STA** (`sliding_tile_attn.py`), **block-sparse** (`block_sparse_attn.py`),
**VMoBA** (`vmoba.py`), **RainFusion**, **LASER**, **sparse-linear**.

**Trade-off.** Lossy (algorithm-level approximation). Selected via
`--attention-backend` / `--component-attention-backends`; quality depends on
sparsity + routing.

---

## 5. Token pruning

**Idea.** Not all spatial tokens need full refinement at every step. Drop a fraction
of video tokens during the (less sensitive) middle refine steps, then scatter back.

**How** (`efficiency/techniques/token_prune.py` + `stages/ltx_2_denoising.py`).
At LTX stage-2 midpoint steps (1-2): score video tokens by **feature norm**
(`method=feat_norm`), keep the top `keep_ratio` (default 0.5), run the block stack on
the kept tokens, and **compensate** the dropped ones from the previous step (`prev`).
Wired model-agnostically via the efficiency framework (`PRUNABLE_TOKENS` capability).

**Trade-off.** Lossy. Confined to the refine stage's middle steps to limit quality
impact; verified to keep 31620/63240 tokens at the LTX 1080p shape.

---

## How they compose

A pipeline is a subset of these five (see [pipelines.md](pipelines.md)). They are
**structurally non-interfering** when each owns a distinct seam: one attention
backend, one FFN precision, one token-set owner, one cache policy per stage. The
`runtime/efficiency/` framework encodes this (`compose()` checks exclusive seams);
numeric composition of the lossy ones is bounded by each method's off==identity
invariant plus empirical measurement at the official config.
