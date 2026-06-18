# SANA-Video & Cosmos3-Super — clean single-line acceleration (branch `sana-video`)

State of this branch: each model has ONE clean acceleration line (plus the dense
baseline). All overlapping/experimental paths removed; every toggle is an explicit
flag/env. Speedups below are GB200, warmup-excluded, at each model's official spec.

## SANA-Video  =  EasyCache + fusion
- Cache: **EasyCache** (`--easycache`, calibration-free) — the only step-skip method.
- Fusion: `--compile` (torch.compile) + `--linattn-bf16` + `--qkv-merge`.
- Removed: TeaCache, late-step-skip, TaylorSeer, NVFP4/ffn-lp quant. All flags explicit
  (no env-default switches). 720p LTX-2 VAE latent denorm fixed.

| res  | spec               | denoise | warm (e2e) |
|------|--------------------|---------|------------|
| 480p | 832×480, 81f, 50s  | **2.85×** | 2.61× |
| 720p | 1280×704, 81f, 50s | **2.28×** | 2.14× |
(EasyCache alone ~1.5×; the rest is fusion/compile. EasyCache skips ~16/50 steps.)

## Cosmos3-Super (64B)  =  TeaCache + step-selective NVFP4
- Cache: **TeaCache** (rel-L1 accumulate + threshold) with optional polynomial
  coefficient calibration (`SGLANG_COSMOS3_TEACACHE_COEFFICIENTS`, identity = off).
- Quant: **step-selective NVFP4** — TE fp4 on GEN linears (gate_up/down/qkv/out),
  with first/last denoise steps kept BF16 (`FP4_SKIP_FIRST_STEPS`/`_LAST_STEPS`).
- Removed: PAB, GEN-token pruning, cache-dit, the EasyCache method. (SmoothQuant /
  SVDQuant / PISA0 live on other branches, not here.)
- Official spec: 1280×720, 189f, 24fps, 35 steps, guidance 6.0, flow_shift 10.0, 4×GPU.

| config (TeaCache thr/start/max + NVFP4 step-skip)        | total  | denoise |
|----------------------------------------------------------|--------|---------|
| aggressive: 1.30 / 8 / 4  + NVFP4(first0/last3)          | **2.66×** | 2.91× |
| conservative: 1.15 / 10 / 3 + NVFP4(first3/last3 dense)  | **2.26×** | 2.43× |
| TeaCache-only (no quant), 1.30/8/4                        | 2.18×  | 2.33× |
(NVFP4 adds ~1.22× on top of TeaCache. Aggressive ≈ PSNR 16 dB; lower thr = higher quality.)

Side-by-side demos: HF dataset `yitongl/ltx23-shares` (`sana-video-480p/`,
`sana-video-720p/`, `cosmos3-super-64b/`).
