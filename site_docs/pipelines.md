# Optimization pipelines

Three video models, each reduced to **one clean acceleration line** = a specific
assembly of the [five methods](acceleration.md). Every model also has a dense
`baseline`. Numbers are GB200, warmup-excluded, official spec.

## SANA-Video — EasyCache + fusion + compile

The lightest model (2B, single GPU): **calibration-free cache + lossless fusion +
graph compile**. No quantization, no sparse attention.

**Stack**

1. **EasyCache** — calibration-free adaptive step-skip; reuse a step's output when
   the relative change from the previous step is small.
2. **Fusion** — `linattn-bf16` (bf16 linear-attention KV aggregation) + `qkv-merge`
   (one merged QKV GEMM). Both lossless.
3. **torch.compile** — inductor graph compile.

!!! warning "Compile mode"
    The generic default `max-autotune` **deadlocks** at `cuda.synchronize()` on a
    *cold* inductor cache (a grouped-conv Triton template hangs). So:

    - default `--compile` → safe inductor `default` mode (~2.10×, runs cold anywhere);
    - `--max-autotune` → fast path (~2.56× once warm): subprocess autotune (skips the
      hanging conv) + a persistent inductor cache (first cold run warms it).

**Entry**: `scripts/sana/sana_video_sglang_run.py` (1 GPU)

| config (480p, 832×480, 81f, 50 steps) | warm | speedup |
|---|---|---|
| baseline (dense) | 28.5 s | 1.00× |
| EasyCache + fusion + compile (`default`) | 13.5 s | **2.10×** |
| + `--max-autotune` (warm cache) | 11.0 s | **2.56×** |

## Cosmos3-Super 64B — TeaCache + step-selective NVFP4

The largest model (64B, 4-GPU sequence parallel): **cache + low-precision quant**;
attention stays dense.

**Stack**

1. **TeaCache** — rel-L1 accumulate + threshold step-skip, optional polynomial
   coefficient calibration. Shipped: `thr 1.15 / start 10 / max-continuous 3`.
2. **Step-selective NVFP4** — TransformerEngine 4-bit on the GEN-layer linears
   (gate_up / down / qkv / out), **first 3 + last 3 denoise steps kept BF16**.
   Needs Blackwell + transformer_engine; older GPUs gracefully fall back to BF16.

!!! note "Prompts"
    Cosmos3 uses **structured-JSON** prompts (a bare sentence gives poor quality).
    The official robot-plate prompt ships at `prompts/cosmos/robot_plate.json` and
    must be passed explicitly via `PROMPT_FILE`.

**Entry**: `scripts/cosmos/slurm_cosmos3_super.sh [baseline|fullopt]` (4 GPU,
1280×720, 189f, 35 steps, guidance 6.0, flow_shift 10.0, max_seq 4096)

| config | warm | speedup |
|---|---|---|
| baseline (dense) | 97.2 s | 1.00× |
| fullopt: TeaCache 1.15/10/3 + NVFP4(first3/last3 dense) | 43.1 s | **~2.26×** |

## LTX-2.3 1080p/10s — the full five-method stack

The HQ two-stage pipeline (1 GPU). This line uses **all five methods at once**.

**Stack (`fullopt`)**

1. **KWL kernel fusion** — ~13 lossless AdaLN / qknorm+RoPE / dual-modulate /
   ada-values / residual-gate / FFN-proj+GELU / gate-to-out / audio-QKVG fusions +
   tiled-VAE compile.
2. **Stage-1 SCSP cache** — step-skip on stage-1 denoise (`8of15_last_29calls`).
3. **Stage-2 PISA sparse attention** — piecewise sparse attn on `transformer_2`,
   dense layers 0-1.
4. **NVFP4 video FFN** — TE 4-bit on the video FFN linears.
5. **Stage-2 midpoint token-prune** — drop ~50% of video tokens at refine steps 1-2
   (feat-norm scoring, prev-step compensation).

**Entry**: `scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh [baseline|fullopt]` (1 GPU,
1088×1920, 241f, stage-1 30 steps + stage-2 3-sigma, guidance 3.0, distilled LoRA;
`WARMUP=true` by default; `fullopt` is self-contained — no extra env).

| config | warm | speedup |
|---|---|---|
| baseline (dense two-stage) | 95.7 s | 1.00× |
| fullopt (5-method stack) | 39.2 s | **~2.4×** |

## At a glance

| Model | Cache | Quant | Kernel | Sparse | Token-prune | speedup |
|---|---|---|---|---|---|---|
| SANA-Video | EasyCache | — | fusion+compile | — | — | 2.1–2.56× |
| Cosmos3-Super | TeaCache | NVFP4 | — | — | — | ~2.26× |
| LTX-2.3 | SCSP | NVFP4 | KWL | PISA | yes | ~2.4× |
