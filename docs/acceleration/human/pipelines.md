# Optimization pipelines (human)

Three video models, each reduced to **one clean acceleration line** = a specific
assembly of the [five methods](methods.md). Every model also has a dense
`baseline` for reference. Numbers are GB200, warmup-excluded, official spec.

---

## 1. SANA-Video — *EasyCache + fusion + compile*

The lightest model (2B, single GPU). The acceleration line is **calibration-free
cache + lossless fusion + graph compile** — no quantization, no sparse attention.

**Stack**
1. **EasyCache** — calibration-free adaptive step-skip; reuses a step's output when
   the relative change from the previous step is small.
2. **Fusion** — `linattn-bf16` (bf16 linear-attention KV aggregation) + `qkv-merge`
   (one merged QKV GEMM). Both lossless.
3. **torch.compile** — inductor graph compile.

**Design notes**
- Compile mode is the subtle part. The generic default is `max-autotune`, whose
  in-process GEMM/conv autotune **deadlocks** at `cuda.synchronize()` on a *cold*
  inductor cache (a grouped-conv Triton template hangs). So:
  - default `--compile` → safe inductor `default` mode (~2.10x, runs cold anywhere);
  - `--max-autotune` → the fast path (~2.56x once warm): subprocess autotune (skips
    the hanging conv) + a persistent inductor cache (first cold run warms it).
- 480p uses Wan2.1-VAE, 720p uses LTX-2-VAE (different latent denorm).

**Entry**: `scripts/sana/sana_video_sglang_run.py` (1 GPU)

**Measured (480p, 832×480, 81f, 50 steps)**
| config | warm | speedup |
|---|---|---|
| baseline (dense) | 28.5 s | 1.00× |
| EasyCache + fusion + compile (`default`) | 13.5 s | **2.10×** |
| + `--max-autotune` (warm cache) | 11.0 s | **2.56×** |

---

## 2. Cosmos3-Super 64B — *TeaCache + step-selective NVFP4*

The largest model (64B, 4-GPU sequence parallel). The line is **cache + low-precision
quant**; attention stays dense.

**Stack**
1. **TeaCache** — rel-L1 accumulate + threshold step-skip, with optional polynomial
   coefficient calibration. Shipped config: `thr 1.15 / start 10 / max-continuous 3`.
2. **Step-selective NVFP4** — TransformerEngine 4-bit on the GEN-layer linears
   (gate_up / down / qkv / out), with the **first 3 and last 3 denoise steps kept
   BF16** (the most quality-sensitive). Needs Blackwell + transformer_engine; on
   older GPUs it gracefully falls back to BF16.

**Design notes**
- Official spec: 1280×720, 189f, 24fps, 35 steps, guidance 6.0, flow_shift 10.0,
  max_seq 4096, 4 GPUs. Prompts are **structured-JSON** (a bare sentence gives poor
  quality); the official robot-plate prompt is shipped at `prompts/cosmos/robot_plate.json`
  and must be passed explicitly via `PROMPT_FILE`.
- The entry collapses to `baseline | fullopt` (no variant sweep).

**Entry**: `scripts/cosmos/slurm_cosmos3_super.sh [baseline|fullopt]` (4 GPU)

**Measured (1280×720, 189f, 35 steps)**
| config | warm | speedup |
|---|---|---|
| baseline (dense) | 97.2 s | 1.00× |
| fullopt: TeaCache 1.15/10/3 + NVFP4(first3/last3 dense) | 43.1 s | **~2.26×** |

---

## 3. LTX-2.3 1080p/10s — *the full five-method stack*

The HQ two-stage pipeline (1 GPU). This line uses **all five methods at once** and
is the reference for how they compose.

**Stack (= `fullopt`)**
1. **KWL kernel fusion** — ~13 lossless AdaLN / qknorm+RoPE / dual-modulate /
   ada-values / residual-gate / FFN-proj+GELU / gate-to-out / audio-QKVG fusions +
   tiled-VAE compile.
2. **Stage-1 SCSP cache** — step-skip on stage-1 denoise (`8of15_last_29calls`).
3. **Stage-2 PISA sparse attention** — piecewise sparse attn on stage-2
   (`transformer_2`), dense layers 0-1.
4. **NVFP4 video FFN** — TE 4-bit on the video FFN linears.
5. **Stage-2 midpoint token-prune** — drop ~50% of video tokens at refine steps 1-2
   (feat-norm scoring, prev-step compensation).

**Design notes**
- Official two-stage HQ: 1088×1920, 241f, 24fps, stage-1 30 steps + stage-2 3-sigma
  refine, distilled LoRA merged for stage 2, guidance 3.0.
- `WARMUP=true` by default so the one-time compile cost is excluded from timing.
- Self-contained: `fullopt` bakes the whole recipe, no extra env needed.

**Entry**: `scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh [baseline|fullopt]` (1 GPU)

**Measured (1088×1920, 241f)**
| config | warm | speedup |
|---|---|---|
| baseline (dense two-stage) | 95.7 s | 1.00× |
| fullopt (5-method stack) | 39.2 s | **~2.4×** |

---

## At a glance

| Model | Cache | Quant | Kernel | Sparse | Token-prune | speedup |
|---|---|---|---|---|---|---|
| SANA-Video | EasyCache | — | fusion+compile | — | — | 2.1–2.56× |
| Cosmos3-Super | TeaCache | NVFP4 | — | — | — | ~2.26× |
| LTX-2.3 | SCSP | NVFP4 | KWL | PISA | yes | ~2.4× |
