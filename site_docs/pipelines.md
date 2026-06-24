# Optimization pipelines

Each model has one dense `baseline` and one `fullopt` recipe. Numbers below are
warmup-excluded and measured at the model's target resolution.

## Cosmos3-Super 64B: TeaCache + NVFP4

Cosmos3-Super is the 4xB200 path. The optimized line keeps attention dense and
combines cache reuse with 4-bit TransformerEngine linears on the middle denoising
region.

**Stack**

1. **TeaCache**: relative-L1 accumulation with threshold, start-step, and
   max-continuous-hit controls.
2. **NVFP4**: TransformerEngine 4-bit on `gate_up`, `down`, `qkv`, and `out`
   linears, with dense boundary steps.

**Entry**

```bash
MODEL_REPO=nvidia/Cosmos3-Super \
ROOT=out/cosmos3 PROMPT_FILE=prompt.txt PROMPT_TAG=demo \
bash scripts/cosmos/slurm_cosmos3_super.sh baseline   # or: fullopt
```

| config | warm | speedup |
|---|---:|---:|
| baseline | 97.2 s | 1.00x |
| fullopt: TeaCache 1.15/start10/max3 + NVFP4 | 43.1 s | ~2.26x |

## LTX-2.3 1080p/10s: kernel fusion + cache + PISA + NVFP4 + token-prune

LTX-2.3 is the 1xB200 two-stage HQ path. The optimized line composes all five
methods on the stage boundaries where each method owns a distinct part of the
pipeline.

**Stack**

1. **Kernel fusion**: AdaLN, QK-norm+RoPE, dual modulation, Ada values,
   FFN projection+GELU, gate-to-out, audio QKVG, and tiled-VAE compile.
2. **Stage-1 cache**: fixed-schedule cache on stage-1 denoising.
3. **Stage-2 PISA sparse attention**: piecewise sparse attention on
   `transformer_2`, with dense early layers.
4. **NVFP4 video FFN**: TransformerEngine 4-bit on video FFN linears.
5. **Stage-2 token-prune**: drop low-salience video tokens at midpoint refine
   steps and scatter results back.

**Entry**

```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh baseline
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt
```

| config | warm | speedup |
|---|---:|---:|
| baseline | 95.7 s | 1.00x |
| fullopt | 39.2 s | ~2.4x |

## SANA-Video: EasyCache + fusion + compile

SANA-Video is the 1xB200 480p path. The optimized line uses cache reuse, fused
linear-attention work, merged QKV projection, and `torch.compile`.

**Stack**

1. **EasyCache**: calibration-free adaptive cache reuse from a subsampled latent
   change estimate.
2. **Fusion**: bf16 linear-attention KV aggregation and one merged QKV GEMM.
3. **torch.compile**: graph compile for the hot denoising path.

**Entry**

```bash
PY=.conda/ltx23/bin/python
$PY scripts/sana/sana_video_sglang_run.py \
    --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
    --prompt "a corgi running on the beach" --output out/sana_fullopt \
    --easycache 0.1 --linattn-bf16 --qkv-merge --compile
```

| config | warm | speedup |
|---|---:|---:|
| baseline | 29.4 s | 1.00x |
| fullopt: EasyCache + fusion + compile | 10.6 s | 2.77x |

## At a glance

| Model | Cache | Quant | Kernel | Sparse | Token-prune | Speedup |
|---|---|---|---|---|---|---:|
| Cosmos3-Super | TeaCache | NVFP4 | - | - | - | ~2.26x |
| LTX-2.3 | stage-1 cache | NVFP4 | fusion | PISA | yes | ~2.4x |
| SANA-Video | EasyCache | - | fusion+compile | - | - | 2.77x |
