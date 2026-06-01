# LTX-2.3 HQ 1080p 10s 46s Single-GPU Run

This note documents the current fastest LTX-2.3 HQ single-GPU 1080p 10s run in this repo.
The measured run is the SGLang HQ pipeline with KWL kernels, stage-1 cache-core,
stage-2 piecewise sparse video self-attention, and Transformer Engine NVFP4 video FFN linears.

## Measured Result

Artifact directory:

```bash
outputs/ltx23-hq-best-plus-nvfp4-1080p10s/allkwl_cache8_pisa_preproj_ropecache_te_nvfp4
```

Measured `perf.json`:

```text
total: 45.871 s
stage 1 denoise: 25.039 s
stage 2 refinement: 14.889 s
VAE decode: 5.768 s
text encoder: 0.085 s
text connector: 0.034 s
```

The measurement is the request/runtime duration emitted by `--perf-dump-path`. It excludes
Slurm queue time and model/server startup. The run used one visible GPU (`CUDA_VISIBLE_DEVICES=0`)
on a B200-class node.

Main artifacts:

```bash
outputs/ltx23-hq-best-plus-nvfp4-1080p10s/allkwl_cache8_pisa_preproj_ropecache_te_nvfp4/out.mp4
outputs/ltx23-hq-best-plus-nvfp4-1080p10s/allkwl_cache8_pisa_preproj_ropecache_te_nvfp4/perf.json
outputs/ltx23-hq-best-plus-nvfp4-1080p10s/allkwl_cache8_pisa_preproj_ropecache_te_nvfp4/hq_semantics.json
outputs/ltx23-hq-best-plus-nvfp4-1080p10s/bf16_vs_te_nvfp4_side_by_side.mp4
```

## Launch

Run from the repo root:

```bash
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
```

Direct single-GPU launch inside an allocated GPU shell:

```bash
export CUDA_VISIBLE_DEVICES=0
export ROOT=outputs/ltx23-hq-best-plus-nvfp4-1080p10s
export OUT_DIR=$ROOT/allkwl_cache8_pisa_preproj_ropecache_te_nvfp4
export FORCE=1
export WARMUP=true
export WARMUP_STEPS=15
export MASTER_PORT=30017

export SGLANG_HQ_VARIANT=kwl_stage1_cache_core_stage2_sparse
export SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1
export SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET=8of15_last_29calls

export SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN=1
export SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX=1
export SGLANG_HQ_KWL_FUSED_QK_ROPE=1
export SGLANG_HQ_KWL_FUSED_RMS_ADALN=1
export SGLANG_HQ_KWL_FUSED_ADALN=1
export SGLANG_HQ_KWL_FUSED_QKNORM_ROPE=1
export SGLANG_HQ_KWL_FUSED_DUAL_MODULATE=1
export SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL=1
export SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE=1
export SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU=1
export SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT=1
export SGLANG_HQ_KWL_FUSED_AUDIO_QKVG=1
export SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE=1
export SGLANG_HQ_KWL_COMPILE_TILED_VAE=1

export SGLANG_LTX2_PREPROJECT_PROMPTS=1
export SGLANG_LTX2_CACHE_ROPE_EMB=1

bash scripts/run_ltx23_sglang_hq_1080p10s.sh "$SGLANG_HQ_VARIANT"
```

Equivalent Slurm submission:

```bash
sbatch -A nvr_elm_llm -p batch -N 1 --gpus-per-node=1 --exclusive \
  --cpus-per-task=16 --mem=0 -t 03:00:00 -J ltx23-hq-46s \
  -o outputs/slurm/ltx23-hq-46s-%j.out \
  -e outputs/slurm/ltx23-hq-46s-%j.err \
  --wrap 'cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer && \
    CUDA_VISIBLE_DEVICES=0 \
    ROOT=outputs/ltx23-hq-best-plus-nvfp4-1080p10s \
    OUT_DIR=outputs/ltx23-hq-best-plus-nvfp4-1080p10s/allkwl_cache8_pisa_preproj_ropecache_te_nvfp4 \
    FORCE=1 WARMUP=true WARMUP_STEPS=15 MASTER_PORT=30017 \
    SGLANG_HQ_VARIANT=kwl_stage1_cache_core_stage2_sparse \
    SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1 \
    SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET=8of15_last_29calls \
    SGLANG_HQ_KWL_SHARE_BLOCK0_SELF_ATTN=1 \
    SGLANG_HQ_KWL_SHARE_GUIDANCE_PREFIX=1 \
    SGLANG_HQ_KWL_FUSED_QK_ROPE=1 \
    SGLANG_HQ_KWL_FUSED_RMS_ADALN=1 \
    SGLANG_HQ_KWL_FUSED_ADALN=1 \
    SGLANG_HQ_KWL_FUSED_QKNORM_ROPE=1 \
    SGLANG_HQ_KWL_FUSED_DUAL_MODULATE=1 \
    SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL=1 \
    SGLANG_HQ_KWL_FUSED_RESIDUAL_GATE=1 \
    SGLANG_HQ_KWL_FUSED_FFN_PROJ_IN_GELU=1 \
    SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT=1 \
    SGLANG_HQ_KWL_FUSED_AUDIO_QKVG=1 \
    SGLANG_HQ_KWL_ENABLE_FUSED_QKNORM_ROPE=1 \
    SGLANG_HQ_KWL_COMPILE_TILED_VAE=1 \
    SGLANG_LTX2_PREPROJECT_PROMPTS=1 \
    SGLANG_LTX2_CACHE_ROPE_EMB=1 \
    bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl_stage1_cache_core_stage2_sparse'
```

The runner writes the exact generate command to:

```bash
$OUT_DIR/run_command.txt
```

## Model Assets

The measured run used local repo-owned assets under `outputs/`:

```text
model path: outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493
spatial upsampler: outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
distilled LoRA: outputs/LTX-2.3-official-files/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
```

The runner keeps all Hugging Face, Triton, TorchInductor, CUDA, and temporary caches under
`outputs/.cache` or `outputs/.tmp` so repeated runs can reuse local cache state.

## Inference Semantics

Pipeline:

```text
LTX2TwoStageHQPipeline
height=1088 width=1920 num_frames=241 fps=24 seed=42
stage 1: 15 steps, res2s sampler, LoRA strength 0.25
stage 2: 3 refinement steps, sigmas [0.909375, 0.725, 0.421875, 0.0], LoRA strength 0.5
video CFG: 3.0, video rescale: 0.45
audio CFG: 7.0, audio rescale: 1.0
negative prompt: official long HQ negative prompt from scripts/run_ltx23_sglang_hq_1080p10s.sh
```

Stage 1 runs the multi-step base HQ path. Stage 2 uses the official distilled LoRA refinement
path through `--component-paths.distilled_lora`.

## Optimization Stack

### KWL kernel/runtime flags

These are kernel-wise or scheduling optimizations controlled through the `SGLANG_HQ_KWL_*`
variables in the launch command:

```text
share block-0 self attention
share guidance prefix
fused QK RoPE
fused RMS/AdaLN and AdaLN paths
fused QKNorm + RoPE
fused dual modulation
fused Ada values
fused residual gate
fused FFN proj_in + GELU
compiled gate-to-output path
fused audio QKVG
compiled tiled VAE decode
prompt preprojection
RoPE embedding cache
```

Primary code:

```text
python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py
python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py
python/sglang/multimodal_gen/runtime/pipelines_core/stages/text_connector.py
python/sglang/multimodal_gen/runtime/pipelines_core/stages/upsampling.py
```

### Stage-1 cache-core

The 46s recipe enables `SGLANG_LTX2_STAGE1_CACHE_CORE_ENABLED=1` through the
`kwl_stage1_cache_core_stage2_sparse` variant and uses:

```text
SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET=8of15_last_29calls
```

This is an algorithmic approximation. It skips/cache-hits selected stage-1 DiT calls while keeping
the HQ two-stage control flow intact.

Primary code:

```text
python/sglang/multimodal_gen/runtime/cache/ltx2_stage1_cache_core.py
python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py
```

### Stage-2 piecewise sparse attention

The `kwl_stage1_cache_core_stage2_sparse` variant routes stage 2 transformer attention through:

```text
--component-attention-backends transformer=fa,transformer_2=piecewise_attn
piecewise_sparsity=0.9
piecewise_block_size=64
piecewise_only_video_self_attention=true
piecewise_stage1_schedule=false
piecewise_dense_layers=none
piecewise_stage1_dense_layers=none
piecewise_stage2_dense_layers=none
piecewise_approx_remainder=true
piecewise_route_mode=score
piecewise_dense_fallback=fa
```

Only video self-attention is replaced by piecewise sparse attention. Non-video self-attention and
cross-attention fall back to dense FA/SDPA paths. This is an algorithmic approximation.

Primary code:

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py
python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py
```

### Transformer Engine NVFP4 video FFN

The 46s recipe enables TE NVFP4 only for video FFN projections:

```text
SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1
SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=1
SGLANG_LTX2_TE_NVFP4_DISABLE_RHT=1
SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1
SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION=1
```

Affected layers:

```text
transformer_blocks.*.ff.proj_in
transformer_blocks.*.ff.proj_out
```

This is low precision and therefore algorithmically lossy. It does not replace attention QKV or
cross-attention projections in this recipe.

Primary code:

```text
python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py
```

## Lossless vs Lossy Parts

Lossless or intended kernel/runtime-only parts:

```text
KWL fusion flags
prompt preprojection
RoPE cache
resident two-stage execution
VAE tiling compile
```

Algorithmic or low-precision parts in the 46s recipe:

```text
stage-1 cache-core preset 8of15_last_29calls
stage-2 piecewise sparse video self-attention at 90% sparsity
TE NVFP4 video FFN proj_in/proj_out
```

Therefore the 46s number is not a purely lossless KWL result. For a stricter comparison, run
`SGLANG_HQ_VARIANT=kwl` without cache, sparse attention, or TE NVFP4.


## 45s-Aligned Kernel-Only Follow-up

After restoring the accepted 45s HQ boundary, commit `66c3a010e` exposes additional HQ KWL
kernel switches without changing their defaults. The baseline remains unchanged unless the new
switches are explicitly set.

Same-node comparison, warmup excluded, same prompt/seed, same stage-1 cache preset and same
stage-2 PISA config:

```text
artifact root: outputs/ltx23-hq-45aligned-kernelopts-1080p10s
baseline_45aligned total: 46.600 s
  stage1: 25.868 s
  stage2: 14.983 s
  decode: 5.479 s
kernel_only_opts total: 45.083 s
  stage1: 25.120 s
  stage2: 14.388 s
  decode: 5.345 s
speedup vs same-run baseline: 1.034x
```

The aligned invariants were verified in both `hq_semantics.json` files:

```text
stage1_cache_core_preset=8of15_last_29calls
component_attention_backends=transformer=fa,transformer_2=piecewise_attn
piecewise_sparsity=0.9
piecewise_block_size=64
piecewise_approx_remainder=true
piecewise_route_mode=score
```

Extra switches requested for `kernel_only_opts`:

```text
SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE=1
SGLANG_HQ_KWL_COMPILE_GATE_TO_OUT_RESIDUAL=1
SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU=1
SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE=1
```

Runtime caveat: `TE NVFP4 fused proj_in+GELU` fell back at runtime with a cuBLAS unsupported-parameter
error, so the measured `1.034x` should not be attributed to that path. Further isolated ablation is
needed to split the gain between CA dual modulation, gate-to-out residual compile, TE proj-out epilogue,
and normal run-to-run variance.

Artifacts:

```bash
outputs/ltx23-hq-45aligned-kernelopts-1080p10s/summary_kernel_only_opts.json
outputs/ltx23-hq-45aligned-kernelopts-1080p10s/baseline_45aligned/out.mp4
outputs/ltx23-hq-45aligned-kernelopts-1080p10s/kernel_only_opts/out.mp4
outputs/ltx23-hq-45aligned-kernelopts-1080p10s/baseline-vs-kernel-only-side-by-side.mp4
```

## Useful Comparison Artifacts

```bash
outputs/ltx23-hq-extraopts-ablation-1080p10s/allkwl_cache8_pisa_preproj_ropecache_noprofile/perf.json
outputs/ltx23-hq-best-plus-nvfp4-1080p10s/bf16_vs_te_nvfp4_side_by_side.mp4
```

The BF16 best path before TE NVFP4 was measured at about `49.376 s`. The TE NVFP4 run above was
`45.871 s`, about `1.076x` faster for the same prompt/settings, with NVFP4 low-precision FFN enabled.
