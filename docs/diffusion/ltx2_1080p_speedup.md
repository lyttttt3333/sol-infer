# LTX2 1080p 10s Single-GPU Speedup

This note records the current best LTX-2.3 1080p/10s single-GPU result and the exact techniques used for the run.

## Benchmark Setup

- Model: `Lightricks/LTX-2.3`
- Pipeline: `LTX2TwoStagePipeline`
- Device scope: single GPU (`CUDA_VISIBLE_DEVICES=0`, `--num-gpus 1`)
- Resolution: `1088x1920`
- Frames/FPS: `241` frames at `24` fps
- Stage 1: `30` steps
- Stage 2: official 3-sigma refinement (`0.909375`, `0.725`, `0.421875`)
- Guidance scale: `3.0`
- LoRA: official LTX-2.3 distilled LoRA merged for stage 2
- Baseline: official Diffusers LTX-2.3 two-stage benchmark, no compile

The Diffusers baseline script intentionally stops at video VAE decode and does not include video postprocess, audio VAE decode, vocoder, or file saving. The SGLang best configuration is reproduced by `scripts/run_ltx23_best_1080p_single_gpu.sh`.

## Result

| Metric | Time | Speedup vs Diffusers no-compile | Speedup vs Diffusers compile |
| --- | ---: | ---: | ---: |
| Diffusers no-compile baseline | `119.811s` | `1.000x` | - |
| Diffusers compile baseline | `88.052s` | `1.361x` | `1.000x` |
| SGLang best, full measured request | `59.332s` | `2.019x` | `1.484x` |
| SGLang best, through video VAE decode | `54.915s` | `2.182x` | `1.603x` |

The 2x target from the Diffusers no-compile baseline is `59.905s`. The full measured request is `0.574s` under that target, and the through-video-decode metric is `4.990s` under that target.

## Enabled Techniques

These are semantics-preserving at the algorithm level. Kernel-level numeric ordering differences are expected and ignored for this comparison.

- `SGLANG_LTX2_FUSED_ADALN=1`: fuse AdaLN residual, norm, scale, and shift work.
- `SGLANG_LTX2_FUSED_QKNORM_ROPE=1`: fuse Q/K norm and RoPE preparation.
- `SGLANG_LTX2_FUSED_DUAL_MODULATE=1`: fuse paired video/audio modulation.
- `SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1`: fuse grouped Ada value updates.
- `SGLANG_LTX2_FUSED_RESIDUAL_GATE=1`: fuse residual gate updates.
- `SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1`: fuse FFN input projection and GELU.
- `SGLANG_LTX2_COMPILE_GATE_TO_OUT=1`: compile the attention gate-to-output subgraph.
- `SGLANG_LTX2_FUSED_AUDIO_QKVG=1`: fuse audio attention Q/K/V/gate projections.
- `SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=1`: share equivalent block-0 self-attention work across guidance branches.
- `SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=1`: share CFG/STG prefix computation before the first STG divergence block, then expand back to the full branch batch.
- `SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=1`: compile the shape-specific tiled video VAE decoder.
- `SGLANG_LTX2_VAE_COMPILE_MODE=max-autotune-no-cudagraphs`: use Inductor max-autotune for the tiled VAE decoder compile.

## Not Counted As Speedup Techniques

- The official two-stage design and resident two-stage placement are treated as baseline configuration.
- `SGLANG_DIFFUSION_DECODE_PROFILE=1` is only instrumentation.
- Fast video postprocess is not counted because the comparison stops at decode.
- Whole-model compile/CUDA graph, FA split tuning, FA SM margin tuning, global QKV fusion, KV fusion, RoPE embedding cache, and prompt preprojection were tested but were not part of the best result because they were neutral or slower.
