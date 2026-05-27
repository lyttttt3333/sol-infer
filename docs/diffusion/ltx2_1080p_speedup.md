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

## CODA Roadmap Coverage

The retained LTX2 work covers the CODA roadmap items that can be expressed as local normalization, activation, modulation, residual, and fixed-shape decode fusion in this codebase. The remaining unfused transformer cost is mostly in attention core kernels and large cuBLASLt linear projections. Attention core is outside CODA GEMM-epilogue scope, while the remaining linear-projection opportunities require a true custom GEMM epilogue for operations such as `proj_out + gate/residual` or Q/K projection plus qk-norm/RoPE. The lightweight alternatives tested here, including `torch.compile`, separate Triton epilogues, cublasLt wrappers, DeepGEMM, and FlashInfer GEMM paths, were not retained unless they improved the full 1080p/10s run.

## Not Counted As Speedup Techniques

- The official two-stage design and resident two-stage placement are treated as baseline configuration.
- `SGLANG_DIFFUSION_DECODE_PROFILE=1` is only instrumentation.
- Fast video postprocess is not counted because the comparison stops at decode.
- Whole-model compile/CUDA graph, FA split tuning, FA SM margin tuning, global QKV fusion, KV fusion, RoPE embedding cache, and prompt preprojection were tested but were not part of the best result because they were neutral or slower.
- Stage-2 merged LoRA wrapper unwrapping (`SGLANG_LTX2_UNWRAP_MERGED_STAGE2_LORA`) was tested at full 1080p/10s and rejected: `66.487s` full request, `64.829s` through video VAE decode.
- Selective STG self-attention branch skipping was tested at full 1080p/10s and rejected: `64.895s` full request.
- The 8-warps LTX2 fused qknorm+RoPE variant improved microbenchmarks but was rejected at full 1080p/10s: `64.603s` full request, `60.393s` through video VAE decode.
- In-place LTX2 fused qknorm+RoPE improved microbenchmarks but was rejected at full 1080p/10s: `65.284s` full request, `61.104s` through video VAE decode.
- Cross-attention dual modulation (`SGLANG_LTX2_FUSED_CA_DUAL_MODULATE=1`) improved microbenchmarks but was rejected on top of the current best full 1080p/10s stack: `66.498s` full request, `61.624s` through video VAE decode.
- Ada direct video norm/residual fusion (`SGLANG_LTX2_FUSED_ADA_DIRECT=1`) improved microbenchmarks but was rejected on top of the current best full 1080p/10s stack: `68.702s` full request, `64.597s` through video VAE decode.
- In-place residual gate (`SGLANG_LTX2_INPLACE_RESIDUAL_GATE=1`) was implemented as a candidate and rejected on top of the current best full 1080p/10s stack: `67.090s` full request, `62.092s` through video VAE decode.
- Plain fused qknorm fallback (`SGLANG_LTX2_FUSED_QKNORM=1`) was tested on top of the current best full 1080p/10s stack and rejected: `66.685s` full request, `62.497s` through video VAE decode.
- A2V gate-to-output compile (`SGLANG_LTX2_COMPILE_A2V_GATE_TO_OUT=1`) improved isolated video gate-to-output microbenchmarks but was rejected in the full 1080p/10s stack: `74.490s` full request, `68.783s` through video VAE decode.
- Stage-2 audio gate-to-output compile (`SGLANG_LTX2_COMPILE_AUDIO_STAGE2_GATE_TO_OUT=1`) improved an isolated microbenchmark but was rejected on top of the current best full 1080p/10s stack: `68.848s` full request, `63.907s` through video VAE decode.
- Cross-attention-only KV projection fusion (`SGLANG_LTX2_FUSED_CROSS_KV=1`) improved isolated KV-concat microbenchmarks but was rejected on top of the current best full 1080p/10s stack: `64.801s` full request, `60.473s` through video VAE decode.
- Selective audio-query Q plus gate projection fusion (`SGLANG_LTX2_FUSED_AUDIO_Q_GATE=1`) improved the exact gate=32 audio-query microbenchmarks (`1.367x` stage 1, `1.318x` stage 2) but was rejected on top of the current best full 1080p/10s stack: `66.089s` full request, `61.948s` through video VAE decode.
- Video-only FFN input projection plus GELU fusion (`SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU_VIDEO_ONLY=1`) kept the video fused path but disabled it for audio shapes; it was rejected on top of the current best full 1080p/10s stack: `66.191s` full request, `61.581s` through video VAE decode.
- Stage-1 video FFN input projection plus GELU with cached contiguous `weight.T` (`SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU_PRETRANS_STAGE1_VIDEO=1`) improved the isolated `_addmm_activation` microbenchmark (`1.048x` vs the current `weight.t()` view path) but was rejected on top of the current best full 1080p/10s stack: `69.174s` full request, `64.850s` through video VAE decode.
- Audio attention gate in-place apply (`SGLANG_LTX2_INPLACE_AUDIO_GATE_APPLY=1`) improved isolated audio gate-apply microbenchmarks (`1.060x` stage 1, `1.022x` stage 2) but was rejected on top of the current best full 1080p/10s stack: `63.832s` full request, `59.732s` through video VAE decode.
- Audio-only packed Ada values (`SGLANG_LTX2_FUSED_ADA_VALUES_PACKED_AUDIO=1`) improved isolated audio all9 microbenchmarks (`1.167x` stage 1, `1.249x` stage 2) but was rejected on top of the current best full 1080p/10s stack: `66.896s` full request, `62.589s` through video VAE decode.
- FFN output projection plus residual/gate compile was rejected at microbench scale: video speedups were `0.930x` for stage 1 and `0.854x` for stage 2, while audio paths were about `0.55x`.
- FFN output projection plus residual/gate Triton GEMM-epilogue was prototyped as a true CODA-style candidate and rejected at microbench scale: the best tuned stage-1 video variant was `0.347x` versus cuBLASLt `F.linear + addcmul` (`11.454ms` vs `3.975ms`).
- GELU plus linear compile was rejected at microbench scale: video speedups were `0.904x` for stage 1 and `0.866x` for stage 2.
- Q plus gate projection fusion was rejected for the dominant video shapes (`0.967x` stage 1, `0.978x` stage 2); the small audio-only win is covered by the retained `SGLANG_LTX2_FUSED_AUDIO_QKVG=1` path instead.
- Shape-specific M padding for the fixed LTX2 linear shapes was rejected because it was neutral to slower on the dominant projections.
- C++ JIT qknorm-across-heads (`fused_inplace_qknorm_across_heads`) was rejected in this environment because the CUDA toolkit was unavailable to the JIT compiler (`CUDA_HOME` not set/found), so it could not produce verifiable runtime evidence.
- DeepGEMM and FlashInfer GEMM replacement paths were rejected in this environment because the tested BF16 large-GEMM variants either failed to compile on the current GB200/CUDA/PTX toolchain or were slower than the existing cuBLASLt path.
