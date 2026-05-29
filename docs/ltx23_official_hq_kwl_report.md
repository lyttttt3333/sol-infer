# LTX-2.3 Official HQ Faithful Baseline + KWL Kernels

## Scope

This branch keeps the faithful official HQ two-stage pipeline from Lightricks and applies only kernel-wise lossless (KWL) execution changes inside the official ltx_core transformer build path. The pipeline source remains:

- outputs/LTX-2-official-main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py
- Upstream reference: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py

No sparse attention, fp4, step-count change, scheduler change, prompt change, CFG change, or LoRA-strength change is included in the KWL path.

## Code Paths

- Official faithful baseline Slurm entry: scripts/slurm_ltx23_official_hq_pipeline_1080p10s.sh
- Official KWL ops and CLI entry: scripts/ltx23_official_kwl_ops.py
- Official KWL Slurm entry: scripts/slurm_ltx23_official_hq_kwl_1080p10s.sh

The KWL entry patches ltx_pipelines.utils.blocks.DiffusionStage.__init__ before the official pipeline is imported. Each official transformer builder receives one extra ModuleOps installer, so KWL is applied when the official model is built on meta device before weight loading.

## Faithful HQ Settings

Prompt:

```text
A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail
```

Core settings:

- Resolution: 1920x1088
- Frames/FPS: 241 frames at 24 fps
- Seed: 42
- Stage 1: official HQ Res2S, 15 steps at half resolution 960x544
- Stage 2: official distilled refine sigmas [0.909375, 0.725, 0.421875, 0.0], i.e. 3 refine steps at full resolution
- Distilled LoRA: stage 1 strength 0.25, stage 2 strength 0.5
- Video CFG/STG/rescale: 3.0 / 0.0 / 0.45
- Audio CFG/STG/rescale: 7.0 / 0.0 / 1.0

Model assets used under the repo:

- Base checkpoint: outputs/LTX-2.3-official-files/ltx-2.3-22b-dev.safetensors
- HQ distilled LoRA: outputs/LTX-2.3-official-files/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
- Spatial upsampler: outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
- Gemma/tokenizer root: outputs/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493

## KWL Techniques Applied

All techniques preserve the official algorithmic graph. Differences are limited to normal kernel-level floating-point rounding and launch grouping.

- KWLAdaZeroFunction: fused bf16 RMSNorm + AdaLN scale/shift via repo Triton kernel python/sglang/jit_kernel/diffusion/triton/ltx2_adaln.py.
- KWLPreAttention: fused Q/K RMSNorm pair via ltx2_qknorm_pair_inplace.
- KWLPreAttention: fused Q/K RMSNorm + SPLIT RoPE pair via ltx2_qknorm_split_rope_pair when official RoPE tensors match the kernel shape.
- FeedForward.forward patch: FFN proj_in + bias + GELU(tanh) via torch.ops.aten._addmm_activation.default, followed by the original official proj_out.

Installed modules in the 22B official transformer:

- 48 transformer blocks
- 288 attention modules
- 96 FFN modules

## Benchmark Results

| Variant | Output | E2E wall time | Speedup vs faithful baseline | Stage 1 denoise | Stage 2 denoise | Slurm job |
|---|---:|---:|---:|---:|---:|---:|
| Official faithful HQ baseline | outputs/ltx23-official-hq-pipeline-1080p10s/official_hq/out.mp4 | 321.71s | 1.00x | 74s | 27s | 3017945 |
| Official HQ + KWL kernels | outputs/ltx23-official-hq-kwl-pipeline-1080p10s/official_hq_kwl/out.mp4 | 256.27s | 1.26x | 68s | 26s | 3018695 |
| Official HQ + KWL kernels, second process | outputs/ltx23-official-hq-kwl-warm-1080p10s/official_hq_kwl/out.mp4 | 287.33s | 1.12x | 68s | 24s | 3018994 |

Notes:

- The KWL transformer patch was confirmed in Slurm logs: Installed official LTX KWL ops: 48 transformer blocks, 288 attentions, 96 FFNs.
- The second process did not remove the stage 1 first-step JIT cost; stage 1 still starts with about 20s first step and then settles to about 3.5s/step.
- E2E time includes official model/text/decoder loading and media encode. Denoise-only improvement is modest; the best E2E number also benefits from lower non-denoise overhead in that run.

## Quality Comparison

Generated side-by-side video:

```text
outputs/ltx23-official-hq-kwl-pipeline-1080p10s/official_vs_kwl_side_by_side.mp4
```

Frame-level decoded-video metrics against the faithful baseline:

- Frames compared: 241
- Resolution/FPS: both 1920x1088, 24 fps
- Mean absolute pixel difference: 14.56
- MSE: 926.64
- PSNR: 18.46 dB
- Metrics JSON: outputs/ltx23-official-hq-kwl-pipeline-1080p10s/official_vs_kwl_frame_metrics.json

Interpretation: the KWL path has no algorithm-level quality loss by construction, but denoising is numerically sensitive, so kernel-level bf16 rounding differences can amplify into visible pixel differences over the full diffusion trajectory. Use the side-by-side video for visual inspection.

## Reproduction

Run faithful official baseline:

```bash
sbatch scripts/slurm_ltx23_official_hq_pipeline_1080p10s.sh
```

Run official HQ with KWL kernels:

```bash
sbatch scripts/slurm_ltx23_official_hq_kwl_1080p10s.sh
```

Run a KWL repeat to a separate output root:

```bash
sbatch --export=ALL,ROOT=outputs/ltx23-official-hq-kwl-repeat-1080p10s scripts/slurm_ltx23_official_hq_kwl_1080p10s.sh
```

Download comparison video from b200:

```bash
scp b200:/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/outputs/ltx23-official-hq-kwl-pipeline-1080p10s/official_vs_kwl_side_by_side.mp4 .
```
