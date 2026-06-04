# Diffusion Cache Experiments README

This README is the cache-specific home for the current diffusion cache work. It
keeps cache experiment notes out of the repository top-level README and records
what changed, how the acceleration works, what was measured, where the artifacts
are, and what is still open.

## Scope

- LTX-2.3 experiment branch: `cache-hq15-bench`.
- Cosmos3 migration branch: `cosmos3-cache-migration`.
- Base development branch: `ltx2-dit-fusion-report`.
- Main LTX-2.3 target: test cache strategies on the 10s two-stage pipelines,
  including HQ 15-step stage 1 and non-HQ 30-step stage 1.
- Cosmos3 target: migrate and benchmark TeaCache/PAB/Cache-DiT hooks on
  SGLang's Cosmos3 pipeline, with TeaCache as the active NRT follow-up.
- Prompts use concrete scenes instead of abstract stress prompts:
  - Prompt 0: elderly ceramic artist painting blue patterns on a porcelain vase.
  - Prompt 1: red fox running through tall grass at sunrise.

## What changed

Runtime changes:

- Added LTX-2.3 TeaCache residual replay through:
  - `python/sglang/multimodal_gen/runtime/cache/ltx2_teacache.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py`
- Added stage-1 output export and stage-1-only decode support through:
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding_av.py`
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/upsampling.py`
  - `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`
- Added env-first distilled LoRA strength overrides for experiments:
  - `SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_1`
  - `SGLANG_LTX2_STAGE1_DISTILLED_LORA_STRENGTH`
  - `SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_2`
  - `SGLANG_LTX2_STAGE2_DISTILLED_LORA_STRENGTH`

Script changes:

- `scripts/run_ltx23_sglang_hq_1080p10s.sh`
- `scripts/run_ltx23_sglang_nonhq_cache_10s.sh`
- `scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh`

The runners now record seed, stage-1-only flags, saved stage-1 output paths, and
LoRA strengths in the semantics JSON files. They also support:

```bash
STAGE1_ONLY_OUTPUT=1
SAVE_STAGE1_OUTPUT=1
```

`STAGE1_ONLY_OUTPUT=1` decodes the upsampled stage-1 latents and skips stage-2
refine. `SAVE_STAGE1_OUTPUT=1` keeps the normal final `out.mp4` and additionally
saves `stage1_out.mp4`.

## Cache method summary

| Method | How acceleration happens | Current result |
|---|---|---|
| KWL baseline | Computes every denoising step and keeps the selected kernel/fusion optimizations. | Reference for speed and visual comparisons. |
| TeaCache | Compares consecutive modulated inputs. On a hit, skips the LTX2 transformer block stack for that denoising step and replays cached video/audio residuals. Output norm/projection/unpatchify still run. | Best current candidate. Stage-1 speedup is meaningful on HQ and non-HQ. End-to-end speedup is limited in the offloaded two-stage setup because stage 2, decode, LoRA switching, and offload still run. |
| PAB | Reuses attention outputs over configured broadcast windows. | Previous 1:1 speed and identical-output run means the tested configuration did not produce effective skips. Do not use that result as accepted acceleration. |
| DBCache / Cache-DiT | Skips selected DiT blocks from a residual-difference policy. | Implemented through Cache-DiT flags. Aggressive preset had visible quality problems. A milder preset targeting about 1.5x remains worth testing. |
| LTX2 stage1 cache-core | Handwritten residual reuse inside the LTX2 stage-1 block stack. | Separate experimental ablation from TeaCache. Not accepted as the main result. |

## Cosmos3 TeaCache migration

Runtime changes:

- Added Cosmos3 TeaCache residual replay through:
  - `python/sglang/multimodal_gen/runtime/cache/cosmos3_teacache.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py`
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py`
- Added Cosmos3 cache benchmark and report scripts:
  - `scripts/run_cosmos3_cache_matrix.sh`
  - `scripts/make_cosmos3_cache_report.py`

Cosmos3 TeaCache uses the same high-level decision rule as LTX2: compare the
current modulated transformer input with the previous computed one, accumulate a
relative L1 distance, and on a hit skip the Cosmos3 transformer block stack by
replaying the cached residual. The skipped denoising step still runs scheduler
bookkeeping and decode later; the acceleration comes from avoiding transformer
block execution for that step.

NRT calibration on `2026-06-04` with `nvidia/Cosmos3-Nano`, 35 denoise steps,
121 frames, and prompt 0 showed that LTX-style thresholds do not transfer
directly:

| Variant | Threshold | Start | Hits | Skipped steps | Readout |
|---|---:|---:|---:|---|---|
| TeaCache t0.04 | 0.04 | 5 | 0 | `[]` | No real skip. Apparent prompt-0 speedup was first-run baseline noise. |
| TeaCache t0.06 | 0.06 | 5 | 0 | `[]` | No real skip. |
| TeaCache t0.08 | 0.08 | 5 | 0 | `[]` | No real skip. |
| TeaCache t0.12 | 0.12 | 5 | 0 | `[]` | No real skip. Logged `rel_l1` was about `1.00-1.12`. |

Because the observed Cosmos3 `rel_l1` scale is around `1.0`, the active sweep
uses Cosmos3-specific thresholds:

```bash
VARIANTS="baseline teacache_c105_s5 teacache_c110_s5 teacache_c115_s5 teacache_c120_s5"
```

These map to thresholds `1.05`, `1.10`, `1.15`, and `1.20`, with start step `5`
and max continuous hits `1`. Logs keep `SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS=1`
enabled for these variants so each recompute/hit decision can be audited from
the report artifacts.

NRT artifact roots:

```text
Low-threshold 16B sweep:
/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/junsongc/staging/sol-ltx-infer-cosmos3-cache/outputs/cosmos3-teacache-16b-nrt-4594090

Initial high-threshold probe:
/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/junsongc/staging/sol-ltx-infer-cosmos3-cache/outputs/cosmos3-teacache-16b-high-nrt-4594341
```

Completed 16B threshold-scale sweep:

```text
Remote:
/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/junsongc/staging/sol-ltx-infer-cosmos3-cache/outputs/cosmos3-teacache-16b-scale-nrt-4594652

Local quick-look artifacts:
/Users/junsongc/Desktop/s3/cosmos3-teacache-16b-scale-nrt-4594652/compare.mp4
/Users/junsongc/Desktop/s3/cosmos3-teacache-16b-scale-nrt-4594652/benchmark_report.html
```

16B prompt-0 timing:

| Variant | Threshold | Total s | Total x | Denoise s | Denoise x | Hits | Skipped steps | Visual readout |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Baseline | - | 50.305 | 1.000 | 44.385 | 1.000 | - | - | Normal greenhouse/botanist output. |
| TeaCache t1.05/start5 | 1.05 | 49.475 | 1.017 | 43.979 | 1.009 | 6 | `10,13,16,19,22,25` | Fog/noise overlay; not visually acceptable. |
| TeaCache t1.10/start5 | 1.10 | 45.847 | 1.097 | 40.870 | 1.086 | 10 | `5,8,11,14,17,20,23,26,29,32` | Fog/noise overlay; not visually acceptable. |
| TeaCache t1.15/start5 | 1.15 | 38.662 | 1.301 | 33.362 | 1.330 | 10 | `5,8,11,14,17,20,23,26,29,32` | Fog/noise overlay; not visually acceptable. |
| TeaCache t1.20/start5 | 1.20 | 55.403 | 0.908 | 31.734 | 1.399 | 10 | `5,8,11,14,17,20,23,26,29,32` | Fog/noise overlay; not visually acceptable. Total time had a decode/postprocess outlier. |

Readout:

- Cosmos3 TeaCache threshold scale is confirmed to be around `1.x`, not LTX2's
  `0.04-0.08` range.
- `max_hits=1` caps the skip pattern at roughly every third step for this
  prompt. `t1.15` is the best measured timing point in this run, but the visual
  output is not acceptable.
- The first and middle compare frames both show the same fog/noise overlay on
  every TeaCache variant, including the mild `t1.05` case. This points to a
  Cosmos3 TeaCache correctness issue to fix before increasing skip count or
  treating the speedup as usable.

## TeaCache mechanism

For LTX-2.3, TeaCache is stage/pass/shape keyed residual replay. The runtime
computes relative L1 distance between consecutive modulated inputs:

```text
rel_l1 = |current - previous|.mean() / |previous|.mean()
```

After polynomial rescaling, the distance is accumulated. If the accumulator is
below `SGLANG_LTX2_TEACACHE_THRESH`, that denoising step is a cache hit and the
transformer block stack is skipped:

```text
hidden_states = hidden_states + cached_video_residual
audio_hidden_states = audio_hidden_states + cached_audio_residual
```

Stage 2 is disabled by default:

```bash
SGLANG_LTX2_TEACACHE_STAGE2_DISABLE=1
```

That keeps the refinement pass conservative, but it also means total request
time can be dominated by uncached work.

## Latest TeaCache benchmark

Run:

```text
outputs/ltx23-teacache-hq-nonhq-matrix-10s-full-4545670
```

Pipelines:

- HQ: `LTX2TwoStageHQPipeline`, 15 stage-1 steps, `res2s`, 3 stage-2 steps,
  stage-1 LoRA `0.25`, stage-2 LoRA `0.5`.
- Non-HQ: `LTX2TwoStagePipeline`, 30 stage-1 steps, `euler`, 3 stage-2 steps,
  stage-1 LoRA `0.0`, stage-2 LoRA `1.0`.

Main local artifacts:

```text
/Users/junsongc/Desktop/s3/ltx23-teacache-hq-nonhq-matrix-10s-full-4545670/hq/prompt_1/compare.mp4
/Users/junsongc/Desktop/s3/ltx23-teacache-hq-nonhq-matrix-10s-full-4545670/nonhq/prompt_1/compare.mp4
```

Summary over both prompts:

| Pipeline | Variant | Avg total x | Avg stage1 x | Stage-1 skipped steps | Hits/computes | Readout |
|---|---|---:|---:|---|---|---|
| HQ 15-step | TeaCache 0.04/start6 | 0.992 | 1.323 | 6-13 | 24/27 | Conservative skip count. Stage-1 speedup exists, total is neutral/slightly slower. |
| HQ 15-step | TeaCache 0.06/start5 | 0.930 | 1.413 | 5-13 | 27/30 | More stage-1 skip, worse total in offloaded run. |
| HQ 15-step | TeaCache 0.08/start5 | 0.930 | 1.403 | 5-13 | 27/30 | Similar to 0.06/start5; no extra total benefit here. |
| Non-HQ 30-step | TeaCache 0.04/start6 | 1.009 | 1.295 | 8 stage-1 steps | 8/16 | Only neutral/slightly positive end-to-end setting in this run. |
| Non-HQ 30-step | TeaCache 0.06/start5 | 0.907 | 1.428 | 10 stage-1 steps | 10/15 | Better stage-1 speedup, worse total due uncached overhead. |
| Non-HQ 30-step | TeaCache 0.08/start5 | 0.907 | 1.485 | 11 stage-1 steps | 11/14 | Best stage-1 speedup, not best total in this setup. |

Per-prompt timing:

| Pipeline | Prompt | Variant | Total s | Total x | Stage1 s | Stage1 x | Stage2 s | Stage-1 skipped steps | Hits/computes |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| HQ 15-step | 0 | KWL baseline | 343.07 | 1.000 | 150.67 | 1.000 | 68.27 | - | - |
| HQ 15-step | 0 | TeaCache 0.04/start6 | 342.34 | 1.002 | 114.26 | 1.319 | 87.30 | 6-13 | 24/27 |
| HQ 15-step | 0 | TeaCache 0.06/start5 | 345.97 | 0.992 | 105.85 | 1.423 | 88.39 | 5-13 | 27/30 |
| HQ 15-step | 0 | TeaCache 0.08/start5 | 346.26 | 0.991 | 106.19 | 1.419 | 91.22 | 5-13 | 27/30 |
| Non-HQ 30-step | 0 | KWL baseline | 242.78 | 1.000 | 166.49 | 1.000 | 31.91 | - | - |
| Non-HQ 30-step | 0 | TeaCache 0.04/start6 | 238.66 | 1.017 | 128.81 | 1.293 | 38.10 | 6,8,11,13,16,18,21,24 | 8/16 |
| Non-HQ 30-step | 0 | TeaCache 0.06/start5 | 286.78 | 0.847 | 114.97 | 1.448 | 33.39 | 5,7,9,12,14,16,19,21,23,26 | 10/15 |
| Non-HQ 30-step | 0 | TeaCache 0.08/start5 | 286.78 | 0.847 | 110.68 | 1.504 | 33.31 | 5,7,9,11,14,16,18,20,23,25,28 | 11/14 |
| HQ 15-step | 1 | KWL baseline | 351.06 | 1.000 | 148.46 | 1.000 | 68.17 | - | - |
| HQ 15-step | 1 | TeaCache 0.04/start6 | 357.29 | 0.983 | 111.85 | 1.327 | 85.70 | 6-13 | 24/27 |
| HQ 15-step | 1 | TeaCache 0.06/start5 | 404.04 | 0.869 | 105.89 | 1.402 | 86.08 | 5-13 | 27/30 |
| HQ 15-step | 1 | TeaCache 0.08/start5 | 404.12 | 0.869 | 107.07 | 1.387 | 88.57 | 5-13 | 27/30 |
| Non-HQ 30-step | 1 | KWL baseline | 267.71 | 1.000 | 165.03 | 1.000 | 32.38 | - | - |
| Non-HQ 30-step | 1 | TeaCache 0.04/start6 | 267.71 | 1.000 | 127.15 | 1.298 | 33.76 | 6,8,11,13,16,18,21,24 | 8/16 |
| Non-HQ 30-step | 1 | TeaCache 0.06/start5 | 276.47 | 0.968 | 117.18 | 1.408 | 33.21 | 5,7,9,12,14,16,19,21,23,26 | 10/15 |
| Non-HQ 30-step | 1 | TeaCache 0.08/start5 | 276.47 | 0.968 | 112.59 | 1.466 | 33.10 | 5,7,9,11,14,16,18,20,23,25,28 | 11/14 |

Full TeaCache notes are also in [teacache.md](teacache.md).

## Stage-1 output and HQ color probe

The HQ prompt-1 baseline looked overexposed compared with non-HQ. A stage-1-only
probe was added to isolate whether the problem starts before stage-2 refine.

Completed probe:

```text
Remote:
/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/junsongc/staging/sol-ltx-infer-hq15-cache-bench/outputs/ltx23-hq-stage1-lora-ablation-20260602-10

Local:
/Users/junsongc/Desktop/s3/ltx23-hq-stage1-lora-ablation-20260602-10

Main compare:
/Users/junsongc/Desktop/s3/ltx23-hq-stage1-lora-ablation-20260602-10/stage1_lora_ada_compare.mp4
```

First-frame color statistics:

| Case | Luma mean | RGB mean | Clip >=250 RGB | Notes |
|---|---:|---|---|---|
| HQ stage1 KWL, fused-ada on | 200.10 | `[195.84, 220.49, 26.80]` | `[0.3579, 0.4730, 0.0351]` | Stage-1-only output is already very bright. |
| HQ stage1 KWL, fused-ada off | 200.10 | `[195.84, 220.49, 26.80]` | `[0.3579, 0.4730, 0.0351]` | Same as fused-ada on in this probe. |
| Non-HQ stage1 KWL | 99.23 | `[116.77, 102.32, 16.74]` | `[0.0067, 0.0006, 0.0001]` | Same prompt/seed, normal exposure. |
| Old HQ final KWL | 214.15 | `[205.51, 238.31, 19.70]` | `[0.3923, 0.7791, 0.0314]` | Stage-2 refine worsens green clipping. |
| Old non-HQ final KWL | 75.05 | `[85.04, 78.97, 6.74]` | `[0.0006, 0.0000, 0.0000]` | Normal exposure. |

Current interpretation:

- The HQ exposure issue starts in decoded upsampled stage-1 latents, before
  stage-2 refine.
- Stage 2 worsens green clipping, but it is not the sole cause.
- `SGLANG_HQ_KWL_FUSED_ADA_VALUES_ALL` did not change the stage-1-only output in
  this probe.
- The completed probe logs show the stage-1 distilled LoRA adapter loaded and
  applied to 1660 layers at strength `0.25`.
- The first requested LoRA=0 cases were invalid because request `extra` defaults
  overrode the env override and still applied stage-1 LoRA `0.25`. The runtime
  now checks the env override first.
- Follow-up full-video LoRA=0 reruns were attempted but canceled after repeated
  cluster startup stalls at text-encoder load. LoRA=0 visual quality is still an
  open ablation.

## How to rerun

TeaCache HQ/non-HQ matrix:

```bash
bash scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh
```

HQ single variant:

```bash
bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl_teacache_c04_s6
```

Non-HQ single variant:

```bash
bash scripts/run_ltx23_sglang_nonhq_cache_10s.sh kwl_cache_teacache_c04_s6
```

Stage-1-only visual probe:

```bash
STAGE1_ONLY_OUTPUT=1 bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl
```

Save stage-1 output while still running final stage-2 refine:

```bash
SAVE_STAGE1_OUTPUT=1 bash scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh
```

LoRA override probe:

```bash
SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_1=0.0 \
SGLANG_LTX2_DISTILLED_LORA_STRENGTH_STAGE_2=0.5 \
bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl
```

## Validation

The current code/doc update was checked with:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/sol_ltx_pycache python3 -m py_compile \
  python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding_av.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/upsampling.py

bash -n scripts/run_ltx23_sglang_hq_1080p10s.sh \
  scripts/run_ltx23_sglang_nonhq_cache_10s.sh \
  scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh

git diff --check
```

## Open items

- Re-run HQ LoRA=0 after the cluster startup issue is gone, then compare
  stage-1-only and final videos.
- Try a milder DBCache policy if the target is about 1.5x with acceptable
  visuals.
- Re-measure TeaCache in a setup where stage 2/offload/snapshot overhead does
  not dominate the wall clock, because the measured stage-1 speedup is stronger
  than the current end-to-end number.

## Cosmos3 migration

Cosmos3 support was imported from upstream SGLang into the
`cosmos3-cache-migration` branch. The upstream native pipeline serves:

- `nvidia/Cosmos3-Nano`
- `nvidia/Cosmos3-Super`
- `nvidia/Cosmos3-Super-Text2Image`
- `nvidia/Cosmos3-Super-Image2Video`

The local benchmark runner uses the user-facing size labels `16b` and `64b`,
mapped by default to `nvidia/Cosmos3-Nano` and `nvidia/Cosmos3-Super`. Override
the exact checkpoint paths with:

```bash
COSMOS3_16B_MODEL_PATH=/path/or/hf/id
COSMOS3_64B_MODEL_PATH=/path/or/hf/id
COSMOS3_16B_NUM_GPUS=1
COSMOS3_64B_NUM_GPUS=4
```

### Cosmos3 cache hooks

Cosmos3 has two pathways:

- UND text pathway: runs once per prompt and already caches text K/V internally.
- GEN visual pathway: runs every denoising step and is the target for cache
  acceleration.

Implemented hooks:

| Method | Cosmos3 implementation | What gets skipped |
|---|---|---|
| TeaCache | `runtime/cache/cosmos3_teacache.py` plus a hook in `Cosmos3OmniTransformer.forward`. | Full `gen_layers` stack on accepted denoising steps. Norm/projection/unpatchify still run. |
| PAB | `runtime/cache/cosmos3_pab.py`, installed by `Cosmos3DenoisingStage`. | GEN cross-attention output inside each generation layer for broadcast-window hits. |
| DBCache / Cache-DiT | `runtime/cache/cosmos3_block_adapter.py` registers `Cosmos3` and `FSDPCosmos3` with cache-dit. | Cache-DiT block-level reuse over `transformer.gen_layers`. |

TeaCache env knobs:

```bash
SGLANG_COSMOS3_TEACACHE_ENABLED=1
SGLANG_COSMOS3_TEACACHE_THRESH=0.04
SGLANG_COSMOS3_TEACACHE_START=5
SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS=1
```

PAB env knobs:

```bash
SGLANG_COSMOS3_PAB_ENABLED=1
SGLANG_COSMOS3_PAB_CROSS_WINDOW=2
SGLANG_COSMOS3_PAB_WARMUP=5
```

DBCache env knobs use the existing Cache-DiT controls:

```bash
SGLANG_CACHE_DIT_ENABLED=1
SGLANG_CACHE_DIT_FN=2
SGLANG_CACHE_DIT_BN=2
SGLANG_CACHE_DIT_WARMUP=5
SGLANG_CACHE_DIT_RDT=0.12
SGLANG_CACHE_DIT_MC=1
```

### Cosmos3 benchmark runner

The runner uses two concrete prompts:

- Human scene: elderly botanist watering orchids in a greenhouse.
- Animal scene: red fox running across a snowy forest trail.

Default matrix:

```bash
bash scripts/run_cosmos3_cache_matrix.sh
```

Default variants:

```text
baseline teacache_c04_s5 pab_cross2 dbcache_mild
```

Useful overrides:

```bash
ROOT=outputs/cosmos3-cache-matrix-$(date +%Y%m%d-%H%M%S) \
MODEL_SIZES="16b 64b" \
VARIANTS="baseline teacache_c04_s5 teacache_c08_s5 pab_cross2 dbcache_mild dbcache_target15" \
HEIGHT=480 WIDTH=832 NUM_FRAMES=81 NUM_INFERENCE_STEPS=35 \
bash scripts/run_cosmos3_cache_matrix.sh
```

Artifacts:

```text
<ROOT>/<model_size>/prompt_<idx>/<variant>/out.mp4
<ROOT>/<model_size>/prompt_<idx>/<variant>/perf.json
<ROOT>/<model_size>/prompt_<idx>/<variant>/semantics.json
<ROOT>/<model_size>/prompt_<idx>/compare.mp4
<ROOT>/benchmark_summary.json
<ROOT>/benchmark_summary.md
<ROOT>/benchmark_report.html
```

The report parser extracts total time, `Cosmos3DenoisingStage` time, TeaCache
skip stats, and PAB hit stats from `perf.json` plus logs. Cache-DiT uses the
existing cache-dit logs and speedup is read from total/denoise timing.

Current status: code and local syntax checks are complete. Real 16B/64B speed
and visual-quality acceptance still require cluster runs because this Mac does
not have the required GPU/runtime dependencies.
