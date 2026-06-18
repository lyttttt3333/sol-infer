# Sol-LTX-Infer — model deployment & inference-acceleration guide

> This is a fork of SGLang's `multimodal_gen` (diffusion) subsystem. It adds a
> **model-agnostic inference-acceleration framework** at
> `python/sglang/multimodal_gen/runtime/efficiency/` and an optimized LTX-2.3 HQ
> path. **This README is the authoritative entry point — start here.**
> (Component notes live in `python/sglang/multimodal_gen/.claude/CLAUDE.md`.)

---

## TL;DR for a new agent

- To **bring up a new model**: do **Phase 1** (port the diffusers pipeline to an
  SGLang baseline, correctness first), then **Phase 2** (declare a `ModelSpec`
  and switch on acceleration techniques). See below.
- The acceleration techniques split into **runtime techniques** (per-step data-flow
  hooks: token-prune, step-cache) and **build/load model-transforms** (installed
  once: sparse-attention/PISA, NVFP4 precision, KWL fusions). A model adapts by
  writing **one `ModelSpec`**; the framework reuses each technique.
- **⚠️ Before any benchmark or quality test, confirm the official configuration**
  (resolution / fps / frames / steps / sigmas / seed). See
  [Official configuration](#official-configuration-confirm-this-before-testing).
  A "speedup" or "quality" number measured at the wrong resolution/length is meaningless.

---

## Official configuration (CONFIRM THIS BEFORE TESTING)

The LTX-2.3 HQ two-stage reference config. Any timing / quality / VBench number
**must** be produced with these exact settings, or it does not compare:

| param | value |
|---|---|
| resolution | **1088 × 1920** (stage-1 runs half-res 544 × 960, stage-2 full-res) |
| frames | **241** |
| fps | **24** → duration ≈ **10 s** |
| seed | **42** |
| stage-1 | 15 res2s steps (with CFG), distilled-LoRA strength 0.25 |
| stage-2 | 3 res2s steps, sigmas **[0.909375, 0.725, 0.421875, 0.0]**, LoRA strength 0.5 |
| guidance_scale | 3.0 (video CFG) |
| negative prompt | the official long HQ negative prompt in `scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh` |

Canonical run:
```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh dense   # baseline (no optimizations)
# -> outputs/.../perf.json carries total_duration_ms + per-stage `steps`
```
Reference timing at this config (warmed): baseline ≈ **98 s**, full-opt ≈ **41 s**
(≈ 2.39×). A no-warmup run is compile-dominated (~110 s) and is **not** comparable
— always warm (`WARMUP=true`) before quoting a total.

---

## Architecture: the 4 lifecycle phases

A run moves through four phases; acceleration techniques attach at different ones:

```
① ASSEMBLE   runtime/pipelines/<model>_pipeline.py : create_pipeline_stages()
                 -> [TextEncoding, Denoising, Decoding]
② LOAD       stages/base.py:load_model() -> runtime/loader/      [NVFP4 quant here]
③ BUILD      runtime/models/dits/<model>.py __init__             [KWL fusions, attention backend here]
④ RUNTIME    runtime/pipelines_core/stages/<model>_denoising.py  [the per-step loop; step-cache, prune here]
                 |-- per step --> runtime/models/dits/<model>.py forward()  [block loop]
```

This is exactly why techniques split into two kinds (see below): things installed
at ②/③ are **ModelTransforms**; things that hook the per-step loop at ④ are
**Techniques**.

---

## Phase 1 — diffusers pipeline → SGLang baseline (correctness first, zero opt)

Goal: **same output as diffusers at the same seed.** This is the real work; the
framework does not help here.

| diffusers | SGLang landing spot | note |
|---|---|---|
| `pipe.transformer` | `runtime/models/dits/<model>.py` (+ module-level `EntryClass = <Transformer>`) | **route attention through the SGLang attention layer** so backends are swappable later |
| `pipe.vae` / `pipe.text_encoder` | `runtime/models/vaes/` · `runtime/models/encoders/` | often reuse existing |
| `pipe.scheduler` | `runtime/models/schedulers/` | port or reuse |
| weights | `runtime/loader/` | map diffusers state_dict keys → your modules |

Pipeline assembly + registration (auto-discovered by `registry.py` via `EntryClass`):
```python
# runtime/pipelines/my_pipeline.py
class MyPipeline(ComposedPipelineBase):
    pipeline_name = "MyPipeline"
    def create_pipeline_stages(self, server_args):
        return [TextEncodingStage(...), MyDenoisingStage(...), MyDecodingStage(...)]
EntryClass = MyPipeline
```
- Config: `configs/pipeline_configs/<model>.py` (or reuse `diffusers_generic.py` for
  structurally-standard models — may need little/no custom code).

**Phase-1 design choices that make Phase 2 cheap** (do these now):
1. attention goes through the framework attention layer → sparse attention later = a flag.
2. clean block list (`self.transformer_blocks`) → cache / prune seam.
3. clear prunable token segment → prune seam.

**Validation gate:** `sglang generate --model-path ... --prompt "..." --seed 42`
matches diffusers at the official config → **this is your baseline / reference.**

---

## Phase 2 — add inference techniques (the efficiency framework)

Only after the baseline is correct.

### 2a. Write one `ModelSpec` (the only model-specific declaration)

A `ModelSpec` is a small **declarative interface card**: it tells the framework
which structural seams the model offers and how to reach them. No algorithms.

```python
# python/sglang/multimodal_gen/runtime/efficiency/models/<model>_spec.py
@register_model_spec("MyTransformer")          # by transformer class name
def _spec():
    return ModelSpec(
        name="MyModel",
        get_blocks="transformer_blocks",                 # name OR callable -> the live nn.ModuleList
        prunable_segment=lambda h, ctx: (0, h.shape[1]), # which token span is prunable
        swappable_attention=True,                        # attention routes through the framework layer
    )
```
- The accessor (`get_blocks`, `prunable_segment`) can be an **attribute name** (resolved
  via `getattr`) or a **callable** (when reaching the seam needs logic: nested path,
  concatenated `video+text` sequence, computed segment). Both resolve to the real
  live entity on the model instance.
- Capabilities are the model's "slots". `compose()` type-checks a technique's
  required capabilities against the spec and **refuses** at compose-time (clear
  early error) if a slot is missing — the analogue of `BlockAdapterRegister.is_supported()`.

### 2b. Two kinds of "skill" — runnable vs methodology

Not every optimization can be a generic runnable technique. The framework has two kinds:

- **Runnable skills (generic Techniques/Transforms)** — the algorithm is model-independent,
  so you just `compose()` them: **token-prune** (shared `keep_indices` scorer),
  **step-cache**, **teacache**, **sparse-attention selection** (the backend registry is
  model-agnostic). These transfer to a new model for free / near-free.
- **Methodology skills (recipes, not code)** — for **inherently model-specific**
  optimizations whose kernels/quant target one model's exact ops (**KWL operator fusion**,
  **NVFP4 FFN quantization**). There is no generic runnable version; instead you **read the
  recipe and implement it for your model**, then register a per-model `ModelTransform`.
  See **`runtime/efficiency/skills/operator_fusion.md`**.

### 2c. Pick techniques by lifecycle phase

| technique | kind | phase | cost to add |
|---|---|---|---|
| sparse attention (PISA) | runnable | ③ build | **free**: `--component-attention-backends transformer=piecewise_attn` (a flag) |
| **token-prune** | runnable | ④ runtime | scorer is generic; **needs wiring** of model gather/scatter (a guarded block) |
| **step-cache** | runnable | ④ runtime | **needs wiring**: wrap the per-step call |
| **teacache** | runnable | ④ runtime | generic decision core; model stashes its modulated-input signal |
| KWL operator fusions | **methodology** | ③ build | read `skills/operator_fusion.md`, implement model kernels, register a `ModelTransform` |
| NVFP4 precision | **methodology** | ② load | same shape: model-specific quant + a registered transform |

→ Sparse-attention is free once attention routes through the framework layer.
token-prune/step-cache/teacache are runnable but touch `forward` (guarded — OFF
must be byte-identical). KWL/NVFP4 are **not** generic — follow the methodology skill.

### 2c. Validation discipline (add one at a time)

1. Each technique **OFF == byte-identical baseline** (structural guard guarantees it;
   verify it).
2. ON → measure **speedup + quality vs the Phase-1 baseline, at the official config**.
3. Only then add the next.

### 2d. Assemble a preset
```python
def my_model_full_opt():
    return [SparseAttention(...), NVFP4FFN(...), KWLFusions(...),
            TokenPrune(keep_ratio=0.5, ...), StepCache(...)]
# compose(my_model_full_opt(), spec) -> capability + conflict check + ordering -> Plan
```

---

## The efficiency framework (`runtime/efficiency/`)

```
efficiency/
├── schedule.py     Schedule[T] — time-varying params (e.g. "first 2 steps high precision")
├── technique.py    Technique base · Phase · Seam (effect set) · Capability · EXCLUSIVE_SEAMS
├── transform.py    ModelTransform base · TransformPhase (LOAD/BUILD)
├── spec.py         ModelSpec — a model's declared seams (the unified interface/adapter)
├── compose.py      compose() = capability type-check + conflict (effect) check + ordering -> Plan
├── registry.py     register_technique / register_transform / register_model_spec
├── presets.py      ltx_full_opt() — the 5-component assembly
├── techniques/     token_prune.py (shared scorer) · step_cache.py · teacache.py   [runnable]
├── transforms/     sparse_attention.py (PISA) · nvfp4_ffn.py · kwl_fusions.py     [build/load env-triggers]
├── skills/         operator_fusion.md — methodology recipe for model-specific fusion (KWL idea)
└── models/         ltx2_spec.py
```

Core idea: **generic algorithm + per-model `ModelSpec` declaring only the seams +
a registry connecting them** — the same idiom as SGLang's existing
`AttentionBackend` registry and `BlockAdapterRegister`. Two plugin kinds:

- **Technique** (runtime, per-step): inserts hooks into the data flow
  (`before_blocks`/`after_blocks`/`on_step`). e.g. token-prune, step-cache.
- **ModelTransform** (build/load, once): installs a kernel / quantizes weights /
  selects a backend, then runs. e.g. PISA, NVFP4, KWL. Delegates to the existing
  env/registry mechanisms — does not reimplement them.

`compose()` proves **structural** non-interference (exclusive seams: one attention
backend / one FFN precision / one token-set owner; phase ordering). It does **not**
prove numerical composition of lossy techniques — that is bounded by the
off==identity invariant of each technique + empirical measurement at the official config.

Self-test: `.conda/ltx23/bin/python scripts/ltx/efficiency_selftest.py` (CPU, 23 checks).

### LTX-2.3 full-opt status

`ltx_full_opt()` assembles all 5 components. Today: **token-prune is wired through
the framework** (scoring via `keep_indices`, guarded in `ltx_2_denoising.py`; GPU-
validated, stage-2 14.7 s → 11.1 s, total 45.1 s → 41.1 s warmed). PISA / NVFP4 / KWL
currently take effect via their existing env mechanisms (the framework transforms
emit the same env); step-cache (SCSP) still uses the existing cache-core path.

---

## Cluster / running notes

- 4-GPU-minimum QOS; submit via the `scripts/slurm_ltx23_*.sh` jobs. Compile cache
  persists under `outputs/.cache/`.
- On a GPU-less login node, importing the full `sglang` package hangs on CUDA
  enumeration (and torch import is slow off Lustre); run on a GPU node, or for a
  pure-subpackage unit test stub the parent packages (see `scripts/ltx/efficiency_selftest.py`).
- Always `WARMUP=true` before quoting a total time (no-warmup is compile-dominated).

---

*Upstream SGLang README and the older `README_LTX23_46S_HQ.md` are preserved in git
history. This file supersedes them as the project entry point.*
