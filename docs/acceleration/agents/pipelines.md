# Optimization pipelines (agents)

Exact entries, flags, env, params. GB200 warmup-excluded. See
[../human/pipelines.md](../human/pipelines.html) for rationale.

## SANA-Video (1 GPU)
- entry: `scripts/sana/sana_video_sglang_run.py`
- baseline: `python scripts/sana/sana_video_sglang_run.py --output sana_baseline`
- fullopt (2.10×, cold-safe):
  `... --easycache 0.1 --linattn-bf16 --qkv-merge --compile`
- fullopt peak (2.56× warm): add `--max-autotune`  (first cold run warms a persistent
  inductor cache; sets `SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` +
  `TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1` + `TORCHINDUCTOR_CACHE_DIR=~/.cache/sgl_torchinductor`)
- spec: 832×480, 81f, 50 steps. methods: cache(EasyCache)+fusion+compile. no quant/sparse.
- numbers: 28.5s → 13.5s (default) / 11.0s (max-autotune warm)
- caveat: plain `--compile` = inductor `default` mode (max-autotune cold deadlocks).

## Cosmos3-Super 64B (4 GPU)
- entry: `scripts/cosmos/slurm_cosmos3_super.sh [baseline|fullopt]`
- required env: `MODEL_REPO ROOT PROMPT_FILE PROMPT_TAG`; prompt MUST be passed,
  e.g. `PROMPT_FILE=prompts/cosmos/robot_plate.json` (structured-JSON).
- fullopt sets: `VARIANT=teacache_c115_s10_m3` (TeaCache thr1.15/start10/max3) +
  `SGLANG_COSMOS3_FP4_LINEAR=1`, `FP4_TARGETS=gate_up,down,qkv,out`,
  `FP4_SKIP_FIRST_STEPS=3`, `FP4_SKIP_LAST_STEPS=3`.
- spec: 1280×720, 189f, 35 steps, guidance 6.0, flow_shift 10.0, max_seq 4096, tp/sp 4.
- methods: cache(TeaCache)+quant(NVFP4). dense attention.
- numbers: 97.2s → 43.1s (~2.26×). NVFP4 needs Blackwell+TE else BF16 fallback.

## LTX-2.3 1080p/10s (1 GPU)
- entry: `scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh [baseline|fullopt]`
- fullopt is self-contained (no extra env). bakes:
  - KWL fusion: `SGLANG_HQ_KWL_*` → `SGLANG_LTX2_FUSED_*` / compile
  - stage-1 cache: `SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET=8of15_last_29calls`
  - stage-2 PISA: `--component-attention-backends transformer=fa,transformer_2=piecewise_attn`
    (`sparsity 0.9, block 64, route score, stage2_dense_layers 0-1`)
  - NVFP4 video FFN: `SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1`
  - token-prune: `SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_{RATIO=0.5,METHOD=feat_norm,STEPS=1,2}`
- spec: 1088×1920, 241f, stage1 30 steps + stage2 3-sigma, guidance 3.0, distilled LoRA.
  `WARMUP=true` by default (compile excluded from timing).
- methods: all 5 (cache SCSP + quant NVFP4 + KWL fusion + PISA sparse + token-prune).
- numbers: 95.7s → 39.2s (~2.4×).

## matrix
| model | cache | quant | kernel | sparse | prune | entry | GPUs |
|---|---|---|---|---|---|---|---|
| sana | EasyCache | — | KWL+compile | — | — | sana/sana_video_sglang_run.py | 1 |
| cosmos3 | TeaCache | NVFP4 | — | — | — | cosmos/slurm_cosmos3_super.sh | 4 |
| ltx | SCSP | NVFP4 | KWL | PISA | yes | ltx/run_ltx23_sglang_hq_1080p10s.sh | 1 |
