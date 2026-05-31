#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -J ltx23-hq-pis-cachecore
#SBATCH -o outputs/slurm/ltx23-hq-pis-cachecore-%j.out
#SBATCH -e outputs/slurm/ltx23-hq-pis-cachecore-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
mkdir -p outputs/slurm

ROOT="${ROOT:-outputs/ltx23-sglang-hq-stage2pis-cachecore-matrix-1080p10s}"
FORCE="${FORCE:-1}"
WARMUP="${WARMUP:-true}"
WARMUP_STEPS="${WARMUP_STEPS:-15}"
SUMMARY_JSON="$ROOT/benchmark_summary.json"
COMPARE_MP4="$ROOT/stage2pis-cachecore-fourway.mp4"
mkdir -p "$ROOT"

names=(nocache cache12 cache8 cache5)
run_variants=(kwl_stage2_sparse kwl_stage1_cache_core_stage2_sparse kwl_stage1_cache_core_stage2_sparse kwl_stage1_cache_core_stage2_sparse)
presets=(none 12of15_delta05_29calls 8of15_last_29calls 5of15_blend_ema_29calls)
short_labels=("No cache" "12/15 delta" "8/15 last" "5/15 blend")
devices=(0 1 2 3)
ports=(30605 30615 30625 30635)
pids=()

run_case() {
  local name="$1"
  local run_variant="$2"
  local preset="$3"
  local device="$4"
  local port="$5"
  local out_dir="$ROOT/$name"
  echo "[launch] $name variant=$run_variant preset=$preset gpu=$device port=$port"
  (
    export CUDA_VISIBLE_DEVICES="$device"
    export SGLANG_HQ_VARIANT="$run_variant"
    export ROOT="$ROOT"
    export OUT_DIR="$out_dir"
    export FORCE="$FORCE"
    export WARMUP="$WARMUP"
    export WARMUP_STEPS="$WARMUP_STEPS"
    export MASTER_PORT="$port"
    if [[ "$preset" != "none" ]]; then
      export SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET="$preset"
      export SGLANG_LTX2_STAGE1_CACHE_CORE_CACHE_DEVICE="${SGLANG_LTX2_STAGE1_CACHE_CORE_CACHE_DEVICE:-default}"
    else
      unset SGLANG_LTX2_STAGE1_CACHE_CORE_PRESET
    fi
    bash scripts/run_ltx23_sglang_hq_1080p10s.sh "$run_variant"
  ) > "$ROOT/$name.log" 2>&1 &
  pids+=("$!")
}

echo "[start] $(date -Is) root=$ROOT force=$FORCE warmup=$WARMUP warmup_steps=$WARMUP_STEPS"
echo "[reference] external dev_cache_core: outputs/external/ltx_cache_dev_cache_core @ $(git -C outputs/external/ltx_cache_dev_cache_core rev-parse --short HEAD 2>/dev/null || true)"
for i in "${!names[@]}"; do
  run_case "${names[$i]}" "${run_variants[$i]}" "${presets[$i]}" "${devices[$i]}" "${ports[$i]}"
done

set +e
statuses=()
for pid in "${pids[@]}"; do
  wait "$pid"
  statuses+=("$?")
done
set -e

failed=0
for i in "${!names[@]}"; do
  echo "[done-runs] ${names[$i]} status=${statuses[$i]} log=$ROOT/${names[$i]}.log"
  if [[ "${statuses[$i]}" != "0" ]]; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "[error] at least one case failed; inspect $ROOT/*.log" >&2
  exit 1
fi

for name in "${names[@]}"; do
  for required in "$ROOT/$name/out.mp4" "$ROOT/$name/perf.json" "$ROOT/$name/hq_semantics.json"; do
    if [[ ! -s "$required" ]]; then
      echo "[error] missing required output: $required" >&2
      exit 2
    fi
  done
done

.conda/ltx23/bin/python - "$ROOT" "$SUMMARY_JSON" "$COMPARE_MP4" "${names[@]}" -- "${short_labels[@]}" <<'PYSUM'
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

sep = sys.argv.index('--')
root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
compare_mp4 = Path(sys.argv[3])
names = sys.argv[4:sep]
short_labels = sys.argv[sep + 1:]
label_map = dict(zip(names, short_labels))
preset_map = {
    'nocache': 'none',
    'cache12': '12of15_delta05_29calls',
    'cache8': '8of15_last_29calls',
    'cache5': '5of15_blend_ema_29calls',
}
external_validated = {
    'cache12': {'budget': '12/15', 'external_ms_ssim': 0.959, 'external_mae': 0.01731057},
    'cache8': {'budget': '8/15', 'external_ms_ssim': 0.85941, 'external_mae': 0.04040436},
    'cache5': {'budget': '5/15', 'external_ms_ssim': 0.70133, 'external_mae': 0.07535494},
}
stage_patterns = {
    'stage1_s': re.compile(r'LTX2AVDenoisingStage'),
    'stage2_s': re.compile(r'LTX2RefinementStage'),
    'decode_s': re.compile(r'LTX2AVDecodingStage'),
}

def load_perf(name: str):
    d = root / name
    perf = json.loads((d / 'perf.json').read_text())
    semantics = json.loads((d / 'hq_semantics.json').read_text())
    steps = perf.get('steps', []) or []
    stage_times = {key: 0.0 for key in stage_patterns}
    for step in steps:
        stage_name = str(step.get('name', ''))
        for key, pat in stage_patterns.items():
            if pat.search(stage_name):
                stage_times[key] += float(step.get('duration_ms', 0.0)) / 1000.0
    denoise_steps = perf.get('denoise_steps_ms', []) or []
    cache_stats = None
    log_path = root / f'{name}.log'
    if log_path.exists():
        text = log_path.read_text(errors='replace')
        matches = re.findall(r'LTX2 stage1 cache core stats for stage1: (\{.*?\})(?:\n|$)', text)
        if matches:
            try:
                cache_stats = ast.literal_eval(matches[-1])
            except Exception:
                cache_stats = {'raw': matches[-1]}
    item = {
        'label': label_map.get(name, name),
        'preset': preset_map.get(name),
        'video': str(d / 'out.mp4'),
        'perf_json': str(d / 'perf.json'),
        'semantics_json': str(d / 'hq_semantics.json'),
        'log': str(log_path),
        'total_s': float(perf.get('total_duration_ms', 0.0)) / 1000.0,
        'denoise_total_s': sum(float(x.get('duration_ms', 0.0)) for x in denoise_steps) / 1000.0,
        'denoise_step_count': len(denoise_steps),
        'stage1_s': stage_times['stage1_s'],
        'stage2_s': stage_times['stage2_s'],
        'decode_s': stage_times['decode_s'],
        'cache_stats': cache_stats,
        'semantics': semantics,
    }
    item.update(external_validated.get(name, {}))
    if cache_stats:
        item['cache_real_calls'] = cache_stats.get('real_calls')
        item['cache_skipped_calls'] = cache_stats.get('skipped_calls')
        item['cache_hit_rate'] = cache_stats.get('hit_rate')
    return item

results = {name: load_perf(name) for name in names}
base = results['nocache']
for item in results.values():
    item['speedup_vs_nocache_total'] = base['total_s'] / item['total_s'] if item['total_s'] else None
    item['speedup_vs_nocache_stage1'] = base['stage1_s'] / item['stage1_s'] if item['stage1_s'] else None
    item['speedup_vs_nocache_stage2'] = base['stage2_s'] / item['stage2_s'] if item['stage2_s'] else None

items = []
for name in names:
    item = results[name]
    label = f"{item['label']} {item['total_s']:.1f}s"
    if item.get('cache_skipped_calls') is not None:
        label += f" skip{item['cache_skipped_calls']}"
    items.extend(['--item', f"{label}={item['video']}"])
subprocess.run([
    '.conda/ltx23/bin/python',
    'scripts/make_multiway_video.py',
    *items,
    '--out', str(compare_mp4),
    '--cols', '4',
    '--tile-width', '640',
    '--tile-height', '360',
], check=True)

summary = {
    'root': str(root),
    'side_by_side_video': str(compare_mp4),
    'pipeline': 'LTX2TwoStageHQPipeline',
    'resolution': '1920x1088',
    'num_frames': 241,
    'fps': 24,
    'prompt': 'A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail',
    'baseline_variant': 'nocache',
    'baseline_method': 'KWL + stage2 video self-attention PIS, no stage1 cache',
    'cache_reference': {
        'repo': 'https://github.com/lyttttt3333/ltx_cache-/tree/dev_cache_core',
        'local_checkout': 'outputs/external/ltx_cache_dev_cache_core',
        'external_config': 'outputs/external/ltx_cache_dev_cache_core/configs/ltx23_hq_winners.json',
        'note': 'External official-HQ 31-call winner schedules are mapped to this SGLang HQ stage-1 29-call topology by local presets.',
    },
    'variants': results,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PYSUM

echo "[done] side-by-side: $COMPARE_MP4"
echo "[done] summary: $SUMMARY_JSON"
