#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -J ltx23-kwl-pisa-matrix
#SBATCH -o outputs/slurm/ltx23-kwl-pisa-matrix-%j.out
#SBATCH -e outputs/slurm/ltx23-kwl-pisa-matrix-%j.err

set -euo pipefail

cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer
ROOT="${ROOT:-outputs/ltx23-kwl-pisa-matrix-1080p10s}"
mkdir -p "$ROOT" outputs/slurm

run_one() {
  local mode="$1"
  local out_dir="$ROOT/$mode"
  echo "========== RUN $mode -> $out_dir =========="
  MODE="$mode" OUT_DIR="$out_dir" bash scripts/ltx/slurm_ltx23_kwl_stage2_pisa_1080p10s.sh
}

run_one kwl
run_one kwl_pisa_stage2
run_one kwl_pisa_all

.conda/ltx23/bin/python - <<'PY2'
import json
import os
from pathlib import Path
root = Path(os.environ.get('ROOT', 'outputs/ltx23-kwl-pisa-matrix-1080p10s'))
rows = []
for mode in ['kwl', 'kwl_pisa_stage2', 'kwl_pisa_all']:
    path = root / mode / 'summary.json'
    d = json.loads(path.read_text())
    rows.append(d)
kwl_total = next(d['total_s'] for d in rows if d['mode'] == 'kwl')
for d in rows:
    d['speedup_vs_matrix_kwl'] = kwl_total / d['total_s']
summary = {
    'root': str(root),
    'kwl_total_s': kwl_total,
    'rows': rows,
}
(root / 'matrix_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY2

.conda/ltx23/bin/python scripts/make_side_by_side_video.py \
  --left "$ROOT/kwl/out.mp4" \
  --right "$ROOT/kwl_pisa_stage2/out.mp4" \
  --out "$ROOT/kwl-vs-stage2-pisa-side-by-side.mp4" \
  --left-label "kwl" \
  --right-label "kwl + stage2 PISA"

.conda/ltx23/bin/python scripts/make_side_by_side_video.py \
  --left "$ROOT/kwl/out.mp4" \
  --right "$ROOT/kwl_pisa_all/out.mp4" \
  --out "$ROOT/kwl-vs-all-pisa-side-by-side.mp4" \
  --left-label "kwl" \
  --right-label "kwl + all PISA"
