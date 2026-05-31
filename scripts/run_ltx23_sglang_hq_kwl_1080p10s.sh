#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_HQ_VARIANT=kwl
exec "$SCRIPT_DIR/run_ltx23_sglang_hq_1080p10s.sh" "$@"
