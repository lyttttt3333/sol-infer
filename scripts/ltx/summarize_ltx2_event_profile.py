#!/usr/bin/env python3
"""Summarize LTX2 CUDA event profile output.

The event profile contains nested timers and often includes warmup/first-use
outliers. This script keeps parent timers separate, then reports a leaf-ish
hotspot ranking with an optional steady estimate that subtracts the largest
sample for each event bucket.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


PARENT_PREFIXES = (
    "ltx2_dit_block::",
    "ltx2_attention::",
    "ltx2_feedforward::",
)


def split_scoped_name(name: str) -> tuple[str, str, str]:
    match = re.match(r"ltx2_phase::([^:]+)::step_([^:]+)::(.*)", name)
    if match is None:
        return "other", "", name
    return match.group(1), match.group(2), match.group(3)


def category(rest: str) -> str:
    if rest.startswith("ltx2_official_fa4_attention::"):
        return "dense_fa_attention_core"
    if rest.startswith("ltx2_attention_core::"):
        return "attention_backend_core"
    if rest.startswith("ltx2_attention_proj::"):
        return "attention_linear_proj_gate_out"
    if "te_nvfp4_ffn_proj" in rest:
        return "ffn_te_nvfp4_linear"
    if rest.startswith("ltx2_fused_ffn_proj_in_gelu"):
        return "ffn_proj_in_gelu_fused"
    if rest.startswith("ltx2_compiled_gate_to_out"):
        return "attention_gate_to_out_compile"
    if rest.startswith("ltx2_fused_ca_dual_modulate"):
        return "ca_dual_modulation_fused"
    if rest.startswith("ltx2_fused_dual_modulate"):
        return "dual_modulation_fused"
    if (
        rest.startswith("ltx2_fused_adaln")
        or rest.startswith("ltx2_fused_rms_adaln")
        or rest.startswith("ltx2_fused_ada_values")
    ):
        return "adaln_modulation_fused"
    lower = rest.lower()
    if "rope" in lower or "qknorm" in lower:
        return "qknorm_rope"
    if rest.startswith("ltx2_feedforward_proj") or rest.startswith("ltx2_ffn"):
        return "ffn_other"
    return "other_leaf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--top", type=int, default=24)
    args = parser.parse_args()

    data = json.loads(args.profile.read_text())
    rows = data["stats"]

    parents: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    aggregate: dict[tuple[str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    top_events: list[tuple[float, int, str, str, str, str, float]] = []

    for row in rows:
        stage, step, rest = split_scoped_name(row["name"])
        total_ms = float(row["total_ms"])
        count = int(row["count"])
        max_ms = float(row.get("max_ms", 0.0))

        if rest.startswith(PARENT_PREFIXES):
            parent_key = (stage, rest.split("::", 1)[0])
            parents[parent_key][0] += total_ms
            parents[parent_key][1] += count
            continue

        cat = category(rest)
        steady_ms = max(0.0, total_ms - max_ms) if count > 1 else total_ms
        steady_count = max(1, count - 1) if count > 1 else count
        item = aggregate[(stage, cat)]
        item[0] += total_ms
        item[1] += count
        item[2] += steady_ms
        item[3] += steady_count
        item[4] = max(item[4], max_ms)
        top_events.append((total_ms, count, stage, step, cat, rest, max_ms))

    print(
        f"profile={args.profile} event_count={data.get('event_count')} "
        f"stat_rows={len(rows)}"
    )
    print()
    print("Parent timers are nested and should not be summed as percentages:")
    for (stage, name), values in sorted(
        parents.items(), key=lambda item: item[1][0], reverse=True
    )[: args.top]:
        print(
            f"{stage:7s} {name:24s} "
            f"total_ms={values[0]:10.1f} count={int(values[1]):7d}"
        )

    print()
    print("Leaf-ish category ranking; steady_ms subtracts one max sample per event:")
    for (stage, cat), values in sorted(
        aggregate.items(), key=lambda item: item[1][2], reverse=True
    )[: args.top]:
        total_ms, count, steady_ms, steady_count, max_ms = values
        steady_avg_us = 1000.0 * steady_ms / max(1.0, steady_count)
        print(
            f"{stage:7s} {cat:32s} total_ms={total_ms:10.1f} "
            f"steady_ms={steady_ms:10.1f} count={int(count):7d} "
            f"steady_avg_us={steady_avg_us:9.2f} max_ms={max_ms:8.1f}"
        )

    print()
    print("Top individual leaf-ish events:")
    for total_ms, count, stage, step, cat, rest, max_ms in sorted(
        top_events, reverse=True
    )[: args.top]:
        print(
            f"{total_ms:10.1f}ms count={count:5d} {stage:6s} "
            f"step={step:>3s} {cat:32s} {rest[:96]} max_ms={max_ms:.1f}"
        )


if __name__ == "__main__":
    main()
