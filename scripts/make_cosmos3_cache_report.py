#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stage_ms(perf: dict[str, Any], name: str) -> float | None:
    for item in perf.get("steps", []):
        if item.get("name") == name:
            return float(item.get("duration_ms", 0.0))
    stages = perf.get("stages", {})
    if name in stages:
        return float(stages[name])
    return None


def _parse_cache_stats(log_path: Path, label: str) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    pattern = re.compile(rf"{re.escape(label)}: (\{{.*\}})")
    last = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            last = match.group(1)
    if last is None:
        return None
    try:
        return ast.literal_eval(last)
    except Exception:
        return {"raw": last}


def _summarize_teacache(stats: dict[str, Any] | None) -> str:
    if not stats:
        return ""
    parts = []
    for key, value in sorted(stats.items()):
        if not isinstance(value, dict):
            continue
        skipped = value.get("skipped_steps", [])
        parts.append(
            f"{key}: hits={value.get('hits', 0)}/computes={value.get('computes', 0)}, "
            f"skip={skipped}"
        )
    return "; ".join(parts)


def _summarize_pab(stats: dict[str, Any] | None) -> str:
    if not stats:
        return ""
    total_hits = 0
    total_computes = 0
    skipped = set()
    for value in stats.values():
        if not isinstance(value, dict):
            continue
        total_hits += int(value.get("hits", 0) or 0)
        total_computes += int(value.get("computes", 0) or 0)
        skipped.update(value.get("skipped_steps", []) or [])
    return f"hits={total_hits}/computes={total_computes}, skip={sorted(skipped)}"


def _compare_videos(reference_path: Path, candidate_path: Path) -> dict[str, Any] | None:
    if cv2 is None or np is None:
        return None
    if not reference_path.exists() or not candidate_path.exists():
        return None

    reference = cv2.VideoCapture(str(reference_path))
    candidate = cv2.VideoCapture(str(candidate_path))
    if not reference.isOpened() or not candidate.isOpened():
        reference.release()
        candidate.release()
        return None

    frames = 0
    abs_sum = 0.0
    sq_sum = 0.0
    pixel_count = 0
    frame_mean_abs = []
    frame_psnr = []
    max_abs = 0
    while True:
        ok_reference, reference_frame = reference.read()
        ok_candidate, candidate_frame = candidate.read()
        if not ok_reference or not ok_candidate:
            break
        if reference_frame.shape != candidate_frame.shape:
            candidate_frame = cv2.resize(
                candidate_frame,
                (reference_frame.shape[1], reference_frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        diff = reference_frame.astype(np.float32) - candidate_frame.astype(np.float32)
        abs_diff = np.abs(diff)
        sq_diff = diff * diff
        mae = float(abs_diff.mean())
        mse = float(sq_diff.mean())
        frame_mean_abs.append(mae)
        frame_psnr.append(
            float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        )
        abs_sum += float(abs_diff.sum())
        sq_sum += float(sq_diff.sum())
        pixel_count += int(abs_diff.size)
        max_abs = max(max_abs, int(abs_diff.max()))
        frames += 1

    reference.release()
    candidate.release()

    if frames == 0 or pixel_count == 0:
        return None
    mse_all = sq_sum / pixel_count
    finite_psnr = [value for value in frame_psnr if math.isfinite(value)]
    return {
        "frames": frames,
        "mean_abs_pixel_diff": abs_sum / pixel_count,
        "p95_frame_mean_abs_pixel_diff": float(np.percentile(frame_mean_abs, 95)),
        "max_abs_pixel_diff": max_abs,
        "mean_psnr_db": (
            float("inf")
            if mse_all == 0.0
            else 20.0 * math.log10(255.0 / math.sqrt(mse_all))
        ),
        "min_frame_psnr_db": min(finite_psnr) if finite_psnr else float("inf"),
    }


def _variant_label(variant: str) -> str:
    match = re.fullmatch(r"teacache_c([0-9]+)_s([0-9]+)(?:_m([0-9]+))?", variant)
    if match:
        threshold = int(match.group(1)) / 100.0
        start = int(match.group(2))
        max_hits = int(match.group(3) or 1)
        return f"TeaCache t{threshold:.2f} start{start} max{max_hits}"
    return {
        "baseline": "Baseline",
        "teacache_c04_s5": "TeaCache t0.04 start5",
        "teacache_c06_s5": "TeaCache t0.06 start5",
        "teacache_c08_s5": "TeaCache t0.08 start5",
        "teacache_c12_s5": "TeaCache t0.12 start5",
        "teacache_c16_s5": "TeaCache t0.16 start5",
        "teacache_c20_s5": "TeaCache t0.20 start5",
        "teacache_c30_s5": "TeaCache t0.30 start5",
        "teacache_c105_s5": "TeaCache t1.05 start5",
        "teacache_c110_s5": "TeaCache t1.10 start5",
        "teacache_c115_s5": "TeaCache t1.15 start5",
        "teacache_c120_s5": "TeaCache t1.20 start5",
        "pab_cross2": "PAB cross window2",
        "pab_cross3": "PAB cross window3",
        "dbcache_mild": "DBCache mild",
        "dbcache_target15": "DBCache target1.5x",
    }.get(variant, variant)


def collect(root: Path, model_sizes: list[str], variants: list[str], prompt_count: int):
    rows = []
    baseline_by_model_prompt: dict[tuple[str, int], dict[str, Any]] = {}
    for model_size in model_sizes:
        for prompt_idx in range(prompt_count):
            base_perf = _read_json(
                root / model_size / f"prompt_{prompt_idx}" / "baseline" / "perf.json"
            )
            baseline_by_model_prompt[(model_size, prompt_idx)] = {
                "total_ms": float(base_perf.get("total_duration_ms", 0.0) or 0.0),
                "denoise_ms": _stage_ms(base_perf, "Cosmos3DenoisingStage") or 0.0,
                "video": root
                / model_size
                / f"prompt_{prompt_idx}"
                / "baseline"
                / "out.mp4",
            }

    for model_size in model_sizes:
        for prompt_idx in range(prompt_count):
            baseline = baseline_by_model_prompt[(model_size, prompt_idx)]
            for variant in variants:
                case_dir = root / model_size / f"prompt_{prompt_idx}" / variant
                perf = _read_json(case_dir / "perf.json")
                semantics = _read_json(case_dir / "semantics.json")
                log_path = root / "logs" / f"{model_size}_prompt{prompt_idx}_{variant}.log"
                total_ms = float(perf.get("total_duration_ms", 0.0) or 0.0)
                denoise_ms = _stage_ms(perf, "Cosmos3DenoisingStage") or 0.0
                teacache_stats = _parse_cache_stats(log_path, "Cosmos3 TeaCache stats")
                pab_stats = _parse_cache_stats(log_path, "Cosmos3 PAB stats")
                output_path = case_dir / "out.mp4"
                psnr = (
                    None
                    if variant == "baseline"
                    else _compare_videos(baseline["video"], output_path)
                )
                rows.append(
                    {
                        "model_size": model_size,
                        "model_path": semantics.get("model_path", ""),
                        "prompt_index": prompt_idx,
                        "variant": variant,
                        "variant_label": semantics.get(
                            "variant_label", _variant_label(variant)
                        ),
                        "total_s": total_ms / 1000.0 if total_ms else None,
                        "denoise_s": denoise_ms / 1000.0 if denoise_ms else None,
                        "total_speedup": (
                            baseline["total_ms"] / total_ms
                            if total_ms and baseline["total_ms"]
                            else None
                        ),
                        "denoise_speedup": (
                            baseline["denoise_ms"] / denoise_ms
                            if denoise_ms and baseline["denoise_ms"]
                            else None
                        ),
                        "teacache": _summarize_teacache(teacache_stats),
                        "pab": _summarize_pab(pab_stats),
                        "psnr": psnr,
                        "mean_psnr_db": (
                            None if psnr is None else psnr.get("mean_psnr_db")
                        ),
                        "min_frame_psnr_db": (
                            None if psnr is None else psnr.get("min_frame_psnr_db")
                        ),
                        "mean_abs_pixel_diff": (
                            None if psnr is None else psnr.get("mean_abs_pixel_diff")
                        ),
                        "output": str(output_path),
                        "perf": str(case_dir / "perf.json"),
                        "log": str(log_path),
                    }
                )
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _aggregate_by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    aggregates = []
    for variant, items in grouped.items():
        min_frame_psnr = [
            row.get("min_frame_psnr_db")
            for row in items
            if row.get("min_frame_psnr_db") is not None
        ]
        aggregates.append(
            {
                "variant": variant,
                "variant_label": items[0].get("variant_label", variant),
                "samples": len(items),
                "mean_total_speedup": _finite_mean(
                    [row.get("total_speedup") for row in items]
                ),
                "mean_denoise_speedup": _finite_mean(
                    [row.get("denoise_speedup") for row in items]
                ),
                "mean_psnr_db": _finite_mean(
                    [row.get("mean_psnr_db") for row in items]
                ),
                "worst_min_frame_psnr_db": (
                    min(min_frame_psnr) if min_frame_psnr else None
                ),
                "mean_abs_pixel_diff": _finite_mean(
                    [row.get("mean_abs_pixel_diff") for row in items]
                ),
            }
        )

    return sorted(
        aggregates,
        key=lambda item: (
            0 if item["variant"] == "baseline" else 1,
            item["variant_label"],
        ),
    )


def write_reports(root: Path, rows: list[dict[str, Any]]) -> None:
    aggregates = _aggregate_by_variant(rows)
    summary = {"aggregates": aggregates, "rows": rows}
    (root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    headers = [
        "Model",
        "Prompt",
        "Variant",
        "Total s",
        "Total x",
        "Denoise s",
        "Denoise x",
        "TeaCache",
        "PAB",
        "Mean PSNR dB",
        "Min Frame PSNR dB",
        "Mean Abs Diff",
        "Output",
    ]
    aggregate_headers = [
        "Variant",
        "Samples",
        "Mean Total x",
        "Mean Denoise x",
        "Mean PSNR dB",
        "Worst Min Frame PSNR dB",
        "Mean Abs Diff",
    ]
    md_lines = ["# Cosmos3 Cache Benchmark", "", "## Averages By Variant", ""]
    md_lines.append("|" + "|".join(aggregate_headers) + "|")
    md_lines.append("|" + "|".join(["---"] * len(aggregate_headers)) + "|")
    for row in aggregates:
        md_lines.append(
            "|"
            + "|".join(
                [
                    row["variant_label"],
                    str(row["samples"]),
                    _fmt(row["mean_total_speedup"]),
                    _fmt(row["mean_denoise_speedup"]),
                    _fmt(row["mean_psnr_db"]),
                    _fmt(row["worst_min_frame_psnr_db"]),
                    _fmt(row["mean_abs_pixel_diff"]),
                ]
            )
            + "|"
        )
    md_lines.extend(["", "## Per Sample", "", "|" + "|".join(headers) + "|"])
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        md_lines.append(
            "|"
            + "|".join(
                [
                    row["model_size"],
                    str(row["prompt_index"]),
                    row["variant_label"],
                    _fmt(row["total_s"]),
                    _fmt(row["total_speedup"]),
                    _fmt(row["denoise_s"]),
                    _fmt(row["denoise_speedup"]),
                    row["teacache"],
                    row["pab"],
                    _fmt(row["mean_psnr_db"]),
                    _fmt(row["min_frame_psnr_db"]),
                    _fmt(row["mean_abs_pixel_diff"]),
                    row["output"],
                ]
            )
            + "|"
        )
    (root / "benchmark_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    html_rows = []
    for row in rows:
        cells = [
            row["model_size"],
            str(row["prompt_index"]),
            row["variant_label"],
            _fmt(row["total_s"]),
            _fmt(row["total_speedup"]),
            _fmt(row["denoise_s"]),
            _fmt(row["denoise_speedup"]),
            row["teacache"],
            row["pab"],
            _fmt(row["mean_psnr_db"]),
            _fmt(row["min_frame_psnr_db"]),
            _fmt(row["mean_abs_pixel_diff"]),
            row["output"],
        ]
        html_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>"
        )
    aggregate_rows = []
    for row in aggregates:
        cells = [
            row["variant_label"],
            str(row["samples"]),
            _fmt(row["mean_total_speedup"]),
            _fmt(row["mean_denoise_speedup"]),
            _fmt(row["mean_psnr_db"]),
            _fmt(row["worst_min_frame_psnr_db"]),
            _fmt(row["mean_abs_pixel_diff"]),
        ]
        aggregate_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>"
        )
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Cosmos3 Cache Benchmark</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f6f8fa; text-align: left; position: sticky; top: 0; }}
    td:nth-child(4), td:nth-child(5), td:nth-child(6), td:nth-child(7),
    td:nth-child(10), td:nth-child(11), td:nth-child(12) {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
  </style>
</head>
<body>
  <h1>Cosmos3 Cache Benchmark</h1>
  <h2>Averages By Variant</h2>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in aggregate_headers)}</tr></thead>
    <tbody>{''.join(aggregate_rows)}</tbody>
  </table>
  <h2>Per Sample</h2>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
</body>
</html>
"""
    (root / "benchmark_report.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-sizes", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect(
        root,
        model_sizes=args.model_sizes.split(),
        variants=args.variants.split(),
        prompt_count=args.prompt_count,
    )
    write_reports(root, rows)


if __name__ == "__main__":
    main()
