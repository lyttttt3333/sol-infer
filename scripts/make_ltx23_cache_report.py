#!/usr/bin/env python3
"""Build a compact cache benchmark report for LTX-2.3 runs."""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
from pathlib import Path
from typing import Any


TEACACHE_RE = re.compile(r"LTX2 TeaCache stats for .*?: (\{.*\})")
STAGE1_CACHE_RE = re.compile(r"LTX2 stage1 cache core stats for .*?: (\{.*\})")


def _split_variants(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _stage_seconds(perf: dict[str, Any]) -> dict[str, float]:
    stage_times: dict[str, float] = {}
    for item in perf.get("steps", []) or []:
        name = str(item.get("name", ""))
        duration = float(item.get("duration_ms", 0.0)) / 1000.0
        if "LTX2AVDenoisingStage" in name or "AVDenoising" in name:
            stage_times["stage1"] = duration
        elif "LTX2RefinementStage" in name or "Refinement" in name:
            stage_times["stage2"] = duration
    return stage_times


def _denoise_seconds(perf: dict[str, Any]) -> float:
    return sum(
        float(item.get("duration_ms", 0.0))
        for item in perf.get("denoise_steps_ms", []) or []
    ) / 1000.0


def _literal_stats_from_log(log_path: Path, pattern: re.Pattern[str]) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    found: dict[str, Any] = {}
    for match in pattern.finditer(log_path.read_text(errors="replace")):
        try:
            found = ast.literal_eval(match.group(1))
        except Exception:
            continue
    return found


def _label(variant: str) -> str:
    labels = {
        "kwl": "KWL baseline",
        "kwl_teacache_c04_s6": "TeaCache t=0.04 start=6",
        "kwl_teacache_c06_s5": "TeaCache t=0.06 start=5",
        "kwl_teacache_c08_s5": "TeaCache t=0.08 start=5",
        "kwl_cache_teacache_c04_s6": "TeaCache t=0.04 start=6",
        "kwl_cache_teacache_c06_s5": "TeaCache t=0.06 start=5",
        "kwl_cache_teacache_c08_s5": "TeaCache t=0.08 start=5",
        "kwl_stage1_cache_core": "Stage1 cache-core",
    }
    return labels.get(variant, variant)


def _load_case(root: Path, pipeline: str, prompt_index: int, variant: str) -> dict[str, Any]:
    case_dir = root / pipeline / f"prompt_{prompt_index}" / variant
    semantics_name = "hq_semantics.json" if pipeline == "hq" else "nonhq_semantics.json"
    perf = _load_json(case_dir / "perf.json")
    sem = _load_json(case_dir / semantics_name)
    stage_times = _stage_seconds(perf)
    log_path = root / "logs" / f"{pipeline}_prompt{prompt_index}_{variant}.log"
    teacache_stats = _literal_stats_from_log(log_path, TEACACHE_RE)
    stage1_cache_stats = _literal_stats_from_log(log_path, STAGE1_CACHE_RE)
    return {
        "pipeline": pipeline,
        "prompt_index": prompt_index,
        "variant": variant,
        "label": _label(variant),
        "dir": str(case_dir),
        "video": str(case_dir / "out.mp4"),
        "perf_json": str(case_dir / "perf.json"),
        "semantics_json": str(case_dir / semantics_name),
        "log": str(log_path),
        "exists": (case_dir / "out.mp4").exists() and (case_dir / "perf.json").exists(),
        "total_s": float(perf.get("total_duration_ms", 0.0)) / 1000.0,
        "denoise_s": _denoise_seconds(perf),
        "stage1_s": stage_times.get("stage1"),
        "stage2_s": stage_times.get("stage2"),
        "denoise_step_count": len(perf.get("denoise_steps_ms", []) or []),
        "teacache_stats": teacache_stats,
        "stage1_cache_core_stats": stage1_cache_stats,
        "semantics": sem,
        "prompt": sem.get("prompt", ""),
    }


def _speedup(base: float | None, value: float | None) -> float | None:
    if not base or not value:
        return None
    return base / value


def _annotate_speedups(cases: list[dict[str, Any]]) -> None:
    if not cases:
        return
    base = cases[0]
    for case in cases:
        case["speedup_total"] = _speedup(base.get("total_s"), case.get("total_s"))
        case["speedup_denoise"] = _speedup(base.get("denoise_s"), case.get("denoise_s"))
        case["speedup_stage1"] = _speedup(base.get("stage1_s"), case.get("stage1_s"))
        case["speedup_stage2"] = _speedup(base.get("stage2_s"), case.get("stage2_s"))


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _teacache_short(case: dict[str, Any]) -> str:
    stats = case.get("teacache_stats") or {}
    if not stats:
        return "-"
    parts = []
    for stage in ("stage1", "stage2"):
        item = stats.get(stage)
        if not item:
            continue
        calls = item.get("calls", 0)
        hits = item.get("hits", 0)
        computes = item.get("computes", 0)
        skipped = item.get("skipped_steps", [])
        parts.append(
            f"{stage}: hits={hits}, computes={computes}, calls={calls}, skip_steps={skipped}"
        )
    return "; ".join(parts) if parts else "-"


def _markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "# LTX-2.3 Cache Benchmark",
        "",
        f"Root: `{summary['root']}`",
        "",
        "| Pipeline | Prompt | Variant | Total s | Denoise s | Stage1 s | Stage2 s | Total x | Denoise x | Stage1 x | TeaCache stats | Video |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for prompt_key, prompt_data in summary["per_prompt"].items():
        prompt_index = prompt_data["prompt_index"]
        for pipeline in ("hq", "nonhq"):
            for case in prompt_data["pipelines"].get(pipeline, []):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            pipeline,
                            str(prompt_index),
                            case["label"],
                            _fmt(case.get("total_s")),
                            _fmt(case.get("denoise_s")),
                            _fmt(case.get("stage1_s")),
                            _fmt(case.get("stage2_s")),
                            _fmt(case.get("speedup_total"), 3),
                            _fmt(case.get("speedup_denoise"), 3),
                            _fmt(case.get("speedup_stage1"), 3),
                            _teacache_short(case).replace("|", "\\|"),
                            f"`{case['video']}`",
                        ]
                    )
                    + " |"
                )
    lines.append("")
    return "\n".join(lines)


def _rel(root: Path, path: str) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path


def _html_report(summary: dict[str, Any], root: Path) -> str:
    rows = []
    for prompt_data in summary["per_prompt"].values():
        prompt_index = prompt_data["prompt_index"]
        for pipeline in ("hq", "nonhq"):
            for case in prompt_data["pipelines"].get(pipeline, []):
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(pipeline)}</td>"
                    f"<td>{prompt_index}</td>"
                    f"<td>{html.escape(case['label'])}</td>"
                    f"<td>{_fmt(case.get('total_s'))}</td>"
                    f"<td>{_fmt(case.get('denoise_s'))}</td>"
                    f"<td>{_fmt(case.get('stage1_s'))}</td>"
                    f"<td>{_fmt(case.get('stage2_s'))}</td>"
                    f"<td>{_fmt(case.get('speedup_total'), 3)}</td>"
                    f"<td>{_fmt(case.get('speedup_denoise'), 3)}</td>"
                    f"<td>{_fmt(case.get('speedup_stage1'), 3)}</td>"
                    f"<td>{html.escape(_teacache_short(case))}</td>"
                    f"<td><a href='{html.escape(_rel(root, case['video']))}'>video</a></td>"
                    "</tr>"
                )
    videos = []
    for prompt_data in summary["per_prompt"].values():
        prompt = prompt_data.get("prompt") or ""
        prompt_index = prompt_data["prompt_index"]
        for pipeline in ("hq", "nonhq"):
            compare = root / pipeline / f"prompt_{prompt_index}" / "compare.mp4"
            if compare.exists():
                videos.append(
                    "<section>"
                    f"<h2>{html.escape(pipeline.upper())} prompt {prompt_index}</h2>"
                    f"<p>{html.escape(prompt)}</p>"
                    f"<video src='{html.escape(_rel(root, str(compare)))}' controls muted loop></video>"
                    "</section>"
                )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LTX-2.3 Cache Benchmark</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #151515; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f3f3; text-align: left; }}
    video {{ width: min(100%, 1280px); background: #000; display: block; margin: 8px 0 28px; }}
    code {{ background: #f3f3f3; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>LTX-2.3 Cache Benchmark</h1>
  <p>Root: <code>{html.escape(summary['root'])}</code></p>
  <table>
    <thead>
      <tr><th>Pipeline</th><th>Prompt</th><th>Variant</th><th>Total s</th><th>Denoise s</th><th>Stage1 s</th><th>Stage2 s</th><th>Total x</th><th>Denoise x</th><th>Stage1 x</th><th>TeaCache stats</th><th>Video</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {''.join(videos)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--hq-variants", required=True)
    parser.add_argument("--nonhq-variants", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    hq_variants = _split_variants(args.hq_variants)
    nonhq_variants = _split_variants(args.nonhq_variants)
    per_prompt: dict[str, Any] = {}
    for prompt_index in range(args.prompt_count):
        hq_cases = [_load_case(root, "hq", prompt_index, variant) for variant in hq_variants]
        nonhq_cases = [
            _load_case(root, "nonhq", prompt_index, variant)
            for variant in nonhq_variants
        ]
        _annotate_speedups(hq_cases)
        _annotate_speedups(nonhq_cases)
        prompt = next(
            (case.get("prompt") for case in hq_cases + nonhq_cases if case.get("prompt")),
            "",
        )
        per_prompt[f"prompt_{prompt_index}"] = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "pipelines": {"hq": hq_cases, "nonhq": nonhq_cases},
        }

    summary = {
        "root": str(root),
        "hq_variants": hq_variants,
        "nonhq_variants": nonhq_variants,
        "prompt_count": args.prompt_count,
        "per_prompt": per_prompt,
        "notes": {
            "hq_pipeline": "LTX2TwoStageHQPipeline, 15 stage-1 steps, res2s sampler.",
            "nonhq_pipeline": "LTX2TwoStagePipeline, 30 stage-1 steps, euler sampler.",
            "teacache": "LTX2 residual replay skips the transformer block stack and still runs output norm/projection/unpatchify.",
        },
    }
    (root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (root / "benchmark_summary.md").write_text(_markdown_table(summary))
    (root / "benchmark_report.html").write_text(_html_report(summary, root))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
