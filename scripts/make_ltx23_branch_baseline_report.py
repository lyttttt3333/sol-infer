#!/usr/bin/env python3
"""Create report and side-by-side video for LTX2 branch-baseline same-noise runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

DEFAULT_ROOT = "outputs/ltx23-branch-baselines-same-noise-1080p10s"
DEFAULT_AUDIT = "outputs/ltx_branch_audit/branch_baseline_report.md"

VARIANTS = [
    {
        "name": "diffusers_corrected_oldlora",
        "label": "Diffusers",
        "source": "official Diffusers + local scheduler reset",
        "kind": "diffusers",
        "notes": "Stage 1 multi-step; stage 2 distilled LoRA refine; scheduler reset; old local distilled LoRA.",
    },
    {
        "name": "sglang_dense_main",
        "label": "Dense",
        "source": "origin/main dense SGLang baseline",
        "kind": "sglang",
        "notes": "No KWL, no sparse attention, no NVFP4.",
    },
    {
        "name": "kwl_fusion_report",
        "label": "KWL",
        "source": "ltx2-dit-fusion-report KWL setting",
        "kind": "sglang",
        "notes": "Kernel-wise lossless fusion envs enabled.",
    },
    {
        "name": "sparse_bringup_piecewise",
        "label": "Sparse",
        "source": "ltx-sparse-attn-bringup setting",
        "kind": "sglang",
        "notes": "Piecewise sparse attention, sparsity=0.9, block=64, video self-attention only.",
    },
    {
        "name": "stage1_sparse_schedule",
        "label": "S1 Sched",
        "source": "ltx-stage1-sparse-schedule setting",
        "kind": "sglang",
        "notes": "Stage 1 first 5 steps dense, then sparsity ramps 0.8 to 0.9; stage 2 sparse 0.9.",
    },
    {
        "name": "nvfp4_piecewise",
        "label": "NVFP4+Sparse",
        "source": "ltx2-nvfp4-two-stage-cleanup + local fused FP4 setting",
        "kind": "sglang",
        "notes": "Selective NVFP4 video attention/FFN transformer overrides plus piecewise attention.",
    },
]

PAIR_NAMES = [
    ("stage1_video_initial", "diffusers_stage1_video_initial.pt", "sglang_stage1_video_initial.pt"),
    ("stage1_audio_initial", "diffusers_stage1_audio_initial.pt", "sglang_stage1_audio_initial.pt"),
    ("stage2_video_noise", "diffusers_stage2_video_noise.pt", "sglang_stage2_video_noise.pt"),
    ("stage2_audio_noise", "diffusers_stage2_audio_noise.pt", "sglang_stage2_audio_noise.pt"),
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    return payload["latents"]


def compare_latents(shared_dir: Path, variant_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    latent_dir = variant_dir / "latents"
    for key, shared_name, sglang_name in PAIR_NAMES:
        shared_path = shared_dir / shared_name
        sglang_path = latent_dir / sglang_name
        if not shared_path.exists() or not sglang_path.exists():
            result[key] = {
                "available": False,
                "shared_path": str(shared_path),
                "sglang_path": str(sglang_path),
            }
            continue
        a = load_latent(shared_path)
        b = load_latent(sglang_path)
        a_cast = a.to(dtype=b.dtype)
        delta = (a_cast.float() - b.float()).abs()
        raw_delta = (a.float() - b.float()).abs()
        result[key] = {
            "available": True,
            "shape_equal": list(a.shape) == list(b.shape),
            "diffusers_dtype": str(a.dtype),
            "sglang_dtype": str(b.dtype),
            "max_abs_after_cast": float(delta.max().item()),
            "mean_abs_after_cast": float(delta.mean().item()),
            "max_abs_raw": float(raw_delta.max().item()),
            "mean_abs_raw": float(raw_delta.mean().item()),
        }
    return result


def timing_for_variant(root: Path, spec: dict[str, str]) -> dict[str, Any] | None:
    vdir = root / spec["name"]
    video = vdir / "out.mp4"
    if not video.exists():
        return None
    row: dict[str, Any] = {
        "variant": spec["name"],
        "label": spec["label"],
        "branch_source": spec["source"],
        "notes": spec["notes"],
        "output_video": str(video),
    }
    if spec["kind"] == "diffusers":
        data = load_json(vdir / "perf_diffusers.json") or load_json(vdir / "summary.json")
        if data is None:
            return row
        timings = data.get("timings_s", {})
        row.update(
            {
                "total_s": data.get("strict_pipeline_s", data.get("total_s")),
                "stage1_s": timings.get("actual.stage1_pipeline_s", data.get("stage1_pipeline_s")),
                "stage2_s": timings.get("actual.stage2_pipeline_s", data.get("stage2_pipeline_s")),
                "decode_s": timings.get("actual.video_vae_decode_s", data.get("decode_s")),
            }
        )
    else:
        data = load_json(vdir / "summary.json")
        if data is None:
            data = load_json(vdir / "perf.json") or {}
            steps = {s.get("name"): s.get("duration_ms", 0.0) / 1000.0 for s in data.get("steps", [])}
            row.update(
                {
                    "total_s": data.get("total_duration_ms", 0.0) / 1000.0 if data else None,
                    "stage1_s": steps.get("LTX2AVDenoisingStage"),
                    "stage2_s": steps.get("LTX2RefinementStage"),
                    "decode_s": steps.get("LTX2AVDecodingStage"),
                }
            )
        else:
            row.update(
                {
                    "total_s": data.get("total_s"),
                    "stage1_s": data.get("denoise_s"),
                    "stage2_s": data.get("refine_s"),
                    "decode_s": data.get("decode_s"),
                }
            )
    return row


def draw_label(frame: np.ndarray, text: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def make_multiway_video(rows: list[dict[str, Any]], out_path: Path, cell_w: int = 512) -> dict[str, Any]:
    videos = [Path(row["output_video"]) for row in rows]
    caps = [cv2.VideoCapture(str(path)) for path in videos]
    if not all(cap.isOpened() for cap in caps):
        bad = [str(videos[i]) for i, cap in enumerate(caps) if not cap.isOpened()]
        raise RuntimeError(f"failed to open videos: {bad}")
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0 for cap in caps))
    src_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    src_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT) or 1088)
    cell_h = max(1, int(round(cell_w * src_h / src_w)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cell_w * len(caps), cell_h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {out_path}")
    written = 0
    for _ in range(frame_count):
        cells = []
        for cap, row in zip(caps, rows):
            ok, frame = cap.read()
            if not ok:
                break
            cell = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            total = row.get("total_s")
            label = f"{row['label']} {total:.1f}s" if isinstance(total, (int, float)) else row["label"]
            draw_label(cell, label)
            cells.append(cell)
        if len(cells) != len(caps):
            break
        writer.write(np.concatenate(cells, axis=1))
        written += 1
    writer.release()
    for cap in caps:
        cap.release()
    return {"path": str(out_path), "fps": fps, "frames": written, "width": cell_w * len(rows), "height": cell_h}


def write_markdown(root: Path, audit_path: Path, rows: list[dict[str, Any]], alignment: dict[str, Any], video_info: dict[str, Any] | None) -> Path:
    report_path = root / "final_branch_baseline_report.md"
    lines: list[str] = []
    lines.append("# LTX Branch Same-Noise Baseline Report")
    lines.append("")
    if audit_path.exists():
        lines.append("## Branch Audit")
        lines.append("")
        lines.append(f"Static audit source: `{audit_path}`")
        lines.append("")
    lines.append("## Runtime Results")
    lines.append("")
    lines.append("| Variant | Branch/source | Total s | Stage 1 s | Stage 2 s | Decode s | Notes |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        def fmt(value: Any) -> str:
            return f"{value:.3f}" if isinstance(value, (int, float)) else "NA"
        lines.append(
            f"| `{row['variant']}` | {row['branch_source']} | {fmt(row.get('total_s'))} | "
            f"{fmt(row.get('stage1_s'))} | {fmt(row.get('stage2_s'))} | {fmt(row.get('decode_s'))} | {row['notes']} |"
        )
    lines.append("")
    lines.append("## Same-Noise Verification")
    lines.append("")
    lines.append("Diffusers dumps the shared stage1 video/audio initial latents and stage2 video/audio noise. SGLang variants load those tensors and dump their received tensors for verification.")
    lines.append("")
    lines.append("| Variant | Pairs available | Max abs after bf16 cast | Status |")
    lines.append("| --- | ---: | ---: | --- |")
    for variant, checks in alignment.items():
        available = [v for v in checks.values() if v.get("available")]
        if available:
            max_abs = max(v.get("max_abs_after_cast", float("inf")) for v in available)
            status = "pass" if max_abs == 0.0 and all(v.get("shape_equal") for v in available) else "check"
            lines.append(f"| `{variant}` | {len(available)}/4 | {max_abs:.6g} | {status} |")
        else:
            lines.append(f"| `{variant}` | 0/4 | NA | missing dumps |")
    lines.append("")
    if video_info is not None:
        lines.append("## Side-By-Side Video")
        lines.append("")
        lines.append(f"Video: `{video_info['path']}`")
        lines.append(f"Shape: `{video_info['width']}x{video_info['height']}`, frames: `{video_info['frames']}`, fps: `{video_info['fps']:.3f}`")
        lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['variant']}`: `{row['output_video']}`")
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--audit-report", default=DEFAULT_AUDIT)
    parser.add_argument("--output-video", default="")
    args = parser.parse_args()

    root = Path(args.root)
    audit_path = Path(args.audit_report)
    rows = [row for spec in VARIANTS if (row := timing_for_variant(root, spec)) is not None]
    if not rows:
        raise SystemExit(f"No completed variant outputs found under {root}")

    shared_dir = root / "shared_noise"
    alignment: dict[str, Any] = {}
    for row in rows:
        if row["variant"] == "diffusers_corrected_oldlora":
            continue
        alignment[row["variant"]] = compare_latents(shared_dir, root / row["variant"])
    (root / "same_noise_alignment.json").write_text(json.dumps(alignment, indent=2, sort_keys=True) + "\n")

    video_info = None
    if len(rows) >= 2:
        out_video = Path(args.output_video) if args.output_video else root / "branch-baselines-same-noise-multiway.mp4"
        video_info = make_multiway_video(rows, out_video)
    summary = {
        "root": str(root),
        "rows": rows,
        "same_noise_alignment": alignment,
        "side_by_side_video": video_info,
    }
    (root / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_path = write_markdown(root, audit_path, rows, alignment, video_info)
    print(json.dumps({"summary": str(root / "final_summary.json"), "report": str(report_path), "video": video_info}, indent=2))


if __name__ == "__main__":
    main()
