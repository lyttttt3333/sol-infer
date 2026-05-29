#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROMPT = (
    "A cinematic 10 second aerial shot of an antique brass clockwork train crossing "
    "a snowy mountain bridge at sunrise, steam drifting through golden light, "
    "smooth camera movement, high detail"
)

VARIANTS = [
    ("diffusers", "Diffusers"),
    ("kwl", "KWL"),
    ("kwl_sparse_stage1", "KWL+Sparse S1"),
    ("kwl_sparse_stage2", "KWL+Sparse S2"),
    ("kwl_sparse_stage1_stage2", "KWL+Sparse S1+S2"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def make_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for key, label in VARIANTS:
        summary = read_json(root / key / "summary.json")
        row = {
            "variant": key,
            "label": label,
            "total_s": float(summary["total_s"]),
            "output_video": str(root / key / "out.mp4"),
        }
        if key == "diffusers":
            row.update(
                {
                    "stage1_s": summary.get("stage1_pipeline_s"),
                    "stage2_s": summary.get("stage2_pipeline_s"),
                    "decode_s": summary.get("decode_s"),
                }
            )
        else:
            row.update(
                {
                    "stage1_s": summary.get("denoise_s"),
                    "stage2_s": summary.get("refine_s"),
                    "decode_s": summary.get("decode_s"),
                }
            )
        rows.append(row)

    kwl_total = next(row["total_s"] for row in rows if row["variant"] == "kwl")
    for row in rows:
        row["speedup_vs_kwl"] = kwl_total / row["total_s"] if row["total_s"] else None
    return rows


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    aggregate = {
        "root": str(root),
        "prompt": PROMPT,
        "warmup": {"enabled": True, "steps": 30},
        "resolution": {"width": 1920, "height": 1088, "num_frames": 241, "fps": 24},
        "piecewise": {
            "sparsity": "0.9",
            "block_size": 64,
            "only_video_self": True,
            "approx_remainder": True,
            "route_mode": "score",
        },
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    lines = [
        "# LTX2.3 Train Valley 5-Way",
        "",
        f"Prompt: {PROMPT}",
        "",
        "Warmup: enabled, 30 steps before measured run.",
        "",
        "| Variant | E2E s | Stage 1 s | Stage 2 s | Decode s | Speedup vs KWL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        speedup = row["speedup_vs_kwl"]
        speedup_text = f"{speedup:.3f}x" if speedup else "n/a"
        lines.append(
            "| {label} | {total_s:.3f} | {stage1_s:.3f} | {stage2_s:.3f} | "
            "{decode_s:.3f} | {speedup} |".format(
                label=row["label"],
                total_s=row["total_s"],
                stage1_s=row["stage1_s"],
                stage2_s=row["stage2_s"],
                decode_s=row["decode_s"],
                speedup=speedup_text,
            )
        )
    (root / "table.md").write_text("\n".join(lines) + "\n")


def fit_frame(frame: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    x = (cell_w - new_w) // 2
    y = (cell_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def draw_label(cell: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 12, 34
    cv2.putText(cell, text, (x, y), font, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(cell, text, (x, y), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def make_video(root: Path, rows: list[dict[str, Any]]) -> None:
    cv2.setNumThreads(1)
    paths = [root / key / "out.mp4" for key, _ in VARIANTS]
    caps = [cv2.VideoCapture(str(path)) for path in paths]
    for path, cap in zip(paths, caps):
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")

    fps = caps[0].get(cv2.CAP_PROP_FPS) or 24.0
    cell_w, cell_h = 640, 362
    out = root / "train-valley-5way.mp4"
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cell_w * len(VARIANTS), cell_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {out}")

    frames = 0
    try:
        while True:
            cells = []
            for cap, row in zip(caps, rows):
                ok, frame = cap.read()
                if not ok:
                    print(
                        f"wrote {out} frames={frames} fps={fps} "
                        f"size={cell_w * len(VARIANTS)}x{cell_h}"
                    )
                    return
                cell = fit_frame(frame, cell_w, cell_h)
                draw_label(cell, f"{row['label']}  {row['total_s']:.1f}s")
                cells.append(cell)
            writer.write(np.concatenate(cells, axis=1))
            frames += 1
    finally:
        writer.release()
        for cap in caps:
            cap.release()


def main() -> None:
    root = Path("outputs/ltx23-train-valley-fiveway-1080p10s")
    rows = make_rows(root)
    write_summary(root, rows)
    make_video(root, rows)


if __name__ == "__main__":
    main()
