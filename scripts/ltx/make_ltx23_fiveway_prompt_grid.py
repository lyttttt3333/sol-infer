#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VARIANTS = [
    ("diffusers", "Diffusers BF16"),
    ("kernel_bf16", "Kernel BF16"),
    ("nvfp4", "NVFP4"),
    ("sparse_bf16", "Sparse BF16"),
    ("nvfp4_sparse", "Sparse+NVFP4"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_prompts(root: Path) -> list[dict[str, str]]:
    path = root / "prompts.json"
    if not path.exists():
        raise FileNotFoundError(f"missing prompts file: {path}")
    prompts = json.loads(path.read_text())
    if len(prompts) != 5:
        raise ValueError(f"expected 5 prompts, got {len(prompts)}")
    return prompts


def get_total_seconds(run_dir: Path, variant: str) -> float | None:
    if variant == "diffusers":
        perf = read_json(run_dir / "perf_diffusers.json")
        if "strict_pipeline_s" in perf:
            return float(perf["strict_pipeline_s"])
        timings = perf.get("timings_s", {})
        keys = ["actual.stage1_pipeline_s", "actual.stage2_pipeline_s", "actual.video_vae_decode_s"]
        if all(k in timings for k in keys):
            return float(sum(timings[k] for k in keys))
        return None
    summary = read_json(run_dir / "summary.json")
    if "total_s" in summary:
        return float(summary["total_s"])
    perf = read_json(run_dir / "perf.json")
    if "total_duration_ms" in perf:
        return float(perf["total_duration_ms"]) / 1000.0
    return None


def fit_frame(frame: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    y = (cell_h - new_h) // 2
    x = (cell_w - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def draw_label(frame: np.ndarray, label: str, row_label: str | None = None) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 12, 34
    cv2.putText(frame, label, (x, y), font, 0.78, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), font, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    if row_label:
        y2 = frame.shape[0] - 16
        cv2.putText(frame, row_label, (x, y2), font, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, row_label, (x, y2), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def open_captures(paths: list[Path]) -> list[cv2.VideoCapture]:
    caps = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        caps.append(cap)
    return caps


def release_all(caps: list[cv2.VideoCapture]) -> None:
    for cap in caps:
        cap.release()


def make_writer(out: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {out}")
    return writer


def compare_videos(a_path: Path, b_path: Path) -> dict[str, Any]:
    a = cv2.VideoCapture(str(a_path))
    b = cv2.VideoCapture(str(b_path))
    if not a.isOpened() or not b.isOpened():
        raise RuntimeError(f"could not open videos for comparison: {a_path}, {b_path}")

    frames = 0
    abs_sum = 0.0
    sq_sum = 0.0
    pixel_count = 0
    frame_mean_abs = []
    frame_psnr = []
    max_abs = 0
    while True:
        ok_a, fa = a.read()
        ok_b, fb = b.read()
        if not ok_a or not ok_b:
            break
        if fa.shape != fb.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]), interpolation=cv2.INTER_AREA)
        da = fa.astype(np.float32) - fb.astype(np.float32)
        abs_diff = np.abs(da)
        sq_diff = da * da
        mae = float(abs_diff.mean())
        mse = float(sq_diff.mean())
        frame_mean_abs.append(mae)
        frame_psnr.append(float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse)))
        abs_sum += float(abs_diff.sum())
        sq_sum += float(sq_diff.sum())
        pixel_count += int(abs_diff.size)
        max_abs = max(max_abs, int(abs_diff.max()))
        frames += 1
    a.release()
    b.release()

    if frames == 0 or pixel_count == 0:
        raise RuntimeError(f"no overlapping frames for comparison: {a_path}, {b_path}")
    mean_abs = abs_sum / pixel_count
    mse_all = sq_sum / pixel_count
    finite_psnr = [x for x in frame_psnr if math.isfinite(x)]
    return {
        "frames": frames,
        "mean_abs_pixel_diff": mean_abs,
        "p95_frame_mean_abs_pixel_diff": float(np.percentile(frame_mean_abs, 95)),
        "max_abs_pixel_diff": max_abs,
        "mean_psnr_db": float("inf") if mse_all == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse_all)),
        "min_frame_psnr_db": min(finite_psnr) if finite_psnr else float("inf"),
    }


def variant_paths(prompt_dir: Path) -> list[Path]:
    return [prompt_dir / key / "out.mp4" for key, _ in VARIANTS]


def variant_labels(prompt_dir: Path) -> list[str]:
    labels = []
    for key, label in VARIANTS:
        total_s = get_total_seconds(prompt_dir / key, key)
        if total_s is None:
            labels.append(label)
        else:
            labels.append(f"{label} {total_s:.1f}s")
    return labels


def make_prompt_fiveway(prompt_dir: Path, out: Path, cell_w: int, cell_h: int) -> dict[str, Any]:
    paths = variant_paths(prompt_dir)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    caps = open_captures(paths)
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 24.0
    writer = make_writer(out, fps, (cell_w * len(VARIANTS), cell_h))
    labels = variant_labels(prompt_dir)
    frames = 0
    while True:
        cells = []
        for cap, label in zip(caps, labels):
            ok, frame = cap.read()
            if not ok:
                release_all(caps)
                writer.release()
                return {"out": str(out), "frames": frames, "fps": fps, "size": [cell_w * len(VARIANTS), cell_h]}
            cell = fit_frame(frame, cell_w, cell_h)
            draw_label(cell, label)
            cells.append(cell)
        writer.write(np.concatenate(cells, axis=1))
        frames += 1


def make_combined_grid(root: Path, prompts: list[dict[str, str]], out: Path, cell_w: int, cell_h: int) -> dict[str, Any]:
    all_paths: list[Path] = []
    row_labels: list[str] = []
    cell_labels: list[str] = []
    for prompt in prompts:
        prompt_dir = root / prompt["slug"]
        all_paths.extend(variant_paths(prompt_dir))
        row_labels.extend([prompt["slug"]] * len(VARIANTS))
        cell_labels.extend(variant_labels(prompt_dir))
    for path in all_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    caps = open_captures(all_paths)
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 24.0
    width = cell_w * len(VARIANTS)
    height = cell_h * len(prompts)
    writer = make_writer(out, fps, (width, height))
    frames = 0
    while True:
        rows = []
        idx = 0
        for _prompt in prompts:
            cells = []
            for _variant in VARIANTS:
                cap = caps[idx]
                ok, frame = cap.read()
                if not ok:
                    release_all(caps)
                    writer.release()
                    return {"out": str(out), "frames": frames, "fps": fps, "size": [width, height]}
                cell = fit_frame(frame, cell_w, cell_h)
                draw_label(cell, cell_labels[idx], row_labels[idx])
                cells.append(cell)
                idx += 1
            rows.append(np.concatenate(cells, axis=1))
        writer.write(np.concatenate(rows, axis=0))
        frames += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cell-width", type=int, default=768)
    parser.add_argument("--cell-height", type=int, default=432)
    args = parser.parse_args()

    root = Path(args.root)
    prompts = load_prompts(root)
    consistency = {}
    fiveways = []
    for prompt in prompts:
        prompt_dir = root / prompt["slug"]
        consistency[prompt["slug"]] = compare_videos(
            prompt_dir / "diffusers" / "out.mp4",
            prompt_dir / "kernel_bf16" / "out.mp4",
        )
        fiveways.append(
            make_prompt_fiveway(
                prompt_dir,
                prompt_dir / "fiveway.mp4",
                args.cell_width,
                args.cell_height,
            )
        )

    combined = make_combined_grid(root, prompts, Path(args.out), args.cell_width, args.cell_height)
    report = {
        "root": str(root),
        "variants": [{"key": key, "label": label} for key, label in VARIANTS],
        "cell_size": [args.cell_width, args.cell_height],
        "combined": combined,
        "fiveways": fiveways,
        "diffusers_vs_kernel_bf16_consistency": consistency,
    }
    report_path = root / "fiveway_grid_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
