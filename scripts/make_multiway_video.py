#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_item(value: str) -> tuple[Path, str]:
    if "=" not in value:
        return Path(value), Path(value).stem
    label, path = value.split("=", 1)
    return Path(path), label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", action="append", required=True, help="label=video.mp4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cols", type=int, default=0)
    parser.add_argument("--tile-width", type=int, default=768)
    parser.add_argument("--tile-height", type=int, default=512)
    args = parser.parse_args()

    items = [parse_item(v) for v in args.item]
    caps = []
    for path, label in items:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {label}: {path}")
        caps.append(cap)
    fps = next((cap.get(cv2.CAP_PROP_FPS) for cap in caps if cap.get(cv2.CAP_PROP_FPS)), 24.0) or 24.0
    cols = args.cols or len(caps)
    rows = (len(caps) + cols - 1) // cols
    tile_w = args.tile_width
    tile_h = args.tile_height
    out_w = cols * tile_w
    out_h = rows * tile_h
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.out}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0
    while True:
        frames = []
        ok_any = False
        for cap in caps:
            ok, frame = cap.read()
            if ok:
                ok_any = True
                if frame.shape[1] != tile_w or frame.shape[0] != tile_h:
                    frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            else:
                frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            frames.append(frame)
        if not ok_any:
            break
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for idx, ((_, label), frame) in enumerate(zip(items, frames)):
            row = idx // cols
            col = idx % cols
            y = row * tile_h
            x = col * tile_w
            canvas[y : y + tile_h, x : x + tile_w] = frame
            cv2.putText(canvas, label, (x + 18, y + 42), font, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(canvas, label, (x + 18, y + 42), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
        frame_count += 1
    writer.release()
    for cap in caps:
        cap.release()
    print(f"wrote {args.out} frames={frame_count} fps={fps} size={out_w}x{out_h}")


if __name__ == "__main__":
    main()
