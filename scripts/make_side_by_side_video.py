import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--left-label', default='BF16 baseline')
    parser.add_argument('--right-label', default='NVFP4')
    args = parser.parse_args()

    left = cv2.VideoCapture(args.left)
    right = cv2.VideoCapture(args.right)
    if not left.isOpened():
        raise RuntimeError(f'Could not open left video: {args.left}')
    if not right.isOpened():
        raise RuntimeError(f'Could not open right video: {args.right}')

    fps = left.get(cv2.CAP_PROP_FPS) or right.get(cv2.CAP_PROP_FPS) or 24.0
    lw = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    lh = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rw = int(right.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh = int(right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if lw <= 0 or lh <= 0 or rw <= 0 or rh <= 0:
        raise RuntimeError('Invalid video dimensions')

    target_h = min(lh, rh)
    target_w = min(lw, rw)
    out_w = target_w * 2
    out_h = target_h
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open writer: {args.out}')

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0
    while True:
        ok_l, frame_l = left.read()
        ok_r, frame_r = right.read()
        if not ok_l or not ok_r:
            break
        if frame_l.shape[0] != target_h or frame_l.shape[1] != target_w:
            frame_l = cv2.resize(frame_l, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if frame_r.shape[0] != target_h or frame_r.shape[1] != target_w:
            frame_r = cv2.resize(frame_r, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frame = np.concatenate([frame_l, frame_r], axis=1)
        for x, label in [(24, args.left_label), (target_w + 24, args.right_label)]:
            cv2.putText(frame, label, (x, 48), font, 1.25, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, label, (x, 48), font, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_count += 1

    writer.release()
    left.release()
    right.release()
    print(f'wrote {args.out} frames={frame_count} fps={fps} size={out_w}x{out_h}')


if __name__ == '__main__':
    main()
