#!/usr/bin/env python3
"""Compare official and SGLang LTX2 stage2 debug dumps."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch


def _chunked_stats(x: torch.Tensor, y: torch.Tensor, chunk_rows: int) -> tuple[float, float, int, tuple[int, int, float, float] | None]:
    xf = x.reshape(-1, x.shape[-1])
    yf = y.reshape(-1, y.shape[-1])
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    num_diff = 0
    first_diff: tuple[int, int, float, float] | None = None

    for start in range(0, xf.shape[0], chunk_rows):
        xc = xf[start : start + chunk_rows]
        yc = yf[start : start + chunk_rows]
        if torch.equal(xc, yc):
            count += xc.numel()
            continue

        ne = xc != yc
        if first_diff is None:
            nz = ne.nonzero(as_tuple=False)
            if nz.numel():
                row = int(start + nz[0, 0])
                col = int(nz[0, 1])
                first_diff = (row, col, float(xc[nz[0, 0], nz[0, 1]]), float(yc[nz[0, 0], nz[0, 1]]))
        num_diff += int(ne.sum())
        d = (xc.float() - yc.float()).abs()
        max_abs = max(max_abs, float(d.max()))
        sum_abs += float(d.sum())
        count += xc.numel()

    return max_abs, sum_abs / max(count, 1), num_diff, first_diff


def _print_key(name: str, official: dict, sglang: dict, chunk_rows: int) -> None:
    x = official.get(name)
    y = sglang.get(name)
    if x is None or y is None:
        print(f"{name}: missing official={x is None} sglang={y is None}", flush=True)
        return

    if torch.equal(x, y):
        print(f"{name}: exact True shape={tuple(x.shape)} dtype={x.dtype}", flush=True)
        return

    max_abs, mean_abs, num_diff, first = _chunked_stats(x, y, chunk_rows)
    print(
        f"{name}: exact False official_shape={tuple(x.shape)} sglang_shape={tuple(y.shape)} "
        f"official_dtype={x.dtype} sglang_dtype={y.dtype} max={max_abs} mean={mean_abs} "
        f"num_diff={num_diff} first={first}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["video_latents_in", "audio_latents_in", "video_denoised", "audio_denoised"],
    )
    args = parser.parse_args()

    for call_idx in range(args.calls):
        print(f"\nCALL {call_idx}", flush=True)
        official = torch.load(
            args.official_dir / f"official_stage2_denoiser_call_{call_idx:02d}.pt",
            map_location="cpu",
            mmap=False,
        )
        sglang = torch.load(
            args.sglang_dir / f"sglang_stage2_denoiser_call_{call_idx:02d}.pt",
            map_location="cpu",
            mmap=False,
        )
        print(f"call_index: {official.get('call_index')} {sglang.get('call_index')}", flush=True)
        print(f"step_index: {official.get('step_index')} {sglang.get('step_index')}", flush=True)
        if "sigmas" in official:
            print(f"official sigmas: {official['sigmas']}", flush=True)
        if "sigma" in sglang:
            print(f"sglang sigma: {sglang['sigma']}", flush=True)
        for key in args.keys:
            _print_key(key, official, sglang, args.chunk_rows)
        del official, sglang
        gc.collect()


if __name__ == "__main__":
    main()
