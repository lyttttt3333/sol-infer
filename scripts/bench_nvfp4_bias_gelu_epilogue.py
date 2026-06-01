#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.jit_kernel.nvfp4 import (
    cutlass_scaled_fp4_mm,
    cutlass_scaled_fp4_mm_bias_gelu,
    scaled_fp4_quant,
)
from sglang.jit_kernel.diffusion.triton.ltx2_gelu import ltx2_bias_gelu_tanh_inplace


def _sync() -> None:
    torch.cuda.synchronize()


def _time_ms(fn, iters: int) -> list[float]:
    out = None
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    if out is not None:
        # Keep one observable use so Python cannot drop references too early.
        _ = float(out.flatten()[0].float().cpu())
    return times


def _summary(xs: list[float]) -> dict[str, float | list[float]]:
    return {
        "avg_ms": statistics.mean(xs),
        "median_ms": statistics.median(xs),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "samples_ms": xs,
    }


def run_case(name: str, m: int, n: int, k: int, dtype: torch.dtype, warmup: int, iters: int) -> dict[str, object]:
    torch.manual_seed(1234)
    device = "cuda"
    a = torch.randn((m, k), device=device, dtype=dtype) * 0.5
    b = torch.randn((n, k), device=device, dtype=dtype) * 0.5
    bias = torch.randn((n,), device=device, dtype=dtype) * 0.1
    a_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    b_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    alpha = torch.tensor([1.0], device=device, dtype=torch.float32)

    a_fp4, a_sf = scaled_fp4_quant(a, a_scale)
    b_fp4, b_sf = scaled_fp4_quant(b, b_scale)

    def baseline():
        y = cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, dtype)
        return ltx2_bias_gelu_tanh_inplace(y, bias)

    def fused():
        return cutlass_scaled_fp4_mm_bias_gelu(a_fp4, b_fp4, a_sf, b_sf, alpha, bias, dtype)

    for _ in range(warmup):
        baseline()
        fused()
    _sync()

    y0 = baseline()
    y1 = fused()
    _sync()
    diff = (y0.float() - y1.float()).abs()

    base_times = _time_ms(baseline, iters)
    fused_times = _time_ms(fused, iters)
    base_med = statistics.median(base_times)
    fused_med = statistics.median(fused_times)
    return {
        "name": name,
        "shape": {"m": m, "n": n, "k": k, "dtype": str(dtype).replace("torch.", "")},
        "baseline_cutlass_plus_triton_bias_gelu": _summary(base_times),
        "fused_cutlass_bias_gelu_epilogue": _summary(fused_times),
        "speedup_median": base_med / fused_med if fused_med else None,
        "speedup_avg": statistics.mean(base_times) / statistics.mean(fused_times),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "ltx", "ltx_sweep"], default="smoke")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("outputs/kernel_fusion_benchmarks/nvfp4_bias_gelu_epilogue.json"))
    args = parser.parse_args()

    if args.mode == "smoke":
        cases = [("smoke", 256, 256, 256)]
    elif args.mode == "ltx":
        cases = [
            ("stage1_video_ffn_proj_in", 3 * 15810, 16384, 4096),
            ("stage2_video_ffn_proj_in", 63240, 16384, 4096),
        ]
    else:
        cases = [
            (f"ltx_proj_in_m{m}", m, 16384, 4096)
            for m in (1024, 2048, 4096, 8192, 16384, 32768)
        ]

    results = []
    for name, m, n, k in cases:
        try:
            results.append(run_case(name, m, n, k, torch.bfloat16, args.warmup, args.iters))
        except Exception as exc:
            results.append({
                "name": name,
                "shape": {"m": m, "n": n, "k": k, "dtype": "bfloat16"},
                "error": repr(exc),
            })
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "mode": args.mode,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
