#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang.jit_kernel.nvfp4 import (
    cutlass_scaled_fp4_mm,
    cutlass_scaled_fp4_mm_per_col_residual_gate,
    scaled_fp4_quant,
)
from sglang.jit_kernel.diffusion.triton.ltx2_gelu import ltx2_bias_residual_gate


def _sync() -> None:
    torch.cuda.synchronize()


def _time_ms(fn, iters: int) -> list[float]:
    out = None
    xs = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        xs.append((time.perf_counter() - t0) * 1000.0)
    if out is not None:
        _ = float(out.flatten()[0].float().cpu())
    return xs


def _summary(xs: list[float]) -> dict[str, object]:
    return {
        "avg_ms": statistics.mean(xs),
        "median_ms": statistics.median(xs),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "samples_ms": xs,
    }


def run_case(name: str, m: int, n: int, k: int, dtype: torch.dtype, warmup: int, iters: int) -> dict[str, object]:
    torch.manual_seed(5678)
    device = "cuda"
    a = torch.randn((m, k), device=device, dtype=dtype) * 0.5
    b = torch.randn((n, k), device=device, dtype=dtype) * 0.5
    residual = torch.randn((m, n), device=device, dtype=dtype) * 0.5
    gate = torch.randn((n,), device=device, dtype=dtype) * 0.1
    bias = torch.randn((n,), device=device, dtype=dtype) * 0.1
    bias_gate = (bias.float() * gate.float()).to(dtype).contiguous()
    one = torch.tensor([1.0], device=device, dtype=torch.float32)

    a_fp4, a_sf = scaled_fp4_quant(a, one)
    b_fp4, b_sf = scaled_fp4_quant(b, one)

    def baseline():
        y = cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, one, dtype)
        return ltx2_bias_residual_gate(y.view(1, m, n), residual.view(1, m, n), gate.view(1, n), bias).view(m, n)

    def fused():
        return cutlass_scaled_fp4_mm_per_col_residual_gate(
            a_fp4, b_fp4, a_sf, b_sf, one, residual, gate, bias_gate, dtype
        )

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
    return {
        "name": name,
        "shape": {"m": m, "n": n, "k": k, "dtype": str(dtype).replace("torch.", "")},
        "baseline_cutlass_plus_triton_bias_residual_gate": _summary(base_times),
        "fused_cutlass_per_col_residual_gate_epilogue": _summary(fused_times),
        "speedup_median": statistics.median(base_times) / statistics.median(fused_times),
        "speedup_avg": statistics.mean(base_times) / statistics.mean(fused_times),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "ltx"], default="smoke")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("outputs/kernel_fusion_benchmarks/nvfp4_proj_out_residual_gate_epilogue.json"))
    args = parser.parse_args()
    if args.mode == "smoke":
        cases = [("smoke", 512, 4096, 16384)]
    else:
        cases = [
            ("stage2_video_ffn_proj_out", 63240, 4096, 16384),
            ("stage2_video_attn_to_out", 63240, 4096, 4096),
        ]
    results = []
    for case in cases:
        try:
            results.append(run_case(*case, dtype=torch.bfloat16, warmup=args.warmup, iters=args.iters))
        except Exception as exc:
            name, m, n, k = case
            results.append({"name": name, "shape": {"m": m, "n": n, "k": k}, "error": repr(exc)})
    payload = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "mode": args.mode, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
