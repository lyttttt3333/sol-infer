#!/usr/bin/env python3
"""Benchmark TE NVFP4 FFN proj_out tail fusion for LTX video FFN.

Baseline mirrors the current runtime path:
  TE NVFP4 Linear(includes bias) -> torch.addcmul(residual, update, gate)

Candidate mirrors the new experimental runtime path:
  TE NVFP4 Linear(return_bias=True) -> Triton(update + bias) * gate + residual
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from sglang.jit_kernel.diffusion.triton.ltx2_gelu import ltx2_bias_residual_gate


def sync() -> None:
    torch.cuda.synchronize()


def pad_to_te_m(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, int]:
    original_m = int(x.shape[0])
    pad_rows = (-original_m) % multiple
    if pad_rows:
        return torch.nn.functional.pad(x, (0, 0, 0, pad_rows)), original_m
    return x, original_m


def make_te_pair(dtype: torch.dtype):
    import transformer_engine.pytorch as te

    base = te.Linear(16384, 4096, bias=True, params_dtype=dtype, device="cuda")
    rb = te.Linear(
        16384,
        4096,
        bias=True,
        return_bias=True,
        params_dtype=dtype,
        device="cuda",
    )
    torch.nn.init.normal_(base.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(base.bias, mean=0.0, std=0.02)
    rb.weight = base.weight
    rb.bias = base.bias
    base.eval()
    rb.eval()
    return base, rb


def time_cuda(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        y = fn()
    sync()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        y = fn()
        sync()
        if y.numel() == 0:
            raise RuntimeError("empty output")
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def stats(xs: list[float]) -> dict[str, float | list[float]]:
    return {
        "avg_ms": float(sum(xs) / len(xs)),
        "median_ms": float(statistics.median(xs)),
        "min_ms": float(min(xs)),
        "max_ms": float(max(xs)),
        "samples_ms": xs,
    }


def run_case(name: str, batch: int, tokens: int, dtype: torch.dtype, warmup: int, iters: int) -> dict[str, Any]:
    from transformer_engine.common.recipe import NVFP4BlockScaling
    from transformer_engine.pytorch import fp8_autocast

    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
    )
    m = batch * tokens
    x = torch.randn((m, 16384), device="cuda", dtype=dtype)
    residual = torch.randn((batch, tokens, 4096), device="cuda", dtype=dtype)
    gate = torch.randn((batch, 1, 4096), device="cuda", dtype=dtype)
    baseline_layer, return_bias_layer = make_te_pair(dtype)

    def baseline():
        x_pad, original_m = pad_to_te_m(x)
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            update = baseline_layer(x_pad)
        if int(update.shape[0]) != original_m:
            update = update[:original_m]
        update = update.reshape(batch, tokens, 4096)
        return torch.addcmul(residual, update, gate)

    def candidate():
        x_pad, original_m = pad_to_te_m(x)
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            update, bias = return_bias_layer(x_pad)
        if int(update.shape[0]) != original_m:
            update = update[:original_m]
        update = update.reshape(batch, tokens, 4096)
        return ltx2_bias_residual_gate(update, residual, gate, bias)

    y0 = baseline()
    y1 = candidate()
    sync()
    diff = (y0.float() - y1.float()).abs()
    baseline_times = time_cuda(baseline, warmup, iters)
    candidate_times = time_cuda(candidate, warmup, iters)
    b = stats(baseline_times)
    c = stats(candidate_times)
    return {
        "name": name,
        "batch": batch,
        "tokens": tokens,
        "m": m,
        "dtype": str(dtype).replace("torch.", ""),
        "baseline_te_bias_addcmul": b,
        "candidate_te_return_bias_triton_tail": c,
        "speedup_median": b["median_ms"] / c["median_ms"],
        "speedup_avg": b["avg_ms"] / c["avg_ms"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--output", type=Path, default=Path("outputs/kernel_fusion_benchmarks/te_nvfp4_proj_out_bias_gate.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(20260531)
    torch.cuda.manual_seed_all(20260531)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "results": [
            run_case("stage1_video_ffn_proj_out", 3, 15810, dtype, args.warmup, args.iters),
            run_case("stage2_video_ffn_proj_out", 1, 63240, dtype, args.warmup, args.iters),
        ],
    }
    text = json.dumps(payload, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
