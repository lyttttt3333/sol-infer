#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def gate_to_out(out, gate_logits, weight, bias):
    scaled = out * (2.0 * torch.sigmoid(gate_logits).unsqueeze(-1))
    return F.linear(scaled.reshape(*scaled.shape[:-2], -1), weight, bias)


def gate_to_out_residual(out, gate_logits, weight, bias, residual, output_gate):
    scaled = out * (2.0 * torch.sigmoid(gate_logits).unsqueeze(-1))
    projected = F.linear(scaled.reshape(*scaled.shape[:-2], -1), weight, bias)
    return torch.addcmul(residual, projected, output_gate)


def current_path(compiled_gate_to_out, out, gate_logits, weight, bias, residual, output_gate):
    projected = compiled_gate_to_out(out, gate_logits, weight, bias)
    return torch.addcmul(residual, projected, output_gate)


def sync():
    torch.cuda.synchronize()


def time_cuda(fn, warmup, iters):
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


def stats(xs):
    return {
        "avg_ms": float(sum(xs) / len(xs)),
        "median_ms": float(statistics.median(xs)),
        "min_ms": float(min(xs)),
        "max_ms": float(max(xs)),
        "samples_ms": xs,
    }


def bench_case(name, batch, tokens, heads, dim_head, hidden, warmup, iters):
    dtype = torch.bfloat16
    torch.manual_seed(20260531 + tokens)
    out = torch.randn((batch, tokens, heads, dim_head), device="cuda", dtype=dtype)
    gate_logits = torch.randn((batch, tokens, heads), device="cuda", dtype=dtype)
    weight = torch.randn((hidden, heads * dim_head), device="cuda", dtype=dtype) / ((heads * dim_head) ** 0.5)
    bias = torch.randn((hidden,), device="cuda", dtype=dtype)
    residual = torch.randn((batch, tokens, hidden), device="cuda", dtype=dtype)
    output_gate = torch.randn((batch, 1, hidden), device="cuda", dtype=dtype)

    compiled_gate_to_out = torch.compile(
        gate_to_out,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
    compiled_residual = torch.compile(
        gate_to_out_residual,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
    y0 = current_path(compiled_gate_to_out, out, gate_logits, weight, bias, residual, output_gate)
    y1 = compiled_residual(out, gate_logits, weight, bias, residual, output_gate)
    sync()
    diff = (y0.float() - y1.float()).abs()
    current_times = time_cuda(
        lambda: current_path(compiled_gate_to_out, out, gate_logits, weight, bias, residual, output_gate),
        warmup,
        iters,
    )
    fused_times = time_cuda(
        lambda: compiled_residual(out, gate_logits, weight, bias, residual, output_gate),
        warmup,
        iters,
    )
    cur = stats(current_times)
    fus = stats(fused_times)
    return {
        "name": name,
        "shape": {
            "batch": batch,
            "tokens": tokens,
            "heads": heads,
            "dim_head": dim_head,
            "hidden": hidden,
        },
        "current_compiled_gate_to_out_plus_addcmul": cur,
        "compiled_gate_to_out_residual": fus,
        "speedup_median": cur["median_ms"] / fus["median_ms"],
        "speedup_avg": cur["avg_ms"] / fus["avg_ms"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/kernel_fusion_benchmarks/ltx2_gate_to_out_residual_compile.json"))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "results": [
            bench_case("stage1_video_self_attn", 3, 15810, 32, 128, 4096, args.warmup, args.iters),
            bench_case("stage2_video_self_attn", 1, 63240, 32, 128, 4096, args.warmup, args.iters),
        ],
    }
    text = json.dumps(payload, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
