import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.jit_kernel.diffusion.triton.ltx2_gelu import (
    ltx2_bias_gelu_tanh_inplace,
    ltx2_gelu_tanh_inplace,
)


def _time_cuda(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        y = fn()
        torch.cuda.synchronize()
        del y
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        y = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del y
    return times


def _stats(times: list[float]) -> dict:
    return {
        "median_ms": float(statistics.median(times)),
        "avg_ms": float(sum(times) / len(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "samples_ms": times,
    }


def _bench_case(name: str, shape: tuple[int, int], repeats: int, warmup: int) -> dict:
    torch.manual_seed(20260528 + shape[0] + shape[1])
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((shape[1],), device="cuda", dtype=torch.bfloat16)
    ref = F.gelu(x + bias, approximate="tanh")
    cand = x.clone()
    ltx2_bias_gelu_tanh_inplace(cand, bias)
    diff = (ref.float() - cand.float()).abs()

    x_torch = x.clone()
    x_gelu = x.clone()
    x_bias_gelu = x.clone()

    torch_times = _time_cuda(
        lambda: F.gelu(x_torch + bias, approximate="tanh"),
        repeats=repeats,
        warmup=warmup,
    )
    inplace_two_kernel_times = _time_cuda(
        lambda: ltx2_gelu_tanh_inplace(x_gelu.add_(bias)),
        repeats=repeats,
        warmup=warmup,
    )
    fused_bias_gelu_times = _time_cuda(
        lambda: ltx2_bias_gelu_tanh_inplace(x_bias_gelu, bias),
        repeats=repeats,
        warmup=warmup,
    )
    torch_stats = _stats(torch_times)
    two_kernel_stats = _stats(inplace_two_kernel_times)
    fused_stats = _stats(fused_bias_gelu_times)
    return {
        "shape": {"name": name, "rows": shape[0], "cols": shape[1]},
        "torch_bias_gelu": torch_stats,
        "triton_add_bias_then_inplace_gelu": two_kernel_stats,
        "triton_fused_bias_gelu": fused_stats,
        "speedup_vs_torch": torch_stats["median_ms"] / fused_stats["median_ms"],
        "speedup_vs_two_kernel": two_kernel_stats["median_ms"] / fused_stats["median_ms"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-gelu-inplace/result.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cases = [
        ("stage1_video_ffn_proj_in", (3 * 15810, 16384)),
        ("stage2_video_ffn_proj_in", (63240, 16384)),
    ]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "results": {
            name: _bench_case(name, shape, args.repeats, args.warmup)
            for name, shape in cases
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
