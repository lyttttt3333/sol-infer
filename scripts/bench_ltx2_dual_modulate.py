import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.jit_kernel.diffusion.triton.ltx2_dual_modulate import (
    ltx2_rmsnorm_dual_modulate,
)
from sglang.jit_kernel.diffusion.triton.scale_shift import fuse_scale_shift_kernel


def time_cuda(fn, repeats: int, warmup: int):
    last = None
    for _ in range(warmup):
        last = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        last = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return last, times


def stats(times):
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "ms": times}


def bench(batch: int, seq: int, hidden: int, repeats: int, warmup: int):
    torch.manual_seed(123)
    x = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    scale0 = torch.randn((batch, 1, hidden), device="cuda", dtype=torch.bfloat16)
    shift0 = torch.randn((batch, 1, hidden), device="cuda", dtype=torch.bfloat16)
    scale1 = torch.randn((batch, 1, hidden), device="cuda", dtype=torch.bfloat16)
    shift1 = torch.randn((batch, 1, hidden), device="cuda", dtype=torch.bfloat16)

    def base_existing():
        normed = F.rms_norm(x, normalized_shape=(hidden,), eps=1e-6)
        return (
            fuse_scale_shift_kernel(normed, scale0, shift0, scale_constant=1.0),
            fuse_scale_shift_kernel(normed, scale1, shift1, scale_constant=1.0),
        )

    def fused():
        return ltx2_rmsnorm_dual_modulate(x, scale0, shift0, scale1, shift1, 1e-6)

    (b0, b1), base_times = time_cuda(base_existing, repeats, warmup)
    (f0, f1), fused_times = time_cuda(fused, repeats, warmup)
    return {
        "shape": {"batch": batch, "seq": seq, "hidden": hidden},
        "base_existing": stats(base_times),
        "fused": stats(fused_times),
        "speedup": (sum(base_times) / len(base_times))
        / (sum(fused_times) / len(fused_times)),
        "y0_max_abs_diff": float((b0.float() - f0.float()).abs().max().item()),
        "y1_max_abs_diff": float((b1.float() - f1.float()).abs().max().item()),
        "y0_mean_abs_diff": float((b0.float() - f0.float()).abs().mean().item()),
        "y1_mean_abs_diff": float((b1.float() - f1.float()).abs().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="outputs/ltx23-dual-modulate-microbench/result.json"
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "results": {
            "stage1_video": bench(3, 15810, 4096, args.repeats, args.warmup),
            "stage2_video": bench(1, 63240, 4096, args.repeats, args.warmup),
            "stage1_audio": bench(3, 251, 2048, args.repeats, args.warmup),
            "stage2_audio": bench(1, 251, 2048, args.repeats, args.warmup),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
