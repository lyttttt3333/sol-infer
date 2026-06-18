import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.triton.ltx2_gelu import ltx2_bias_residual_gate


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


def _stats(times):
    return {
        "median_ms": float(statistics.median(times)),
        "avg_ms": float(sum(times)/len(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "samples_ms": times,
    }


def _bench_case(name, shape, repeats, warmup):
    torch.manual_seed(20260528 + shape[0] + shape[1] + shape[2])
    update = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((shape[0], 1, shape[2]), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((shape[2],), device="cuda", dtype=torch.bfloat16)
    ref = torch.addcmul(residual, update + bias, gate)
    cand = ltx2_bias_residual_gate(update, residual, gate, bias)
    diff = (ref.float() - cand.float()).abs()

    def torch_two_kernel():
        u = update + bias
        return torch.addcmul(residual, u, gate)

    def torch_inplace_bias_then_addcmul():
        u = update.clone()
        u.add_(bias)
        return torch.addcmul(residual, u, gate)

    def triton_fused():
        return ltx2_bias_residual_gate(update, residual, gate, bias)

    torch_times = _time_cuda(torch_two_kernel, repeats, warmup)
    inplace_times = _time_cuda(torch_inplace_bias_then_addcmul, repeats, warmup)
    fused_times = _time_cuda(triton_fused, repeats, warmup)
    torch_stats = _stats(torch_times)
    inplace_stats = _stats(inplace_times)
    fused_stats = _stats(fused_times)
    return {
        "shape": {"name": name, "batch": shape[0], "tokens": shape[1], "hidden": shape[2]},
        "torch_two_kernel": torch_stats,
        "torch_clone_inplace_bias_then_addcmul": inplace_stats,
        "triton_fused_bias_residual_gate": fused_stats,
        "speedup_vs_torch_two_kernel": torch_stats["median_ms"] / fused_stats["median_ms"],
        "speedup_vs_clone_inplace": inplace_stats["median_ms"] / fused_stats["median_ms"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-residual-gate-epilogue/result.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cases = [
        ("stage1_video_ffn_proj_out", (3, 15810, 4096)),
        ("stage2_video_ffn_proj_out", (1, 63240, 4096)),
    ]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "results": {name: _bench_case(name, shape, args.repeats, args.warmup) for name, shape in cases},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
