import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


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
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "ms": times,
    }


def eager_linear_gelu(x, weight, bias):
    return F.gelu(F.linear(x, weight, bias), approximate="tanh")


def addmm_activation_linear_gelu(x, weight, bias):
    x_2d = x.reshape(-1, x.shape[-1])
    out = torch.ops.aten._addmm_activation.default(
        bias,
        x_2d,
        weight.t(),
        beta=1,
        alpha=1,
        use_gelu=True,
    )
    return out.reshape(*x.shape[:-1], weight.shape[0])


def bench(name, shape, hidden, inner, repeats, warmup):
    print(f"Running {name} shape={shape} hidden={hidden} inner={inner}", flush=True)
    torch.manual_seed(1234 + len(name))
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((inner, hidden), device="cuda", dtype=torch.bfloat16) / (
        hidden**0.5
    )
    bias = torch.randn((inner,), device="cuda", dtype=torch.bfloat16)

    ref, eager_times = time_cuda(
        lambda: eager_linear_gelu(x, weight, bias), repeats, warmup
    )
    cand, fused_times = time_cuda(
        lambda: addmm_activation_linear_gelu(x, weight, bias), repeats, warmup
    )
    diff = (ref.float() - cand.float()).abs()
    eager_avg = sum(eager_times) / len(eager_times)
    fused_avg = sum(fused_times) / len(fused_times)
    return {
        "shape": list(shape),
        "hidden": hidden,
        "inner": inner,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "eager_linear_gelu": stats(eager_times),
        "addmm_activation": stats(fused_times),
        "speedup": eager_avg / fused_avg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/ltx23-addmm-activation-microbench/result.json",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--which",
        choices=["stage1_video", "stage2_video", "audio", "all"],
        default="all",
    )
    args = parser.parse_args()

    torch.cuda.set_device(0)
    todo = []
    if args.which in ("stage1_video", "all"):
        todo.append(("stage1_video_b3_t15810", (3, 15810, 4096), 4096, 16384))
    if args.which in ("stage2_video", "all"):
        todo.append(("stage2_video_b1_t63240", (1, 63240, 4096), 4096, 16384))
    if args.which in ("audio", "all"):
        todo.append(("stage1_audio_b3_t251", (3, 251, 2048), 2048, 8192))
        todo.append(("stage2_audio_b1_t251", (1, 251, 2048), 2048, 8192))

    payload = {
        "device": torch.cuda.get_device_name(0),
        "results": {
            name: bench(name, shape, hidden, inner, args.repeats, args.warmup)
            for name, shape, hidden, inner in todo
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
