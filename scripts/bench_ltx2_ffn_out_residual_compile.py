import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def eager(x, w, b, residual, gate):
    y = F.linear(x, w, b)
    return torch.addcmul(residual, y, gate)


def split_eager(x, w, b, residual, gate):
    y = F.linear(x, w, b)
    return torch.addcmul(residual, y, gate)


def _time_cuda(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        y = fn()
        torch.cuda.synchronize()
        del y

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        y = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
        del y
    return times


def bench_shape(name, batch, seq, hidden, inner, repeats, warmup):
    print(f"benchmarking {name}: b={batch} seq={seq} hidden={hidden} inner={inner}", flush=True)
    torch.manual_seed(123)
    x = torch.randn((batch, seq, inner), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((hidden, inner), device="cuda", dtype=torch.bfloat16) / (inner**0.5)
    b = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    args = (x, w, b, residual, gate)

    compiled = torch.compile(
        eager,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
    y0 = split_eager(*args)
    torch.cuda.synchronize()
    y1 = compiled(*args)
    torch.cuda.synchronize()
    diff = (y0.float() - y1.float()).abs()

    split_times = _time_cuda(lambda: split_eager(*args), repeats, warmup)
    compiled_times = _time_cuda(lambda: compiled(*args), repeats, warmup)
    split_avg = sum(split_times) / len(split_times)
    compiled_avg = sum(compiled_times) / len(compiled_times)
    return {
        "shape": {
            "batch": batch,
            "seq": seq,
            "hidden": hidden,
            "inner": inner,
        },
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "split_eager_ms": split_times,
        "compiled_ms": compiled_times,
        "split_eager_avg_ms": split_avg,
        "compiled_avg_ms": compiled_avg,
        "speedup": split_avg / compiled_avg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-ffn-out-residual-compile-microbench/result.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    shapes = [
        ("stage1_video", 3, 15810, 4096, 16384),
        ("stage2_video", 1, 63240, 4096, 16384),
        ("stage1_audio", 3, 251, 2048, 8192),
        ("stage2_audio", 1, 251, 2048, 8192),
    ]
    results = {}
    for item in shapes:
        name, batch, seq, hidden, inner = item
        results[name] = bench_shape(name, batch, seq, hidden, inner, args.repeats, args.warmup)
        torch.cuda.empty_cache()

    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
