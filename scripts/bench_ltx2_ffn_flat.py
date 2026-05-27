import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def ffn_direct(x, w1, b1, w2, b2):
    y = F.linear(x, w1, b1)
    y = F.gelu(y, approximate="tanh")
    y = F.linear(y, w2, b2)
    return y


def ffn_flat(x, w1, b1, w2, b2):
    prefix = x.shape[:-1]
    y = F.linear(x.reshape(-1, x.shape[-1]), w1, b1)
    y = F.gelu(y, approximate="tanh")
    y = F.linear(y, w2, b2)
    return y.reshape(*prefix, w2.shape[0])


def time_cuda(fn, repeats, warmup):
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


def bench(name, shape, hidden, inner, repeats, warmup):
    print(f"Running {name} shape={shape}", flush=True)
    torch.manual_seed(20260525 + len(name))
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn((inner, hidden), device="cuda", dtype=torch.bfloat16) / math.sqrt(hidden)
    b1 = torch.randn((inner,), device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn((hidden, inner), device="cuda", dtype=torch.bfloat16) / math.sqrt(inner)
    b2 = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    ref, direct_times = time_cuda(lambda: ffn_direct(x, w1, b1, w2, b2), repeats, warmup)
    cand, flat_times = time_cuda(lambda: ffn_flat(x, w1, b1, w2, b2), repeats, warmup)
    diff = (ref.float() - cand.float()).abs()
    direct_avg = sum(direct_times) / len(direct_times)
    flat_avg = sum(flat_times) / len(flat_times)
    return {
        "shape": list(shape),
        "direct_ms": direct_times,
        "flat_ms": flat_times,
        "direct_avg_ms": direct_avg,
        "flat_avg_ms": flat_avg,
        "speedup": direct_avg / flat_avg,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-ffn-flat-microbench/result.json")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    results = {
        "stage1_video": bench("stage1_video", (3, 15810, 4096), 4096, 16384, args.repeats, args.warmup),
        "stage2_video": bench("stage2_video", (1, 63240, 4096), 4096, 16384, args.repeats, args.warmup),
    }
    payload = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
