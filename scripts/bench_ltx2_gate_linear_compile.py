import argparse
import json
from pathlib import Path

import torch
import torch._dynamo
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 64


def gate_linear(x, weight, bias):
    return F.linear(x, weight, bias)


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


def bench(name, m, hidden, heads, repeats, warmup, modes):
    print(f"Running {name} m={m} hidden={hidden} heads={heads}", flush=True)
    torch.manual_seed(9000 + m + hidden + heads)
    x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((heads, hidden), device="cuda", dtype=torch.bfloat16) / (
        hidden**0.5
    )
    bias = torch.randn((heads,), device="cuda", dtype=torch.bfloat16)
    ref, eager_times = time_cuda(lambda: gate_linear(x, weight, bias), repeats, warmup)
    eager_avg = sum(eager_times) / len(eager_times)
    result = {
        "shape": {"m": m, "hidden": hidden, "heads": heads},
        "eager": stats(eager_times),
        "compiled": {},
    }
    for mode in modes:
        compiled = torch.compile(
            gate_linear,
            mode=mode,
            dynamic=False,
            fullgraph=True,
        )
        cand, compiled_times = time_cuda(
            lambda: compiled(x, weight, bias), repeats, warmup
        )
        diff = (ref.float() - cand.float()).abs()
        compiled_avg = sum(compiled_times) / len(compiled_times)
        result["compiled"][mode] = {
            "stats": stats(compiled_times),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "speedup": eager_avg / compiled_avg,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="outputs/ltx23-gate-linear-compile-microbench/result.json"
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead"],
    )
    args = parser.parse_args()
    torch.cuda.set_device(0)
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "results": {
            "stage1_video": bench(
                "stage1_video", 3 * 15810, 4096, 32, args.repeats, args.warmup, args.modes
            ),
            "stage2_video": bench(
                "stage2_video", 63240, 4096, 32, args.repeats, args.warmup, args.modes
            ),
            "stage1_audio": bench(
                "stage1_audio", 3 * 251, 2048, 32, args.repeats, args.warmup, args.modes
            ),
            "stage2_audio": bench(
                "stage2_audio", 251, 2048, 32, args.repeats, args.warmup, args.modes
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
