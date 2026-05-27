import argparse
import json
from pathlib import Path

import torch


def eager(residual, x, gate):
    return residual + x * gate


def addcmul(residual, x, gate):
    return torch.addcmul(residual, x, gate)


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


def bench(name, shape, gate_shape, repeats, warmup):
    print(f"Running {name} shape={shape} gate={gate_shape}", flush=True)
    torch.manual_seed(12345 + len(name))
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(gate_shape, device="cuda", dtype=torch.bfloat16)
    ref, eager_times = time_cuda(lambda: eager(residual, x, gate), repeats, warmup)
    cand, addcmul_times = time_cuda(lambda: addcmul(residual, x, gate), repeats, warmup)
    diff = (ref.float() - cand.float()).abs()
    ea = sum(eager_times) / len(eager_times)
    aa = sum(addcmul_times) / len(addcmul_times)
    return {
        "shape": list(shape),
        "gate_shape": list(gate_shape),
        "eager_ms": eager_times,
        "addcmul_ms": addcmul_times,
        "eager_avg_ms": ea,
        "addcmul_avg_ms": aa,
        "speedup": ea / aa,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "bit_exact": bool(torch.equal(ref, cand)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-residual-gate-microbench/result.json")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    results = {
        "stage1_video_dim_gate": bench("stage1_video_dim_gate", (3, 15810, 4096), (3, 1, 4096), args.repeats, args.warmup),
        "stage2_video_dim_gate": bench("stage2_video_dim_gate", (1, 63240, 4096), (1, 1, 4096), args.repeats, args.warmup),
        "stage1_audio_dim_gate": bench("stage1_audio_dim_gate", (3, 251, 2048), (3, 1, 2048), args.repeats, args.warmup),
        "stage2_audio_dim_gate": bench("stage2_audio_dim_gate", (1, 251, 2048), (1, 1, 2048), args.repeats, args.warmup),
    }
    payload = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
