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
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "ms": times}


def bench_shape(flashinfer_gemm, name, m, k, n, backends, repeats, warmup):
    print(f"benchmarking {name}: m={m} k={k} n={n}", flush=True)
    torch.manual_seed(2026 + m + k + n)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / (k**0.5)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    weight_t = weight.t()

    ref, torch_times = time_cuda(lambda: F.linear(x, weight, bias), repeats, warmup)
    torch_avg = sum(torch_times) / len(torch_times)
    result = {
        "shape": {"m": m, "k": k, "n": n},
        "torch_linear_bias": stats(torch_times),
        "backends": {},
    }
    for backend in backends:
        def call():
            y = flashinfer_gemm.mm_bf16(
                x,
                weight_t,
                bias=None,
                out_dtype=torch.bfloat16,
                backend=backend,
            )
            return y + bias

        try:
            cand, times = time_cuda(call, repeats, warmup)
            diff = (ref.float() - cand.float()).abs()
            avg = sum(times) / len(times)
            result["backends"][backend] = {
                "ok": True,
                "stats": stats(times),
                "speedup_vs_torch": torch_avg / avg,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
            }
        except Exception as exc:
            result["backends"][backend] = {"ok": False, "error": repr(exc)}
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/ltx23-flashinfer-gemm-bias-split-microbench/result.json",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backends", default="cublaslt,cutlass,cudnn")
    args = parser.parse_args()

    import flashinfer.gemm as flashinfer_gemm

    torch.cuda.set_device(0)
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    shapes = [
        ("stage1_ffn_proj_in", 3 * 15810, 4096, 16384),
        ("stage1_ffn_proj_out", 3 * 15810, 16384, 4096),
        ("stage2_ffn_proj_in", 63240, 4096, 16384),
        ("stage2_ffn_proj_out", 63240, 16384, 4096),
    ]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "results": {
            name: bench_shape(
                flashinfer_gemm, name, m, k, n, backends, args.repeats, args.warmup
            )
            for name, m, k, n in shapes
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
