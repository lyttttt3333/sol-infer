import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gate_linear_n32_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_M, 32), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * K + k_idxs[None, :],
            mask=k_idxs[None, :] < K,
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w), input_precision="ieee")
    bias = tl.load(b_ptr + offs_n).to(tl.float32)
    acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * 32 + offs_n[None, :],
        acc,
        mask=offs_m[:, None] < M,
    )


def gate_linear_triton(x, weight, bias, block_m: int, block_k: int):
    x_2d = x.reshape(-1, x.shape[-1])
    m = x_2d.shape[0]
    k = x_2d.shape[1]
    if weight.shape != (32, k):
        raise ValueError(f"expected weight shape {(32, k)}, got {tuple(weight.shape)}")
    out = torch.empty((m, 32), device=x.device, dtype=x.dtype)
    _gate_linear_n32_kernel[(triton.cdiv(m, block_m),)](
        x_2d,
        weight,
        bias,
        out,
        m,
        k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out.reshape(*x.shape[:-1], 32)


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


def bench(name, m, hidden, repeats, warmup):
    print(f"Running {name} m={m} hidden={hidden}", flush=True)
    torch.manual_seed(10000 + m + hidden)
    x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((32, hidden), device="cuda", dtype=torch.bfloat16) / (
        hidden**0.5
    )
    bias = torch.randn((32,), device="cuda", dtype=torch.bfloat16)
    ref, eager_times = time_cuda(lambda: F.linear(x, weight, bias), repeats, warmup)
    eager_avg = sum(eager_times) / len(eager_times)
    result = {
        "shape": {"m": m, "hidden": hidden, "heads": 32},
        "eager": stats(eager_times),
        "variants": {},
    }
    for block_m in (32, 64, 128):
        for block_k in (64, 128):
            key = f"bm{block_m}_bk{block_k}"
            try:
                cand, times = time_cuda(
                    lambda bm=block_m, bk=block_k: gate_linear_triton(
                        x, weight, bias, bm, bk
                    ),
                    repeats,
                    warmup,
                )
                diff = (ref.float() - cand.float()).abs()
                avg = sum(times) / len(times)
                result["variants"][key] = {
                    "stats": stats(times),
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "speedup": eager_avg / avg,
                }
            except Exception as exc:
                result["variants"][key] = {"error": repr(exc)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="outputs/ltx23-gate-linear-triton-microbench/result.json"
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "results": {
            "stage1_video": bench("stage1_video", 3 * 15810, 4096, args.repeats, args.warmup),
            "stage2_video": bench("stage2_video", 63240, 4096, args.repeats, args.warmup),
            "stage1_audio": bench("stage1_audio", 3 * 251, 2048, args.repeats, args.warmup),
            "stage2_audio": bench("stage2_audio", 251, 2048, args.repeats, args.warmup),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
