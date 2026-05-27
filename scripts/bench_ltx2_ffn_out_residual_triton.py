import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _ffn_out_residual_gate_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    residual_ptr,
    gate_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * K + k_idxs[None, :],
            mask=(offs_n[:, None] < N) & (k_idxs[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w), input_precision="ieee")

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    residual = tl.load(
        residual_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    y = residual + acc * gate
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_ffn_out_residual_gate(x, weight, bias, residual, gate, block_m, block_n, block_k, num_warps):
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    residual2 = residual.reshape(-1, residual.shape[-1]).contiguous()
    gate2 = gate.reshape(-1, gate.shape[-1]).contiguous()
    m = x2.shape[0]
    k = x2.shape[1]
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _ffn_out_residual_gate_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        x2,
        weight,
        bias,
        residual2,
        gate2,
        out,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )
    return out.reshape(*residual.shape)


def eager_ffn_out_residual_gate(x, weight, bias, residual, gate):
    return torch.addcmul(residual, F.linear(x, weight, bias), gate)


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


def stats(times):
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "ms": times}


def bench_shape(name, batch, seq, hidden, inner, variants, repeats, warmup, check_diff):
    print(f"Running {name}: batch={batch} seq={seq} hidden={hidden} inner={inner}", flush=True)
    torch.manual_seed(11000 + batch + seq + hidden + inner)
    x = torch.randn((batch, seq, inner), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden, inner), device="cuda", dtype=torch.bfloat16) / (inner**0.5)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)

    ref, eager_times = time_cuda(
        lambda: eager_ffn_out_residual_gate(x, weight, bias, residual, gate),
        repeats,
        warmup,
    )
    eager_avg = sum(eager_times) / len(eager_times)
    result = {
        "shape": {"batch": batch, "seq": seq, "m": batch * seq, "hidden": hidden, "inner": inner},
        "eager": stats(eager_times),
        "variants": {},
    }
    for block_m, block_n, block_k, num_warps in variants:
        key = f"bm{block_m}_bn{block_n}_bk{block_k}_w{num_warps}"
        print(f"  variant {key}", flush=True)
        try:
            cand, times = time_cuda(
                lambda bm=block_m, bn=block_n, bk=block_k, nw=num_warps: fused_ffn_out_residual_gate(
                    x, weight, bias, residual, gate, bm, bn, bk, nw
                ),
                repeats,
                warmup,
            )
            avg = sum(times) / len(times)
            item = {"stats": stats(times), "speedup": eager_avg / avg}
            if check_diff:
                diff = (ref.float() - cand.float()).abs()
                item["max_abs_diff"] = float(diff.max().item())
                item["mean_abs_diff"] = float(diff.mean().item())
            result["variants"][key] = item
        except Exception as exc:
            result["variants"][key] = {"error": repr(exc)}
            print(f"  error {key}: {exc!r}", flush=True)
        torch.cuda.empty_cache()
    return result


def parse_variant(text):
    parts = [int(x) for x in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("variant must be BLOCK_M,BLOCK_N,BLOCK_K,num_warps")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-ffn-out-residual-triton-microbench/result.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--shape", choices=["stage1_video", "stage2_video", "video", "audio", "all"], default="video")
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        default=None,
        help="Triton variant as BLOCK_M,BLOCK_N,BLOCK_K,num_warps. Can be repeated.",
    )
    parser.add_argument("--check-diff", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    variants = args.variant or [(16, 64, 64, 4), (16, 128, 64, 4), (32, 64, 64, 4)]
    shape_map = {
        "stage1_video": [("stage1_video", 3, 15810, 4096, 16384)],
        "stage2_video": [("stage2_video", 1, 63240, 4096, 16384)],
        "video": [
            ("stage1_video", 3, 15810, 4096, 16384),
            ("stage2_video", 1, 63240, 4096, 16384),
        ],
        "audio": [
            ("stage1_audio", 3, 251, 2048, 8192),
            ("stage2_audio", 1, 251, 2048, 8192),
        ],
        "all": [
            ("stage1_video", 3, 15810, 4096, 16384),
            ("stage2_video", 1, 63240, 4096, 16384),
            ("stage1_audio", 3, 251, 2048, 8192),
            ("stage2_audio", 1, 251, 2048, 8192),
        ],
    }
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "variants": [list(v) for v in variants],
        "results": {},
    }
    for shape in shape_map[args.shape]:
        name, batch, seq, hidden, inner = shape
        payload["results"][name] = bench_shape(
            name, batch, seq, hidden, inner, variants, args.repeats, args.warmup, args.check_diff
        )
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
