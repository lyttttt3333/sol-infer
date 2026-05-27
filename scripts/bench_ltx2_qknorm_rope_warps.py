import argparse
import json
from pathlib import Path

import torch
import triton

from sglang.jit_kernel.diffusion.triton.ltx2_qknorm import (
    _ltx2_qknorm_split_rope_pair_kernel,
    ltx2_qknorm_split_rope_pair,
)


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


def tuned_qknorm_rope(q, k, q_weight, k_weight, cos, sin, num_warps: int):
    batch, q_seq, hidden = q.shape
    _, k_seq, _ = k.shape
    _, heads, _, half_dim = cos.shape
    head_dim = half_dim * 2
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_rows = batch * q_seq
    k_rows = batch * k_seq
    block_half = triton.next_power_of_2(heads * half_dim)
    _ltx2_qknorm_split_rope_pair_kernel[(max(q_rows, k_rows),)](
        q.view(-1, hidden),
        k.view(-1, hidden),
        q_out.view(-1, hidden),
        k_out.view(-1, hidden),
        q_weight,
        k_weight,
        cos,
        sin,
        cos,
        sin,
        q_rows,
        k_rows,
        q_seq,
        k_seq,
        hidden,
        heads,
        head_dim,
        half_dim,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        1e-6,
        BLOCK_HALF=block_half,
        num_warps=num_warps,
    )
    return q_out, k_out


def bench(name, batch, seq, hidden, heads, repeats, warmup):
    print(f"Running {name} batch={batch} seq={seq} hidden={hidden}", flush=True)
    torch.manual_seed(5000 + batch + seq + hidden)
    half_dim = hidden // heads // 2
    q = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    cos = torch.randn((batch, heads, seq, half_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.randn((batch, heads, seq, half_dim), device="cuda", dtype=torch.bfloat16)

    (q_ref, k_ref), ref_times = time_cuda(
        lambda: ltx2_qknorm_split_rope_pair(
            q, k, q_weight, k_weight, cos, sin, cos, sin, 1e-6
        ),
        repeats,
        warmup,
    )
    result = {
        "shape": {
            "batch": batch,
            "seq": seq,
            "hidden": hidden,
            "heads": heads,
        },
        "current": stats(ref_times),
        "variants": {},
    }
    for num_warps in (2, 4, 8, 16):
        try:
            (q_out, k_out), times = time_cuda(
                lambda nw=num_warps: tuned_qknorm_rope(
                    q, k, q_weight, k_weight, cos, sin, nw
                ),
                repeats,
                warmup,
            )
            diff_q = (q_ref.float() - q_out.float()).abs()
            diff_k = (k_ref.float() - k_out.float()).abs()
            result["variants"][str(num_warps)] = {
                "stats": stats(times),
                "speedup_vs_current": result["current"]["avg_ms"]
                / (sum(times) / len(times)),
                "q_max_abs_diff": float(diff_q.max()),
                "k_max_abs_diff": float(diff_k.max()),
                "q_mean_abs_diff": float(diff_q.mean()),
                "k_mean_abs_diff": float(diff_k.mean()),
            }
        except Exception as exc:
            result["variants"][str(num_warps)] = {"error": repr(exc)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/ltx23-qknorm-rope-warps-microbench/result.json",
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "results": {
            "stage1_video": bench( "stage1_video", 3, 15810, 4096, 32, args.repeats, args.warmup),
            "stage2_video": bench("stage2_video", 1, 63240, 4096, 32, args.repeats, args.warmup),
            "stage1_audio": bench("stage1_audio", 3, 251, 2048, 32, args.repeats, args.warmup),
            "stage2_audio": bench("stage2_audio", 1, 251, 2048, 32, args.repeats, args.warmup),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
