import argparse
import json
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.triton.ltx2_ada_values import (
    ltx2_ada_values3,
    ltx2_ada_values9,
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
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "ms": times}


def base_ada_values(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    timestep: torch.Tensor,
    indices: slice,
):
    num_ada_params = int(scale_shift_table.shape[0])
    ada_values = (
        scale_shift_table[indices]
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=timestep.device, dtype=timestep.dtype)
        + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[
            :, :, indices, :
        ]
    ).unbind(dim=2)
    return [t.squeeze(2) for t in ada_values]


def compare_tuple(base, fused):
    return {
        f"out{i}_max_abs_diff": float((b.float() - f.float()).abs().max().item())
        for i, (b, f) in enumerate(zip(base, fused))
    } | {
        f"out{i}_mean_abs_diff": float((b.float() - f.float()).abs().mean().item())
        for i, (b, f) in enumerate(zip(base, fused))
    }


def bench_case(
    batch: int,
    seq: int,
    hidden: int,
    total_params: int,
    start_index: int,
    repeats: int,
    warmup: int,
):
    torch.manual_seed(123 + start_index)
    table = torch.randn((total_params, hidden), device="cuda", dtype=torch.bfloat16)
    timestep = torch.randn(
        (batch, seq, total_params * hidden), device="cuda", dtype=torch.bfloat16
    )
    indices = slice(start_index, start_index + 3)

    def base():
        return base_ada_values(table, batch, timestep, indices)

    def fused():
        return ltx2_ada_values3(table, timestep, start_index)

    base_out, base_times = time_cuda(base, repeats, warmup)
    fused_out, fused_times = time_cuda(fused, repeats, warmup)
    return {
        "shape": {
            "batch": batch,
            "seq": seq,
            "hidden": hidden,
            "total_params": total_params,
            "start_index": start_index,
        },
        "base": stats(base_times),
        "fused": stats(fused_times),
        "speedup": (sum(base_times) / len(base_times))
        / (sum(fused_times) / len(fused_times)),
        **compare_tuple(base_out, fused_out),
    }


def bench_all9(
    batch: int,
    seq: int,
    hidden: int,
    total_params: int,
    repeats: int,
    warmup: int,
):
    torch.manual_seed(987)
    table = torch.randn((total_params, hidden), device="cuda", dtype=torch.bfloat16)
    timestep = torch.randn(
        (batch, seq, total_params * hidden), device="cuda", dtype=torch.bfloat16
    )

    def base_triples():
        return tuple(
            item
            for start in (0, 3, 6)
            for item in base_ada_values(table, batch, timestep, slice(start, start + 3))
        )

    def fused_all9():
        return ltx2_ada_values9(table, timestep)

    base_out, base_times = time_cuda(base_triples, repeats, warmup)
    fused_out, fused_times = time_cuda(fused_all9, repeats, warmup)
    return {
        "shape": {
            "batch": batch,
            "seq": seq,
            "hidden": hidden,
            "total_params": total_params,
        },
        "base_three_triples": stats(base_times),
        "fused_all9": stats(fused_times),
        "speedup": (sum(base_times) / len(base_times))
        / (sum(fused_times) / len(fused_times)),
        **compare_tuple(base_out, fused_out),
    }


def bench_shape(
    batch: int,
    seq: int,
    hidden: int,
    total_params: int,
    repeats: int,
    warmup: int,
):
    return {
        f"slice_{start}_{start + 3}": bench_case(
            batch, seq, hidden, total_params, start, repeats, warmup
        )
        for start in (0, 3, 6)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-ada-values-microbench/result.json")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "results": {
            "stage1_video": bench_shape(3, 15810, 4096, 9, args.repeats, args.warmup),
            "stage2_video": bench_shape(1, 63240, 4096, 9, args.repeats, args.warmup),
            "stage1_audio": bench_shape(3, 251, 2048, 9, args.repeats, args.warmup),
            "stage2_audio": bench_shape(1, 251, 2048, 9, args.repeats, args.warmup),
            "all9_stage1_video": bench_all9(3, 15810, 4096, 9, args.repeats, args.warmup),
            "all9_stage2_video": bench_all9(1, 63240, 4096, 9, args.repeats, args.warmup),
            "all9_stage1_audio": bench_all9(3, 251, 2048, 9, args.repeats, args.warmup),
            "all9_stage2_audio": bench_all9(1, 251, 2048, 9, args.repeats, args.warmup),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
