import argparse
import json
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
    fused_norm_scale_shift,
    fused_scale_residual_norm_scale_shift,
)
from sglang.jit_kernel.diffusion.cutedsl.ada_norm_scale_shift import (
    fused_norm_ada_scale_shift_chunked,
    fused_scale_residual_norm_ada_scale_shift_chunked,
)
from sglang.jit_kernel.diffusion.triton.ltx2_ada_values import (
    ltx2_ada_values9,
    ltx2_ada_values_indices3,
    ltx2_rmsnorm_ada_scale_shift,
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


def compare_tensor(base: torch.Tensor, fused: torch.Tensor, prefix: str):
    diff = (base.float() - fused.float()).abs()
    return {
        f"{prefix}_max_abs_diff": float(diff.max().item()),
        f"{prefix}_mean_abs_diff": float(diff.mean().item()),
    }


def compare_tuple(base, fused):
    out = {}
    for i, (base_tensor, fused_tensor) in enumerate(zip(base, fused)):
        out.update(compare_tensor(base_tensor, fused_tensor, f"out{i}"))
    return out


def current_single_adaln(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    shift_index: int,
    scale_index: int,
    eps: float,
):
    values = ltx2_ada_values9(scale_shift_table, timestep)
    return fused_norm_scale_shift(
        x,
        None,
        None,
        values[scale_index],
        values[shift_index],
        "rms",
        eps,
    )


def current_three_adaln(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    eps: float,
):
    values = ltx2_ada_values9(scale_shift_table, timestep)
    return tuple(
        fused_norm_scale_shift(
            x,
            None,
            None,
            values[scale_index],
            values[shift_index],
            "rms",
            eps,
        )
        for shift_index, scale_index in ((0, 1), (3, 4), (6, 7))
    )


def current_three_adaln_plus_gates(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    eps: float,
):
    values = ltx2_ada_values9(scale_shift_table, timestep)
    gates = (values[2], values[5], values[8])
    adaln = tuple(
        fused_norm_scale_shift(
            x,
            None,
            None,
            values[scale_index],
            values[shift_index],
            "rms",
            eps,
        )
        for shift_index, scale_index in ((0, 1), (3, 4), (6, 7))
    )
    return gates + adaln


def direct_single_adaln(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    shift_index: int,
    scale_index: int,
    eps: float,
):
    return ltx2_rmsnorm_ada_scale_shift(
        x,
        scale_shift_table,
        timestep,
        shift_index,
        scale_index,
        eps,
    )


def cute_direct_single_adaln(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    shift_index: int,
    scale_index: int,
    eps: float,
):
    return fused_norm_ada_scale_shift_chunked(
        x,
        timestep,
        scale_shift_table,
        shift_index,
        scale_index,
        "rms",
        eps,
    )


def direct_three_adaln(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    eps: float,
):
    return tuple(
        ltx2_rmsnorm_ada_scale_shift(
            x,
            scale_shift_table,
            timestep,
            shift_index,
            scale_index,
            eps,
        )
        for shift_index, scale_index in ((0, 1), (3, 4), (6, 7))
    )


def cute_direct_three_adaln(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    eps: float,
):
    return tuple(
        fused_norm_ada_scale_shift_chunked(
            x,
            timestep,
            scale_shift_table,
            shift_index,
            scale_index,
            "rms",
            eps,
        )
        for shift_index, scale_index in ((0, 1), (3, 4), (6, 7))
    )


def cute_direct_three_adaln_plus_gates(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    eps: float,
):
    gates = ltx2_ada_values_indices3(scale_shift_table, timestep, 2, 5, 8)
    return gates + cute_direct_three_adaln(x, scale_shift_table, timestep, eps)


def current_actual_block_ada(
    x: torch.Tensor,
    attn_x: torch.Tensor,
    a2v_x: torch.Tensor,
    a2v_gate: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    eps: float,
):
    values = ltx2_ada_values9(scale_shift_table, timestep)
    norm_self = fused_norm_scale_shift(
        x, None, None, values[1], values[0], "rms", eps
    )
    norm_q, residual_q = fused_scale_residual_norm_scale_shift(
        x, attn_x, values[2], None, None, values[7], values[6], "rms", eps
    )
    norm_mlp, residual_mlp = fused_scale_residual_norm_scale_shift(
        residual_q, a2v_x, a2v_gate, None, None, values[4], values[3], "rms", eps
    )
    return (values[2], values[5], values[8], norm_self, norm_q, residual_q, norm_mlp, residual_mlp)


def cute_actual_block_ada(
    x: torch.Tensor,
    attn_x: torch.Tensor,
    a2v_x: torch.Tensor,
    a2v_gate: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    eps: float,
):
    gates = ltx2_ada_values_indices3(scale_shift_table, timestep, 2, 5, 8)
    norm_self = fused_norm_ada_scale_shift_chunked(
        x, timestep, scale_shift_table, 0, 1, "rms", eps
    )
    norm_q, residual_q = fused_scale_residual_norm_ada_scale_shift_chunked(
        x, attn_x, gates[0], timestep, scale_shift_table, 6, 7, "rms", eps
    )
    norm_mlp, residual_mlp = fused_scale_residual_norm_ada_scale_shift_chunked(
        residual_q, a2v_x, a2v_gate, timestep, scale_shift_table, 3, 4, "rms", eps
    )
    return gates + (norm_self, norm_q, residual_q, norm_mlp, residual_mlp)


def bench_case(
    name: str,
    batch: int,
    seq: int,
    hidden: int,
    total_params: int,
    repeats: int,
    warmup: int,
    eps: float,
):
    torch.manual_seed(2718 + batch + seq + hidden)
    x = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    attn_x = torch.randn_like(x)
    a2v_x = torch.randn_like(x)
    a2v_gate = torch.randn_like(x)
    table = torch.randn((total_params, hidden), device="cuda", dtype=torch.bfloat16)
    timestep = torch.randn(
        (batch, seq, total_params * hidden), device="cuda", dtype=torch.bfloat16
    )

    single_args = (x, table, timestep, 0, 1, eps)
    three_args = (x, table, timestep, eps)
    actual_args = (x, attn_x, a2v_x, a2v_gate, table, timestep, eps)

    current_one_out, current_one_times = time_cuda(
        lambda: current_single_adaln(*single_args), repeats, warmup
    )
    direct_one_out, direct_one_times = time_cuda(
        lambda: direct_single_adaln(*single_args), repeats, warmup
    )
    cute_direct_one_out, cute_direct_one_times = time_cuda(
        lambda: cute_direct_single_adaln(*single_args), repeats, warmup
    )
    current_three_out, current_three_times = time_cuda(
        lambda: current_three_adaln(*three_args), repeats, warmup
    )
    direct_three_out, direct_three_times = time_cuda(
        lambda: direct_three_adaln(*three_args), repeats, warmup
    )
    cute_direct_three_out, cute_direct_three_times = time_cuda(
        lambda: cute_direct_three_adaln(*three_args), repeats, warmup
    )
    current_combo_out, current_combo_times = time_cuda(
        lambda: current_three_adaln_plus_gates(*three_args), repeats, warmup
    )
    cute_combo_out, cute_combo_times = time_cuda(
        lambda: cute_direct_three_adaln_plus_gates(*three_args), repeats, warmup
    )
    current_actual_out, current_actual_times = time_cuda(
        lambda: current_actual_block_ada(*actual_args), repeats, warmup
    )
    cute_actual_out, cute_actual_times = time_cuda(
        lambda: cute_actual_block_ada(*actual_args), repeats, warmup
    )

    current_one_avg = sum(current_one_times) / len(current_one_times)
    direct_one_avg = sum(direct_one_times) / len(direct_one_times)
    cute_direct_one_avg = sum(cute_direct_one_times) / len(cute_direct_one_times)
    current_three_avg = sum(current_three_times) / len(current_three_times)
    direct_three_avg = sum(direct_three_times) / len(direct_three_times)
    cute_direct_three_avg = sum(cute_direct_three_times) / len(cute_direct_three_times)
    current_combo_avg = sum(current_combo_times) / len(current_combo_times)
    cute_combo_avg = sum(cute_combo_times) / len(cute_combo_times)
    current_actual_avg = sum(current_actual_times) / len(current_actual_times)
    cute_actual_avg = sum(cute_actual_times) / len(cute_actual_times)
    return {
        "name": name,
        "shape": {
            "batch": batch,
            "seq": seq,
            "hidden": hidden,
            "total_params": total_params,
            "eps": eps,
        },
        "single_current_all9_plus_adaln": stats(current_one_times),
        "single_direct_ada_adaln": stats(direct_one_times),
        "single_speedup": current_one_avg / direct_one_avg,
        **compare_tensor(current_one_out, direct_one_out, "single"),
        "single_cute_direct_ada_adaln": stats(cute_direct_one_times),
        "single_cute_speedup": current_one_avg / cute_direct_one_avg,
        **compare_tensor(current_one_out, cute_direct_one_out, "single_cute"),
        "three_current_all9_plus_3_adaln": stats(current_three_times),
        "three_direct_3_ada_adaln": stats(direct_three_times),
        "three_speedup": current_three_avg / direct_three_avg,
        **compare_tuple(current_three_out, direct_three_out),
        "three_cute_direct_3_ada_adaln": stats(cute_direct_three_times),
        "three_cute_speedup": current_three_avg / cute_direct_three_avg,
        **{
            f"cute_{k}": v
            for k, v in compare_tuple(
                current_three_out, cute_direct_three_out
            ).items()
        },
        "combo_current_all9_plus_3_adaln_plus_gates": stats(current_combo_times),
        "combo_cute_direct_3_adaln_plus_gate_only": stats(cute_combo_times),
        "combo_cute_speedup": current_combo_avg / cute_combo_avg,
        **{
            f"combo_{k}": v
            for k, v in compare_tuple(current_combo_out, cute_combo_out).items()
        },
        "actual_current_all9_plus_real_adaln": stats(current_actual_times),
        "actual_cute_direct_plus_gate_only": stats(cute_actual_times),
        "actual_cute_speedup": current_actual_avg / cute_actual_avg,
        **{
            f"actual_{k}": v
            for k, v in compare_tuple(current_actual_out, cute_actual_out).items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/ltx23-adaln-from-ada-microbench/result.json",
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "results": {
            "stage1_video": bench_case(
                "stage1_video", 3, 15810, 4096, 9, args.repeats, args.warmup, args.eps
            ),
            "stage2_video": bench_case(
                "stage2_video", 1, 63240, 4096, 9, args.repeats, args.warmup, args.eps
            ),
            "stage1_audio": bench_case(
                "stage1_audio", 3, 251, 2048, 9, args.repeats, args.warmup, args.eps
            ),
            "stage2_audio": bench_case(
                "stage2_audio", 1, 251, 2048, 9, args.repeats, args.warmup, args.eps
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
