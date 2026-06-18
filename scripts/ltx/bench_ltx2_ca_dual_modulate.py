import argparse
import json
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.triton.ltx2_dual_modulate import (
    ltx2_rmsnorm_ca_dual_modulate_from_temb,
    ltx2_rmsnorm_dual_modulate,
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


def bench(batch: int, seq: int, hidden: int, repeats: int, warmup: int):
    torch.manual_seed(123)
    x = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    temb_scale_shift = torch.randn(
        (batch, seq, 4 * hidden), device="cuda", dtype=torch.bfloat16
    )
    temb_gate = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    table = torch.randn((5, hidden), device="cuda", dtype=torch.bfloat16)

    def gate():
        return (
            table[4:, :][None, None, :, :].to(
                dtype=temb_gate.dtype, device=temb_gate.device
            )
            + temb_gate.reshape(batch, seq, 1, -1)
        ).squeeze(2)

    def materialize_scale_shift():
        vals = (
            table[:4, :][None, None, :, :].to(
                dtype=temb_scale_shift.dtype, device=temb_scale_shift.device
            )
            + temb_scale_shift.reshape(batch, seq, 4, -1)
        ).unbind(dim=2)
        return [t.squeeze(2) for t in vals]

    def base_current():
        scale0, shift0, scale1, shift1 = materialize_scale_shift()
        y0, y1 = ltx2_rmsnorm_dual_modulate(
            x, scale0, shift0, scale1, shift1, 1e-6
        )
        return y0, y1, gate()

    def fused_table():
        y0, y1 = ltx2_rmsnorm_ca_dual_modulate_from_temb(
            x, temb_scale_shift, table[:4, :], 1e-6
        )
        return y0, y1, gate()

    (b0, b1, bg), base_times = time_cuda(base_current, repeats, warmup)
    (f0, f1, fg), fused_times = time_cuda(fused_table, repeats, warmup)
    return {
        "shape": {"batch": batch, "seq": seq, "hidden": hidden},
        "base_current_dualmod_plus_materialize": stats(base_times),
        "fused_table_dualmod_plus_gate": stats(fused_times),
        "speedup": (sum(base_times) / len(base_times))
        / (sum(fused_times) / len(fused_times)),
        "y0_max_abs_diff": float((b0.float() - f0.float()).abs().max().item()),
        "y1_max_abs_diff": float((b1.float() - f1.float()).abs().max().item()),
        "gate_max_abs_diff": float((bg.float() - fg.float()).abs().max().item()),
        "y0_mean_abs_diff": float((b0.float() - f0.float()).abs().mean().item()),
        "y1_mean_abs_diff": float((b1.float() - f1.float()).abs().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="outputs/ltx23-ca-dual-modulate-microbench/result.json"
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "results": {
            "stage1_video": bench(3, 15810, 4096, args.repeats, args.warmup),
            "stage2_video": bench(1, 63240, 4096, args.repeats, args.warmup),
            "stage1_audio": bench(3, 251, 2048, args.repeats, args.warmup),
            "stage2_audio": bench(1, 251, 2048, args.repeats, args.warmup),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
