#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.triton.ltx2_gelu import ltx2_bias_residual_gate
from sglang.jit_kernel.nvfp4 import (
    cutlass_scaled_fp4_mm,
    cutlass_scaled_fp4_mm_batched_per_col_residual_gate,
    scaled_fp4_quant,
)
from sglang.srt.utils.common import round_up


def _sync() -> None:
    torch.cuda.synchronize()


def _time_ms(fn, iters: int) -> list[float]:
    out = None
    xs = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        xs.append((time.perf_counter() - t0) * 1000.0)
    if out is not None:
        _ = float(out.flatten()[0].float().cpu())
    return xs


def _summary(xs: list[float]) -> dict[str, object]:
    return {
        "avg_ms": statistics.mean(xs),
        "median_ms": statistics.median(xs),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "samples_ms": xs,
    }


def _pad_activation_scales_for_batched_cutlass(
    x_scale: torch.Tensor, batch_size: int, m_per_batch: int
) -> torch.Tensor:
    rounded_m = round_up(m_per_batch, 128)
    if x_scale.shape[0] == batch_size * rounded_m:
        return x_scale
    out = torch.empty(
        (batch_size * rounded_m, x_scale.shape[1]),
        device=x_scale.device,
        dtype=x_scale.dtype,
    )
    out.zero_()
    for batch_idx in range(batch_size):
        src = batch_idx * m_per_batch
        dst = batch_idx * rounded_m
        out[dst : dst + m_per_batch].copy_(x_scale[src : src + m_per_batch])
    return out


def run_case(
    name: str,
    batch_size: int,
    m_per_batch: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    torch.manual_seed(9102)
    device = "cuda"
    total_m = batch_size * m_per_batch
    a = torch.randn((total_m, k), device=device, dtype=dtype) * 0.5
    b = torch.randn((n, k), device=device, dtype=dtype) * 0.5
    residual = torch.randn((total_m, n), device=device, dtype=dtype) * 0.5
    gate = torch.randn((batch_size, n), device=device, dtype=dtype) * 0.1
    bias = torch.randn((n,), device=device, dtype=dtype) * 0.1
    one = torch.tensor([1.0], device=device, dtype=torch.float32)

    a_fp4, a_sf = scaled_fp4_quant(a, one)
    b_fp4, b_sf = scaled_fp4_quant(b, one)
    a_sf_batched = _pad_activation_scales_for_batched_cutlass(
        a_sf, batch_size, m_per_batch
    )
    a_batches = a.view(batch_size, m_per_batch, k)
    a_fp4_parts = []
    a_sf_parts = []
    for batch_idx in range(batch_size):
        a_fp4_part, a_sf_part = scaled_fp4_quant(a_batches[batch_idx], one)
        a_fp4_parts.append(a_fp4_part)
        a_sf_parts.append(a_sf_part)
    a_fp4_per_batch = torch.cat(a_fp4_parts, dim=0).contiguous()
    a_sf_per_batch = torch.cat(a_sf_parts, dim=0).contiguous()
    b_sf_batched = b_sf.repeat((batch_size, 1)).contiguous()
    gate_alpha = gate.contiguous()
    bias_gate = (bias.float().reshape(1, n) * gate.float()).to(dtype).contiguous()

    def baseline():
        y = cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, one, dtype)
        return ltx2_bias_residual_gate(
            y.view(batch_size, m_per_batch, n),
            residual.view(batch_size, m_per_batch, n),
            gate.view(batch_size, 1, n),
            bias,
        ).view(total_m, n)

    def fused_prepared():
        return cutlass_scaled_fp4_mm_batched_per_col_residual_gate(
            a_fp4,
            b_fp4,
            a_sf_batched,
            b_sf_batched,
            one,
            residual,
            gate_alpha,
            bias_gate,
            dtype,
            batch_size,
            m_per_batch,
        )

    def fused_with_scale_pack():
        a_sf_local = _pad_activation_scales_for_batched_cutlass(
            a_sf, batch_size, m_per_batch
        )
        return cutlass_scaled_fp4_mm_batched_per_col_residual_gate(
            a_fp4,
            b_fp4,
            a_sf_local,
            b_sf_batched,
            one,
            residual,
            gate_alpha,
            bias_gate,
            dtype,
            batch_size,
            m_per_batch,
        )

    def fused_per_batch_quantized():
        return cutlass_scaled_fp4_mm_batched_per_col_residual_gate(
            a_fp4_per_batch,
            b_fp4,
            a_sf_per_batch,
            b_sf_batched,
            one,
            residual,
            gate_alpha,
            bias_gate,
            dtype,
            batch_size,
            m_per_batch,
        )

    for _ in range(warmup):
        baseline()
        fused_prepared()
        fused_with_scale_pack()
        fused_per_batch_quantized()
    _sync()
    y0 = baseline()
    y1 = fused_prepared()
    y2 = fused_per_batch_quantized()
    _sync()
    diff = (y0.float() - y1.float()).abs()
    diff_per_batch = (y0.float() - y2.float()).abs()
    base_times = _time_ms(baseline, iters)
    fused_prepared_times = _time_ms(fused_prepared, iters)
    fused_with_pack_times = _time_ms(fused_with_scale_pack, iters)
    fused_per_batch_times = _time_ms(fused_per_batch_quantized, iters)
    return {
        "name": name,
        "shape": {
            "batch_size": batch_size,
            "m_per_batch": m_per_batch,
            "total_m": total_m,
            "n": n,
            "k": k,
            "dtype": str(dtype).replace("torch.", ""),
            "a_sf_rows_flat": int(a_sf.shape[0]),
            "a_sf_rows_batched": int(a_sf_batched.shape[0]),
            "b_sf_rows": int(b_sf.shape[0]),
            "b_sf_rows_batched": int(b_sf_batched.shape[0]),
            "a_sf_rows_per_batch_quantized": int(a_sf_per_batch.shape[0]),
        },
        "baseline_flat_cutlass_plus_triton_bias_residual_gate": _summary(base_times),
        "fused_batched_cutlass_epilogue_prepared_scales": _summary(
            fused_prepared_times
        ),
        "fused_batched_cutlass_epilogue_with_activation_scale_pack": _summary(
            fused_with_pack_times
        ),
        "fused_batched_cutlass_epilogue_per_batch_quantized_scales": _summary(
            fused_per_batch_times
        ),
        "speedup_prepared_median": statistics.median(base_times)
        / statistics.median(fused_prepared_times),
        "speedup_with_pack_median": statistics.median(base_times)
        / statistics.median(fused_with_pack_times),
        "speedup_per_batch_quantized_median": statistics.median(base_times)
        / statistics.median(fused_per_batch_times),
        "baseline_nan_count": int(torch.isnan(y0).sum().item()),
        "fused_prepared_nan_count": int(torch.isnan(y1).sum().item()),
        "fused_per_batch_quantized_nan_count": int(torch.isnan(y2).sum().item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff_per_batch_quantized": float(diff_per_batch.max().item()),
        "mean_abs_diff_per_batch_quantized": float(diff_per_batch.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "ltx"], default="smoke")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/kernel_fusion_benchmarks/nvfp4_batched_residual_gate_epilogue.json"
        ),
    )
    args = parser.parse_args()
    if args.mode == "smoke":
        cases = [("smoke_batched_ffn", 3, 512, 4096, 16384)]
    else:
        cases = [
            ("stage1_video_ffn_proj_out", 3, 15810, 4096, 16384),
            ("stage1_video_attn_to_out", 3, 15810, 4096, 4096),
        ]
    results = []
    for case in cases:
        try:
            results.append(
                run_case(*case, dtype=torch.bfloat16, warmup=args.warmup, iters=args.iters)
            )
        except Exception as exc:
            name, batch_size, m_per_batch, n, k = case
            results.append(
                {
                    "name": name,
                    "shape": {
                        "batch_size": batch_size,
                        "m_per_batch": m_per_batch,
                        "n": n,
                        "k": k,
                    },
                    "error": repr(exc),
                }
            )
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "mode": args.mode,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
