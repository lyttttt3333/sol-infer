#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark LTX2 video self-attention dense FA vs piecewise sparse kernels.

This intentionally bypasses the full pipeline. Shapes match non-HQ 1080p
LTX2 two-stage video self attention:
  stage1: half-res 960x544 latent grid -> 30 * 17 * 30 = 15300 tokens
  stage2: full-res 1920x1088 latent grid -> 30 * 34 * 60 = 61200 tokens
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Callable

import torch


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def _time_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
        # Keep the result live until after launch; no value sync here.
        assert out is not None
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        assert out is not None
        times.append(float(start.elapsed_time(end)))
    times_sorted = sorted(times)
    return {
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "p90_ms": _percentile(times_sorted, 0.90),
        "max_ms": max(times),
        "iters": iters,
        "warmup": warmup,
    }



def _set_piecewise_allocator() -> None:
    import triton
    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (
        _make_tma_allocator,
    )

    triton.set_allocator(_make_tma_allocator())


def _make_dense_fa(q_bthd: torch.Tensor, k_bthd: torch.Tensor, v_bthd: torch.Tensor, scale: float):
    from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
        flash_attn_varlen_func_op,
    )

    def run() -> torch.Tensor:
        return flash_attn_varlen_func_op(
            q=q_bthd,
            k=k_bthd,
            v=v_bthd,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=q_bthd.shape[1],
            max_seqlen_k=k_bthd.shape[1],
            softmax_scale=scale,
            causal=False,
            return_softmax_lse=False,
            ver=4,
        )

    return run


def _make_dense_sdpa(q_bhtd: torch.Tensor, k_bhtd: torch.Tensor, v_bhtd: torch.Tensor, scale: float):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def run() -> torch.Tensor:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return torch.nn.functional.scaled_dot_product_attention(
                q_bhtd,
                k_bhtd,
                v_bhtd,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )

    return run


def _precompute_piecewise(
    q_bhtd: torch.Tensor,
    k_bhtd: torch.Tensor,
    v_bhtd: torch.Tensor,
    *,
    block_size: int,
    density: float,
    scale: float,
    approx_remainder: bool,
):
    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (
        chunk_reduce_qkv,
        taylor_error_block_indices,
    )

    _set_piecewise_allocator()
    qc, kc, vc, k_var = chunk_reduce_qkv(
        q=q_bhtd,
        k=k_bhtd,
        v=v_bhtd,
        block_size=block_size,
        include_v_centroid=approx_remainder,
    )
    block_indices = taylor_error_block_indices(
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=density,
        scale=scale,
    )
    torch.cuda.synchronize()
    return kc, vc, block_indices


def _make_piecewise_fwd_only(
    q_bhtd: torch.Tensor,
    k_bhtd: torch.Tensor,
    v_bhtd: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor | None,
    block_indices: torch.Tensor,
    *,
    block_size: int,
    scale: float,
    approx_remainder: bool,
):
    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (
        piecewise_attn_fwd,
    )

    def run() -> torch.Tensor:
        _set_piecewise_allocator()
        out, _lse = piecewise_attn_fwd(
            q=q_bhtd,
            k=k_bhtd,
            v=v_bhtd,
            kc=kc,
            vc=vc if vc is not None else kc,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
            approx_remainder=approx_remainder,
        )
        return out

    return run


def _make_piecewise_total(
    q_bhtd: torch.Tensor,
    k_bhtd: torch.Tensor,
    v_bhtd: torch.Tensor,
    *,
    block_size: int,
    density: float,
    scale: float,
    approx_remainder: bool,
):
    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (
        chunk_reduce_qkv,
        piecewise_attn_fwd,
        taylor_error_block_indices,
    )

    def run() -> torch.Tensor:
        _set_piecewise_allocator()
        qc, kc, vc, k_var = chunk_reduce_qkv(
            q=q_bhtd,
            k=k_bhtd,
            v=v_bhtd,
            block_size=block_size,
            include_v_centroid=approx_remainder,
        )
        block_indices = taylor_error_block_indices(
            qc=qc,
            kc=kc,
            k_var=k_var,
            density=density,
            scale=scale,
        )
        out, _lse = piecewise_attn_fwd(
            q=q_bhtd,
            k=k_bhtd,
            v=v_bhtd,
            kc=kc,
            vc=vc if vc is not None else kc,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
            approx_remainder=approx_remainder,
        )
        return out

    return run


def bench_shape(args: argparse.Namespace, *, name: str, tokens: int, density: float) -> dict:
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    shape = (args.batch, tokens, args.heads, args.head_dim)
    q_bthd = torch.randn(shape, device=device, dtype=dtype)
    k_bthd = torch.randn(shape, device=device, dtype=dtype)
    v_bthd = torch.randn(shape, device=device, dtype=dtype)
    q_bhtd = q_bthd.transpose(1, 2).contiguous()
    k_bhtd = k_bthd.transpose(1, 2).contiguous()
    v_bhtd = v_bthd.transpose(1, 2).contiguous()
    scale = args.head_dim ** -0.5

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dense_fa = None
    dense_error = None
    try:
        dense_fa = _time_cuda(
            _make_dense_fa(q_bthd, k_bthd, v_bthd, scale),
            warmup=args.warmup,
            iters=args.iters,
        )
    except Exception as exc:  # Keep SDPA fallback data if FA4 is unavailable.
        dense_error = repr(exc)

    dense_sdpa = None
    if args.bench_sdpa:
        dense_sdpa = _time_cuda(
            _make_dense_sdpa(q_bhtd, k_bhtd, v_bhtd, scale),
            warmup=max(1, args.warmup // 2),
            iters=max(1, args.iters // 2),
        )

    kc, vc, block_indices = _precompute_piecewise(
        q_bhtd,
        k_bhtd,
        v_bhtd,
        block_size=args.block_size,
        density=density,
        scale=scale,
        approx_remainder=args.approx_remainder,
    )
    exact_density = block_indices.shape[-1] / kc.shape[2]

    sparse_fwd_only = _time_cuda(
        _make_piecewise_fwd_only(
            q_bhtd,
            k_bhtd,
            v_bhtd,
            kc,
            vc,
            block_indices,
            block_size=args.block_size,
            scale=scale,
            approx_remainder=args.approx_remainder,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    sparse_total = _time_cuda(
        _make_piecewise_total(
            q_bhtd,
            k_bhtd,
            v_bhtd,
            block_size=args.block_size,
            density=density,
            scale=scale,
            approx_remainder=args.approx_remainder,
        ),
        warmup=max(1, args.warmup // 2),
        iters=max(1, args.iters // 2),
    )

    dense_ref = dense_fa or dense_sdpa
    out = {
        "name": name,
        "tokens": tokens,
        "batch": args.batch,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "block_size": args.block_size,
        "configured_density": density,
        "exact_density": exact_density,
        "actual_sparsity": 1.0 - exact_density,
        "dense_fa4": dense_fa,
        "dense_fa4_error": dense_error,
        "dense_sdpa": dense_sdpa,
        "piecewise_fwd_only": sparse_fwd_only,
        "piecewise_total_route_plus_fwd": sparse_total,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    if dense_ref is not None:
        out["speedup_piecewise_fwd_only_vs_dense"] = dense_ref["mean_ms"] / sparse_fwd_only["mean_ms"]
        out["speedup_piecewise_total_vs_dense"] = dense_ref["mean_ms"] / sparse_total["mean_ms"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx2_piecewise_attn_kernel_bench/result.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bench-sdpa", action="store_true")
    parser.add_argument("--approx-remainder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="name:tokens:density. Can be repeated. Defaults to 1080p stage1/2 sparse schedule shapes.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_grad_enabled(False)

    shapes = args.shape or [
        "stage1_1080p_start_sparse:15300:0.2",
        "stage1_1080p_end_sparse:15300:0.1",
        "stage2_1080p_sparse90:61200:0.1",
    ]
    results = []
    for spec in shapes:
        name, tokens_text, density_text = spec.split(":")
        print(f"[bench] {name} tokens={tokens_text} density={density_text}", flush=True)
        results.append(
            bench_shape(
                args,
                name=name,
                tokens=int(tokens_text),
                density=float(density_text),
            )
        )

    payload = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "settings": vars(args),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
