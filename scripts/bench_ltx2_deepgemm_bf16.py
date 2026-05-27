import argparse
import json
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


DEFAULT_CUDA_HOME = (
    Path(__file__).resolve().parents[1]
    / ".conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
)


def ensure_cuda_home() -> None:
    if "CUDA_HOME" not in os.environ and DEFAULT_CUDA_HOME.exists():
        os.environ["CUDA_HOME"] = str(DEFAULT_CUDA_HOME)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
        lib_path = f"{cuda_home}/lib"
        old_ld = os.environ.get("LD_LIBRARY_PATH")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_path}:{old_ld}" if old_ld else lib_path
        )


def time_cuda(fn: Callable[[], torch.Tensor], repeats: int, warmup: int):
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


def diff_stats(ref: torch.Tensor, cand: torch.Tensor):
    diff = (ref.float() - cand.float()).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "bit_exact": bool(torch.equal(ref, cand)),
    }


def bench_shape(deep_gemm, name: str, m: int, k: int, n: int, repeats: int, warmup: int):
    print(f"benchmarking {name}: m={m} k={k} n={n}", flush=True)
    torch.manual_seed(20260524 + m + k + n)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / (k**0.5)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    weight_t = weight.t().contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    ref_linear, linear_times = time_cuda(
        lambda: F.linear(x, weight, bias), repeats, warmup
    )
    ref_matmul, matmul_times = time_cuda(lambda: x @ weight_t, repeats, warmup)

    def torch_mm_out_no_bias():
        torch.mm(x, weight_t, out=out)
        return out

    def torch_mm_out_bias():
        torch.mm(x, weight_t, out=out)
        out.add_(bias)
        return out

    torch_mm_out, torch_mm_out_times = time_cuda(
        torch_mm_out_no_bias, repeats, warmup
    )
    torch_mm_bias, torch_mm_bias_times = time_cuda(torch_mm_out_bias, repeats, warmup)

    result = {
        "shape": {"m": m, "k": k, "n": n},
        "torch_linear_bias": stats(linear_times),
        "torch_matmul_no_bias": stats(matmul_times),
        "torch_mm_out_no_bias": {
            "stats": stats(torch_mm_out_times),
            **diff_stats(ref_matmul, torch_mm_out),
        },
        "torch_mm_out_then_bias": {
            "stats": stats(torch_mm_bias_times),
            **diff_stats(ref_linear, torch_mm_bias),
        },
        "deepgemm": {},
        "cublaslt": {},
    }

    compiled_dims_variants = ("", "mnk", "nk")
    for compiled_dims in compiled_dims_variants:
        key = f"nt_no_bias_compiled_{compiled_dims or 'none'}"

        def dg_no_bias(compiled_dims=compiled_dims):
            deep_gemm.bf16_gemm_nt(x, weight, out, None, compiled_dims)
            return out

        try:
            cand, times = time_cuda(dg_no_bias, repeats, warmup)
            avg = sum(times) / len(times)
            result["deepgemm"][key] = {
                "ok": True,
                "stats": stats(times),
                "speedup_vs_torch_matmul": (
                    result["torch_matmul_no_bias"]["avg_ms"] / avg
                ),
                "speedup_vs_torch_mm_out": (
                    result["torch_mm_out_no_bias"]["stats"]["avg_ms"] / avg
                ),
                **diff_stats(ref_matmul, cand),
            }
        except Exception as exc:
            result["deepgemm"][key] = {"ok": False, "error": repr(exc)}

        key = f"nt_bias_vec_compiled_{compiled_dims or 'none'}"

        def dg_bias_vec(compiled_dims=compiled_dims):
            deep_gemm.bf16_gemm_nt(x, weight, out, bias, compiled_dims)
            return out

        try:
            cand, times = time_cuda(dg_bias_vec, repeats, warmup)
            avg = sum(times) / len(times)
            result["deepgemm"][key] = {
                "ok": True,
                "stats": stats(times),
                "speedup_vs_torch_linear": (
                    result["torch_linear_bias"]["avg_ms"] / avg
                ),
                **diff_stats(ref_linear, cand),
            }
        except Exception as exc:
            result["deepgemm"][key] = {"ok": False, "error": repr(exc)}

        key = f"nt_then_add_bias_compiled_{compiled_dims or 'none'}"

        def dg_then_add(compiled_dims=compiled_dims):
            deep_gemm.bf16_gemm_nt(x, weight, out, None, compiled_dims)
            out.add_(bias)
            return out

        try:
            cand, times = time_cuda(dg_then_add, repeats, warmup)
            avg = sum(times) / len(times)
            result["deepgemm"][key] = {
                "ok": True,
                "stats": stats(times),
                "speedup_vs_torch_linear": (
                    result["torch_linear_bias"]["avg_ms"] / avg
                ),
                "speedup_vs_torch_mm_out_then_bias": (
                    result["torch_mm_out_then_bias"]["stats"]["avg_ms"] / avg
                ),
                **diff_stats(ref_linear, cand),
            }
        except Exception as exc:
            result["deepgemm"][key] = {"ok": False, "error": repr(exc)}

    for compiled_dims in compiled_dims_variants:
        key = f"nt_no_bias_compiled_{compiled_dims or 'none'}"

        def cublaslt_no_bias(compiled_dims=compiled_dims):
            deep_gemm.cublaslt_gemm_nt(x, weight, out, None, compiled_dims)
            return out

        try:
            cand, times = time_cuda(cublaslt_no_bias, repeats, warmup)
            avg = sum(times) / len(times)
            result["cublaslt"][key] = {
                "ok": True,
                "stats": stats(times),
                "speedup_vs_torch_matmul": (
                    result["torch_matmul_no_bias"]["avg_ms"] / avg
                ),
                "speedup_vs_torch_mm_out": (
                    result["torch_mm_out_no_bias"]["stats"]["avg_ms"] / avg
                ),
                **diff_stats(ref_matmul, cand),
            }
        except Exception as exc:
            result["cublaslt"][key] = {"ok": False, "error": repr(exc)}

    del x, weight, weight_t, bias, out
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="outputs/ltx23-deepgemm-bf16-microbench/result.json"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--include-attn",
        action="store_true",
        help="Also test square hidden-dim projection shapes.",
    )
    args = parser.parse_args()

    ensure_cuda_home()
    import deep_gemm

    torch.cuda.set_device(0)
    shapes = [
        ("stage1_ffn_proj_in", 3 * 15810, 4096, 16384),
        ("stage1_ffn_proj_out", 3 * 15810, 16384, 4096),
        ("stage2_ffn_proj_in", 63240, 4096, 16384),
        ("stage2_ffn_proj_out", 63240, 16384, 4096),
    ]
    if args.include_attn:
        shapes.extend(
            [
                ("stage1_attn_square", 3 * 15810, 4096, 4096),
                ("stage2_attn_square", 63240, 4096, 4096),
            ]
        )

    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "deep_gemm": getattr(deep_gemm, "__file__", None),
        "results": {
            name: bench_shape(deep_gemm, name, m, k, n, args.repeats, args.warmup)
            for name, m, k, n in shapes
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
