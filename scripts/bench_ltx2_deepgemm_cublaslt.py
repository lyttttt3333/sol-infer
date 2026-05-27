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
    return last.detach().clone(), times


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
    torch.manual_seed(20260525 + m + k + n)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / (k**0.5)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    bias_float_2d = bias.float().view(1, n).expand(m, n)
    weight_t = weight.t().contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    out_float = torch.empty((m, n), device="cuda", dtype=torch.float32)
    out_cast = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    ref_linear, linear_times = time_cuda(lambda: F.linear(x, weight, bias), repeats, warmup)
    ref_matmul, matmul_times = time_cuda(lambda: x @ weight_t, repeats, warmup)

    def torch_mm_out_no_bias():
        torch.mm(x, weight_t, out=out)
        return out

    def torch_mm_out_then_bias():
        torch.mm(x, weight_t, out=out)
        out.add_(bias)
        return out

    torch_mm_no_bias, torch_mm_no_bias_times = time_cuda(
        torch_mm_out_no_bias, repeats, warmup
    )
    torch_mm_bias, torch_mm_bias_times = time_cuda(
        torch_mm_out_then_bias, repeats, warmup
    )

    result = {
        "shape": {"m": m, "k": k, "n": n},
        "torch_linear_bias": stats(linear_times),
        "torch_matmul_no_bias": stats(matmul_times),
        "torch_mm_out_no_bias": {
            "stats": stats(torch_mm_no_bias_times),
            **diff_stats(ref_matmul, torch_mm_no_bias),
        },
        "torch_mm_out_then_bias": {
            "stats": stats(torch_mm_bias_times),
            **diff_stats(ref_linear, torch_mm_bias),
        },
        "cublaslt": {},
    }

    variants = {}

    def cublaslt_no_bias():
        deep_gemm.cublaslt_gemm_nt(x, weight, out, None)
        return out

    variants["nt_no_bias"] = (cublaslt_no_bias, ref_matmul, "torch_matmul_no_bias")

    def cublaslt_bias_vec():
        deep_gemm.cublaslt_gemm_nt(x, weight, out, bias)
        return out

    variants["nt_bias_vec"] = (cublaslt_bias_vec, ref_linear, "torch_linear_bias")

    def cublaslt_then_bias():
        deep_gemm.cublaslt_gemm_nt(x, weight, out, None)
        out.add_(bias)
        return out

    variants["nt_then_add_bias"] = (
        cublaslt_then_bias,
        ref_linear,
        "torch_linear_bias",
    )

    def cublaslt_bias_float_out_cast():
        deep_gemm.cublaslt_gemm_nt(x, weight, out_float, bias_float_2d)
        out_cast.copy_(out_float)
        return out_cast

    variants["nt_bias_float_out_cast"] = (
        cublaslt_bias_float_out_cast,
        ref_linear,
        "torch_linear_bias",
    )

    for key, (fn, ref, baseline_key) in variants.items():
        try:
            cand, times = time_cuda(fn, repeats, warmup)
            avg = sum(times) / len(times)
            base = result[baseline_key]
            base_avg = base["avg_ms"] if "avg_ms" in base else base["stats"]["avg_ms"]
            result["cublaslt"][key] = {
                "ok": True,
                "stats": stats(times),
                "speedup_vs_baseline": base_avg / avg,
                **diff_stats(ref, cand),
            }
        except Exception as exc:
            result["cublaslt"][key] = {"ok": False, "error": repr(exc)}

    del (
        x,
        weight,
        weight_t,
        bias,
        bias_float_2d,
        out,
        out_float,
        out_cast,
        ref_linear,
        ref_matmul,
    )
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="outputs/ltx23-deepgemm-cublaslt-microbench/result.json"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--include-attn", action="store_true")
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
