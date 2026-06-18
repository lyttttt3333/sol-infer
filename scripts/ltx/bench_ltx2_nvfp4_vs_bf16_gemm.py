import argparse
import json
import statistics
from pathlib import Path

import flashinfer
import sgl_kernel
import torch
import torch.nn.functional as F


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _time_cuda(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        y = fn()
        torch.cuda.synchronize()
        del y

    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        y = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del y
    return times


def _scale_shape(m: int, k: int) -> tuple[int, int]:
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_k = k // 16
    rounded_scale_k = ((scale_k + 4 - 1) // 4) * 4
    return rounded_m, rounded_scale_k


def _build_fp4_operands(m: int, k: int, n: int):
    # GEMM throughput does not depend on the numeric distribution. Constructing
    # packed operands directly avoids measuring or JIT-compiling activation
    # quantization in this GEMM-only benchmark.
    x_fp4 = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8)
    w_fp4 = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    x_sf = torch.ones(_scale_shape(m, k), device="cuda", dtype=torch.float8_e4m3fn)
    w_sf = torch.ones(_scale_shape(n, k), device="cuda", dtype=torch.float8_e4m3fn)
    alpha = torch.ones((), device="cuda", dtype=torch.float32)
    return x_fp4, w_fp4, x_sf, w_sf, alpha


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _bench_shape(name: str, m: int, k: int, n: int, repeats: int, warmup: int) -> dict:
    torch.manual_seed(123)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / (k**0.5)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    weight_t = weight.t()
    x_fp4, w_fp4, x_sf, w_sf, alpha = _build_fp4_operands(m, k, n)

    def torch_linear():
        return F.linear(x, weight, bias)

    providers = {
        "torch_bf16": torch_linear,
        "flashinfer_bf16_cudnn": lambda: flashinfer.gemm.mm_bf16(
            x, weight_t, bias=bias, out_dtype=torch.bfloat16, backend="cudnn"
        ),
        "flashinfer_bf16_cublaslt": lambda: flashinfer.gemm.mm_bf16(
            x, weight_t, bias=bias, out_dtype=torch.bfloat16, backend="cublaslt"
        ),
        "sgl_kernel_nvfp4_cutlass": lambda: sgl_kernel.cutlass_scaled_fp4_mm(
            x_fp4, w_fp4, x_sf, w_sf, alpha, torch.bfloat16
        ),
        "flashinfer_nvfp4_cudnn": lambda: flashinfer.mm_fp4(
            x_fp4, w_fp4.T, x_sf, w_sf.T, alpha, torch.bfloat16, backend="cudnn"
        ),
        "flashinfer_nvfp4_trtllm": lambda: flashinfer.mm_fp4(
            x_fp4, w_fp4.T, x_sf, w_sf.T, alpha, torch.bfloat16, backend="trtllm"
        ),
        "flashinfer_nvfp4_auto": lambda: flashinfer.mm_fp4(
            x_fp4, w_fp4.T, x_sf, w_sf.T, alpha, torch.bfloat16, backend="auto"
        ),
    }

    results = {}
    for provider, fn in providers.items():
        torch.cuda.empty_cache()
        try:
            times = _time_cuda(fn, repeats=repeats, warmup=warmup)
            median_ms = _median(times)
            results[provider] = {
                "ok": True,
                "median_ms": median_ms,
                "avg_ms": float(sum(times) / len(times)),
                "min_ms": float(min(times)),
                "max_ms": float(max(times)),
                "tflops": float((2 * m * n * k) / (median_ms / 1e3) / 1e12),
            }
        except Exception as exc:
            results[provider] = {"ok": False, "error": repr(exc)}

    bf16_candidates = [
        v["median_ms"]
        for key, v in results.items()
        if key.startswith(("torch_bf16", "flashinfer_bf16")) and v.get("ok")
    ]
    fp4_candidates = [
        v["median_ms"]
        for key, v in results.items()
        if "nvfp4" in key and v.get("ok")
    ]
    best_bf16 = min(bf16_candidates) if bf16_candidates else None
    best_fp4 = min(fp4_candidates) if fp4_candidates else None

    return {
        "shape": {"name": name, "m": m, "k": k, "n": n},
        "best_bf16_ms": best_bf16,
        "best_nvfp4_ms": best_fp4,
        "best_nvfp4_speedup_vs_bf16": (best_bf16 / best_fp4)
        if best_bf16 and best_fp4
        else None,
        "providers": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/ltx23-nvfp4-vs-bf16-gemm-microbench/result.json",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    shapes = [
        ("stage1_ffn_proj_in", 3 * 15810, 4096, 16384),
        ("stage1_ffn_proj_out", 3 * 15810, 16384, 4096),
        ("stage2_ffn_proj_in", 63240, 4096, 16384),
        ("stage2_ffn_proj_out", 63240, 16384, 4096),
    ]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "results": {
            name: _bench_shape(name, m, k, n, args.repeats, args.warmup)
            for name, m, k, n in shapes
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
