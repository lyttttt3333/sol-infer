import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _time_cuda(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        y = fn()
        torch.cuda.synchronize()
        del y

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        y = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
        del y
    return times


def _bench_linear_shape(
    flashinfer_gemm,
    name: str,
    m: int,
    k: int,
    n: int,
    backends: list[str],
    repeats: int,
    warmup: int,
) -> dict:
    torch.manual_seed(123)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / (k**0.5)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    weight_t = weight.t()

    def torch_linear():
        return F.linear(x, weight, bias)

    torch_out = torch_linear()
    torch.cuda.synchronize()
    torch_times = _time_cuda(torch_linear, repeats=repeats, warmup=warmup)

    result: dict[str, object] = {
        "shape": {"m": m, "k": k, "n": n},
        "torch_ms": torch_times,
        "torch_avg_ms": sum(torch_times) / len(torch_times),
        "backends": {},
    }

    for backend in backends:
        torch.cuda.empty_cache()

        def flashinfer_linear():
            return flashinfer_gemm.mm_bf16(
                x,
                weight_t,
                bias=bias,
                out_dtype=torch.bfloat16,
                backend=backend,
            )

        try:
            fi_out = flashinfer_linear()
            torch.cuda.synchronize()
            diff = (torch_out.float() - fi_out.float()).abs()
            fi_times = _time_cuda(flashinfer_linear, repeats=repeats, warmup=warmup)
            avg = sum(fi_times) / len(fi_times)
            result["backends"][backend] = {
                "ok": True,
                "ms": fi_times,
                "avg_ms": avg,
                "speedup_vs_torch": result["torch_avg_ms"] / avg,
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
            }
        except Exception as exc:
            result["backends"][backend] = {
                "ok": False,
                "error": repr(exc),
            }
        finally:
            torch.cuda.synchronize()

    del x, weight, weight_t, bias, torch_out
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/ltx23-flashinfer-bf16-ffn-gemm-microbench/result.json",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--backends",
        default="cudnn,cublaslt,cutlass,auto",
        help="Comma-separated FlashInfer mm_bf16 backends.",
    )
    args = parser.parse_args()

    import flashinfer.gemm as flashinfer_gemm

    torch.cuda.set_device(0)
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]

    shapes = [
        ("stage1_ffn_proj_in", 3 * 15810, 4096, 16384),
        ("stage1_ffn_proj_out", 3 * 15810, 16384, 4096),
        ("stage2_ffn_proj_in", 63240, 4096, 16384),
        ("stage2_ffn_proj_out", 63240, 16384, 4096),
    ]
    results = {}
    for name, m, k, n in shapes:
        print(f"benchmarking {name}: m={m} k={k} n={n}", flush=True)
        results[name] = _bench_linear_shape(
            flashinfer_gemm,
            name,
            m,
            k,
            n,
            backends,
            repeats=args.repeats,
            warmup=args.warmup,
        )

    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
