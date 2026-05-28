import argparse
import json
import statistics
from pathlib import Path

import flashinfer
import torch
import torch.nn.functional as F

from sglang.jit_kernel.nvfp4 import scaled_fp4_quant


DTYPE = torch.bfloat16
GLOBAL_SCALE = 512.0
ALPHA_VALUE = 1.0 / (GLOBAL_SCALE * GLOBAL_SCALE)


def _time_cuda(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        y = fn()
        torch.cuda.synchronize()
        del y

    times = []
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


def _build_fp4_weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp4 = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    weight_sf = torch.ones(_scale_shape(n, k), device="cuda", dtype=torch.float8_e4m3fn)
    return weight_fp4, weight_sf


def _stats(times: list[float]) -> dict[str, float | list[float]]:
    median = float(statistics.median(times))
    return {
        "median_ms": median,
        "avg_ms": float(sum(times) / len(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "samples_ms": times,
    }


def _bench_case(name: str, m: int, repeats: int, warmup: int) -> dict:
    k = 4096
    q_n = 4096
    gate_n = 32
    torch.manual_seed(20260528 + m)
    x = torch.randn((m, k), device="cuda", dtype=DTYPE)
    q_w_fp4, q_w_sf = _build_fp4_weight(q_n, k)
    gate_w_fp4, gate_w_sf = _build_fp4_weight(gate_n, k)
    gate_w_bf16 = torch.randn((gate_n, k), device="cuda", dtype=DTYPE) / (k ** 0.5)
    gate_bias = torch.randn((gate_n,), device="cuda", dtype=DTYPE)
    alpha = torch.tensor(ALPHA_VALUE, device="cuda", dtype=torch.float32)
    global_scale = torch.tensor(GLOBAL_SCALE, device="cuda", dtype=torch.float32)

    def current_flashinfer_quant_q_fp4_gate_bf16():
        x_fp4, x_sf = flashinfer.fp4_quantize(x, global_scale)
        if x_sf.dtype == torch.uint8:
            x_sf = x_sf.view(torch.float8_e4m3fn)
        q = flashinfer.mm_fp4(
            x_fp4,
            q_w_fp4.T,
            x_sf,
            q_w_sf.T,
            alpha,
            DTYPE,
            backend="cudnn",
        )
        gate = F.linear(x, gate_w_bf16, gate_bias)
        return q, gate

    def shared_flashinfer_quant_q_fp4_gate_fp4():
        x_fp4, x_sf = flashinfer.fp4_quantize(x, global_scale)
        if x_sf.dtype == torch.uint8:
            x_sf = x_sf.view(torch.float8_e4m3fn)
        q = flashinfer.mm_fp4(
            x_fp4,
            q_w_fp4.T,
            x_sf,
            q_w_sf.T,
            alpha,
            DTYPE,
            backend="cudnn",
        )
        gate = flashinfer.mm_fp4(
            x_fp4,
            gate_w_fp4.T,
            x_sf,
            gate_w_sf.T,
            alpha,
            DTYPE,
            backend="cudnn",
        )
        gate = gate + gate_bias
        return q, gate

    def shared_sglang_quant_q_fp4_gate_fp4():
        x_fp4, x_sf = scaled_fp4_quant(x, global_scale)
        q = flashinfer.mm_fp4(
            x_fp4,
            q_w_fp4.T,
            x_sf,
            q_w_sf.T,
            alpha,
            DTYPE,
            backend="cudnn",
        )
        gate = flashinfer.mm_fp4(
            x_fp4,
            gate_w_fp4.T,
            x_sf,
            gate_w_sf.T,
            alpha,
            DTYPE,
            backend="cudnn",
        )
        gate = gate + gate_bias
        return q, gate

    providers = {
        "current_flashinfer_quant_q_fp4_gate_bf16": current_flashinfer_quant_q_fp4_gate_bf16,
        "shared_flashinfer_quant_q_fp4_gate_fp4": shared_flashinfer_quant_q_fp4_gate_fp4,
        "shared_sglang_quant_q_fp4_gate_fp4": shared_sglang_quant_q_fp4_gate_fp4,
    }
    results = {}
    for provider, fn in providers.items():
        torch.cuda.empty_cache()
        try:
            results[provider] = {"ok": True, **_stats(_time_cuda(fn, repeats, warmup))}
        except Exception as exc:
            results[provider] = {"ok": False, "error": repr(exc)}
        finally:
            torch.cuda.synchronize()

    current = results["current_flashinfer_quant_q_fp4_gate_bf16"].get("median_ms")
    candidates = {
        k: v["median_ms"]
        for k, v in results.items()
        if k != "current_flashinfer_quant_q_fp4_gate_bf16" and v.get("ok")
    }
    best_provider = min(candidates, key=candidates.get) if candidates else None
    best_ms = candidates[best_provider] if best_provider else None
    return {
        "shape": {"name": name, "m": m, "k": k, "q_n": q_n, "gate_n": gate_n},
        "current_median_ms": current,
        "best_shared_provider": best_provider,
        "best_shared_median_ms": best_ms,
        "speedup_shared_vs_current": (current / best_ms) if current and best_ms else None,
        "positive": bool(current and best_ms and current > best_ms),
        "providers": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-fp4-q-gate-shared/result.json")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cases = [
        ("stage1_video_q_gate", 3 * 15810),
        ("stage2_video_q_gate", 63240),
    ]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "global_scale": GLOBAL_SCALE,
        "results": {},
    }
    for name, m in cases:
        print(f"benchmarking {name}: m={m}", flush=True)
        payload["results"][name] = _bench_case(name, m, args.repeats, args.warmup)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
