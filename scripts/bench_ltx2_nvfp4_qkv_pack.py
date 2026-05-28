import argparse
import json
import statistics
from pathlib import Path

import flashinfer
import torch


DTYPE = torch.bfloat16
GLOBAL_SCALE = 512.0
ALPHA_VALUE = 1.0 / (GLOBAL_SCALE * GLOBAL_SCALE)
ALPHA_RATIOS = (1.0, 1.73, 1.21)


def _time_cuda(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        out = fn()
        torch.cuda.synchronize()
        del out

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del out
    return times


def _scale_shape(m: int, k: int) -> tuple[int, int]:
    rounded_m = ((m + 127) // 128) * 128
    scale_k = k // 16
    rounded_scale_k = ((scale_k + 3) // 4) * 4
    return rounded_m, rounded_scale_k


def _stats(times: list[float]) -> dict[str, float | list[float]]:
    median = float(statistics.median(times))
    return {
        "median_ms": median,
        "avg_ms": float(sum(times) / len(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "samples_ms": times,
    }


def _build_weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.ones(_scale_shape(n, k), device="cuda", dtype=torch.float8_e4m3fn)
    bias = torch.randn((n,), device="cuda", dtype=DTYPE)
    return weight, scale, bias


def _fp4_mm(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    return flashinfer.mm_fp4(
        x_fp4,
        weight.T,
        x_scale,
        weight_scale.T,
        alpha,
        DTYPE,
        backend=backend,
    )


def _bench_case(
    name: str, m: int, k: int, n: int, repeats: int, warmup: int, backend: str
) -> dict:
    torch.manual_seed(20260528 + m + k + n)
    x = torch.randn((m, k), device="cuda", dtype=DTYPE)
    q_alpha = torch.tensor(
        ALPHA_VALUE * ALPHA_RATIOS[0], device="cuda", dtype=torch.float32
    )
    k_alpha = torch.tensor(
        ALPHA_VALUE * ALPHA_RATIOS[1], device="cuda", dtype=torch.float32
    )
    v_alpha = torch.tensor(
        ALPHA_VALUE * ALPHA_RATIOS[2], device="cuda", dtype=torch.float32
    )
    packed_output_scale = torch.cat(
        (
            torch.full((n,), ALPHA_RATIOS[0], device="cuda", dtype=DTYPE),
            torch.full((n,), ALPHA_RATIOS[1], device="cuda", dtype=DTYPE),
            torch.full((n,), ALPHA_RATIOS[2], device="cuda", dtype=DTYPE),
        )
    )
    global_scale = torch.tensor(GLOBAL_SCALE, device="cuda", dtype=torch.float32)

    q_w, q_sf, q_b = _build_weight(n, k)
    k_w, k_sf, k_b = _build_weight(n, k)
    v_w, v_sf, v_b = _build_weight(n, k)
    packed_w = torch.cat((q_w, k_w, v_w), dim=0).contiguous()
    packed_sf = torch.cat((q_sf, k_sf, v_sf), dim=0).contiguous()
    packed_b = torch.cat((q_b, k_b, v_b), dim=0).contiguous()

    x_fp4, x_sf = flashinfer.fp4_quantize(x, global_scale)
    if x_sf.dtype == torch.uint8:
        x_sf = x_sf.view(torch.float8_e4m3fn)

    def separate_quant_each():
        q_x, q_x_sf = flashinfer.fp4_quantize(x, global_scale)
        k_x, k_x_sf = flashinfer.fp4_quantize(x, global_scale)
        v_x, v_x_sf = flashinfer.fp4_quantize(x, global_scale)
        if q_x_sf.dtype == torch.uint8:
            q_x_sf = q_x_sf.view(torch.float8_e4m3fn)
            k_x_sf = k_x_sf.view(torch.float8_e4m3fn)
            v_x_sf = v_x_sf.view(torch.float8_e4m3fn)
        return (
            _fp4_mm(q_x, q_w, q_x_sf, q_sf, q_alpha, backend) + q_b,
            _fp4_mm(k_x, k_w, k_x_sf, k_sf, k_alpha, backend) + k_b,
            _fp4_mm(v_x, v_w, v_x_sf, v_sf, v_alpha, backend) + v_b,
        )

    def separate_shared_quant():
        return (
            _fp4_mm(x_fp4, q_w, x_sf, q_sf, q_alpha, backend) + q_b,
            _fp4_mm(x_fp4, k_w, x_sf, k_sf, k_alpha, backend) + k_b,
            _fp4_mm(x_fp4, v_w, x_sf, v_sf, v_alpha, backend) + v_b,
        )

    def packed_qkv_rescaled():
        y = _fp4_mm(x_fp4, packed_w, x_sf, packed_sf, q_alpha, backend)
        y = y * packed_output_scale + packed_b
        return y.split(n, dim=-1)

    reference = separate_shared_quant()
    candidate = packed_qkv_rescaled()
    diffs = {}
    for idx, (a, b) in enumerate(zip(reference, candidate)):
        delta = (a.float() - b.float()).abs()
        diffs[f"out{idx}_max_abs_diff"] = float(delta.max().item())
        diffs[f"out{idx}_mean_abs_diff"] = float(delta.mean().item())

    quant_each_times = _time_cuda(separate_quant_each, repeats, warmup)
    shared_times = _time_cuda(separate_shared_quant, repeats, warmup)
    packed_times = _time_cuda(packed_qkv_rescaled, repeats, warmup)

    quant_each = _stats(quant_each_times)
    shared = _stats(shared_times)
    packed = _stats(packed_times)
    return {
        "shape": {"name": name, "m": m, "k": k, "n": n, "packed_n": 3 * n},
        "backend": backend,
        "separate_quant_each": quant_each,
        "separate_shared_quant": shared,
        "packed_qkv_rescaled": packed,
        "packed_speedup_vs_quant_each": quant_each["median_ms"] / packed["median_ms"],
        "packed_speedup_vs_shared_quant": shared["median_ms"] / packed["median_ms"],
        **diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-nvfp4-qkv-pack/result.json")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--backend", default="cudnn", choices=["auto", "cudnn", "trtllm"])
    args = parser.parse_args()

    torch.cuda.set_device(0)
    cases = [
        ("stage1_video_self_attn_qkv", 3 * 15810, 4096, 4096),
        ("stage2_video_self_attn_qkv", 63240, 4096, 4096),
    ]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "global_scale": GLOBAL_SCALE,
        "results": {
            name: _bench_case(name, m, k, n, args.repeats, args.warmup, args.backend)
            for name, m, k, n in cases
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
