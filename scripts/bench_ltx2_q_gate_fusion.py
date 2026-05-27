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


def _bench_shape(
    name: str,
    m: int,
    k: int,
    q_out: int,
    gate_out: int,
    repeats: int,
    warmup: int,
) -> dict:
    torch.manual_seed(123)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    q_w = torch.randn((q_out, k), device="cuda", dtype=torch.bfloat16) / (k**0.5)
    q_b = torch.randn((q_out,), device="cuda", dtype=torch.bfloat16)
    gate_w = torch.randn((gate_out, k), device="cuda", dtype=torch.bfloat16) / (
        k**0.5
    )
    gate_b = torch.randn((gate_out,), device="cuda", dtype=torch.bfloat16)
    fused_w = torch.cat([q_w, gate_w], dim=0).contiguous()
    fused_b = torch.cat([q_b, gate_b], dim=0).contiguous()

    def separate():
        return F.linear(x, q_w, q_b), F.linear(x, gate_w, gate_b)

    def fused():
        y = F.linear(x, fused_w, fused_b)
        return y[:, :q_out], y[:, q_out:]

    q0, g0 = separate()
    q1, g1 = fused()
    torch.cuda.synchronize()
    q_diff = (q0.float() - q1.float()).abs()
    g_diff = (g0.float() - g1.float()).abs()

    separate_times = _time_cuda(separate, repeats=repeats, warmup=warmup)
    fused_times = _time_cuda(fused, repeats=repeats, warmup=warmup)
    sep_avg = sum(separate_times) / len(separate_times)
    fused_avg = sum(fused_times) / len(fused_times)
    return {
        "shape": {
            "m": m,
            "k": k,
            "q_out": q_out,
            "gate_out": gate_out,
        },
        "max_abs_diff_q": float(q_diff.max().item()),
        "mean_abs_diff_q": float(q_diff.mean().item()),
        "max_abs_diff_gate": float(g_diff.max().item()),
        "mean_abs_diff_gate": float(g_diff.mean().item()),
        "separate_ms": separate_times,
        "fused_ms": fused_times,
        "separate_avg_ms": sep_avg,
        "fused_avg_ms": fused_avg,
        "speedup": sep_avg / fused_avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/ltx23-q-gate-fusion-microbench/result.json",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    shapes = [
        ("stage1_video_heads32", 3 * 15810, 4096, 4096, 32),
        ("stage2_video_heads32", 63240, 4096, 4096, 32),
        ("stage1_video_heads64", 3 * 15810, 4096, 4096, 64),
        ("stage2_video_heads64", 63240, 4096, 4096, 64),
        ("stage1_audio_heads16", 3 * 251, 2048, 2048, 16),
        ("stage2_audio_heads16", 251, 2048, 2048, 16),
    ]

    results = {}
    for name, m, k, q_out, gate_out in shapes:
        print(f"benchmarking {name}: m={m} k={k} q={q_out} gate={gate_out}", flush=True)
        results[name] = _bench_shape(
            name,
            m,
            k,
            q_out,
            gate_out,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        torch.cuda.empty_cache()

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
