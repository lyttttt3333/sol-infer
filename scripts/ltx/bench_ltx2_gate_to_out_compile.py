import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def gate_to_out(attn_out, gate_logits, out_w, out_b, heads: int, dim_head: int):
    m = attn_out.shape[0]
    scaled = attn_out.view(m, heads, dim_head) * (
        2.0 * torch.sigmoid(gate_logits)
    ).unsqueeze(-1)
    return F.linear(scaled.reshape(m, heads * dim_head), out_w, out_b)


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
    heads: int,
    dim_head: int,
    repeats: int,
    warmup: int,
) -> dict:
    torch.manual_seed(123)
    hidden = heads * dim_head
    attn_out = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16)
    gate_logits = torch.randn((m, heads), device="cuda", dtype=torch.bfloat16)
    out_w = torch.randn((hidden, hidden), device="cuda", dtype=torch.bfloat16) / (
        hidden**0.5
    )
    out_b = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    args = (attn_out, gate_logits, out_w, out_b, heads, dim_head)

    eager = gate_to_out
    compiled = torch.compile(
        gate_to_out,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
    y0 = eager(*args)
    torch.cuda.synchronize()
    y1 = compiled(*args)
    torch.cuda.synchronize()
    diff = (y0.float() - y1.float()).abs()

    eager_times = _time_cuda(lambda: eager(*args), repeats=repeats, warmup=warmup)
    compiled_times = _time_cuda(
        lambda: compiled(*args), repeats=repeats, warmup=warmup
    )
    eager_avg = sum(eager_times) / len(eager_times)
    compiled_avg = sum(compiled_times) / len(compiled_times)
    return {
        "shape": {
            "m": m,
            "heads": heads,
            "dim_head": dim_head,
            "hidden": hidden,
        },
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "eager_ms": eager_times,
        "compiled_ms": compiled_times,
        "eager_avg_ms": eager_avg,
        "compiled_avg_ms": compiled_avg,
        "speedup": eager_avg / compiled_avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/ltx23-gate-to-out-compile-microbench/result.json",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    shapes = [
        ("stage1_video", 3 * 15810, 32, 128),
        ("stage2_video", 63240, 32, 128),
        ("stage1_audio", 3 * 251, 32, 64),
        ("stage2_audio", 251, 32, 64),
    ]
    results = {}
    for name, m, heads, dim_head in shapes:
        print(
            f"benchmarking {name}: m={m} heads={heads} dim_head={dim_head}",
            flush=True,
        )
        results[name] = _bench_shape(
            name,
            m,
            heads,
            dim_head,
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
