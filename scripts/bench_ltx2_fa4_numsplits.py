import argparse
import json
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    flash_attn_varlen_func_op,
)


def _time_cuda(fn, repeats: int, warmup: int) -> tuple[torch.Tensor, list[float]]:
    last = None
    for _ in range(warmup):
        last = fn()
    torch.cuda.synchronize()

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        last = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    assert last is not None
    return last, times


def _bench_shape(
    name: str,
    batch: int,
    seq: int,
    heads: int,
    dim: int,
    splits: list[int],
    repeats: int,
    warmup: int,
    sm_margin: int = 0,
) -> dict:
    torch.manual_seed(123)
    dtype = torch.bfloat16
    q = torch.randn((batch, seq, heads, dim), device="cuda", dtype=dtype)
    k = torch.randn((batch, seq, heads, dim), device="cuda", dtype=dtype)
    v = torch.randn((batch, seq, heads, dim), device="cuda", dtype=dtype)
    scale = dim**-0.5

    ref = None
    results = {}
    for num_splits in splits:
        def call():
            return flash_attn_varlen_func_op(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=seq,
                max_seqlen_k=seq,
                softmax_scale=scale,
                causal=False,
                num_splits=num_splits,
                sm_margin=sm_margin,
                return_softmax_lse=False,
                ver=4,
            )

        try:
            out, times = _time_cuda(call, repeats=repeats, warmup=warmup)
            if ref is None:
                ref = out.detach()
            diff = (out - ref).abs()
            avg = sum(times) / len(times)
            results[str(num_splits)] = {
                "ok": True,
                "ms": times,
                "avg_ms": avg,
                "min_ms": min(times),
                "max_ms": max(times),
                "max_abs_diff_vs_first": float(diff.max().item()),
                "mean_abs_diff_vs_first": float(diff.float().mean().item()),
            }
        except Exception as exc:
            results[str(num_splits)] = {"ok": False, "error": repr(exc)}
        torch.cuda.empty_cache()

    ok = {k: v for k, v in results.items() if v.get("ok")}
    best = min(ok.items(), key=lambda kv: kv[1]["avg_ms"])[0] if ok else None
    baseline = ok.get(str(splits[0]))
    return {
        "name": name,
        "shape": {"batch": batch, "seq": seq, "heads": heads, "dim": dim},
        "sm_margin": sm_margin,
        "baseline_split": splits[0],
        "best_split": int(best) if best is not None else None,
        "best_speedup_vs_baseline": (
            baseline["avg_ms"] / ok[best]["avg_ms"] if best and baseline else None
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="outputs/ltx23-fa4-numsplits-microbench/result.json"
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--splits", default="1,2,4,8,16")
    parser.add_argument("--sm-margins", default="0")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    splits = [int(item) for item in args.splits.split(",") if item.strip()]
    sm_margins = [int(item) for item in args.sm_margins.split(",") if item.strip()]
    shapes = [
        ("stage1_self_b1", 1, 15810, 32, 128),
        ("stage1_self_b3", 3, 15810, 32, 128),
        ("stage1_self_b4", 4, 15810, 32, 128),
        ("stage2_self_b1", 1, 63240, 32, 128),
    ]
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "splits": splits,
        "sm_margins": sm_margins,
        "results": [],
    }
    for sm_margin in sm_margins:
        for name, batch, seq, heads, dim in shapes:
            payload["results"].append(
                _bench_shape(
                    f"{name}_sm{sm_margin}",
                    batch,
                    seq,
                    heads,
                    dim,
                    splits,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    sm_margin=sm_margin,
                )
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
