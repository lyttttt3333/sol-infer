import argparse
import json
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    flash_attn_varlen_func_op,
)


def fa4(q, k, v, seq, scale):
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
        num_splits=1,
        sm_margin=0,
        return_softmax_lse=False,
        ver=4,
    )


def time_cuda(fn, repeats, warmup):
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


def bench(name, batch, seq, heads, dim, repeats, warmup):
    print(f"Running {name} batch={batch} seq={seq}", flush=True)
    torch.manual_seed(123456 + batch + seq)
    dtype = torch.bfloat16
    q = torch.randn((batch, seq, heads, dim), device="cuda", dtype=dtype)
    k = torch.randn((batch, seq, heads, dim), device="cuda", dtype=dtype)
    v = torch.randn((batch, seq, heads, dim), device="cuda", dtype=dtype)
    scale = dim ** -0.5

    def batched():
        return fa4(q, k, v, seq, scale)

    def split_cat():
        return torch.cat([fa4(q[i:i+1], k[i:i+1], v[i:i+1], seq, scale) for i in range(batch)], dim=0)

    def split_prealloc():
        out = torch.empty_like(q)
        for i in range(batch):
            out[i:i+1].copy_(fa4(q[i:i+1], k[i:i+1], v[i:i+1], seq, scale))
        return out

    ref, batched_times = time_cuda(batched, repeats, warmup)
    cat, cat_times = time_cuda(split_cat, repeats, warmup)
    pre, pre_times = time_cuda(split_prealloc, repeats, warmup)
    cat_diff = (ref.float() - cat.float()).abs()
    pre_diff = (ref.float() - pre.float()).abs()
    ba = sum(batched_times) / len(batched_times)
    ca = sum(cat_times) / len(cat_times)
    pa = sum(pre_times) / len(pre_times)
    return {
        "shape": {"batch": batch, "seq": seq, "heads": heads, "dim": dim},
        "batched_ms": batched_times,
        "split_cat_ms": cat_times,
        "split_prealloc_ms": pre_times,
        "batched_avg_ms": ba,
        "split_cat_avg_ms": ca,
        "split_prealloc_avg_ms": pa,
        "split_cat_speedup": ba / ca,
        "split_prealloc_speedup": ba / pa,
        "split_cat_max_abs_diff": float(cat_diff.max()),
        "split_cat_mean_abs_diff": float(cat_diff.mean()),
        "split_cat_bit_exact": bool(torch.equal(ref, cat)),
        "split_prealloc_max_abs_diff": float(pre_diff.max()),
        "split_prealloc_mean_abs_diff": float(pre_diff.mean()),
        "split_prealloc_bit_exact": bool(torch.equal(ref, pre)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-attention-split-batch-microbench/result.json")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    results = {
        "stage1_self_b3": bench("stage1_self_b3", 3, 15810, 32, 128, args.repeats, args.warmup),
        "stage1_self_b4": bench("stage1_self_b4", 4, 15810, 32, 128, args.repeats, args.warmup),
    }
    payload = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
