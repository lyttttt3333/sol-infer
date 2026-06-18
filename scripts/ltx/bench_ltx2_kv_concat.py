import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def time_cuda(fn, repeats: int, warmup: int):
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


def compare_tuple(a, b):
    out = {}
    for i, (x, y) in enumerate(zip(a, b)):
        d = (x.float() - y.float()).abs()
        out[f"out{i}_max_abs_diff"] = float(d.max().item())
        out[f"out{i}_mean_abs_diff"] = float(d.mean().item())
    return out


def bench_case(name, batch, seq, in_dim, out_dim, repeats, warmup):
    print("running", name, flush=True)
    torch.manual_seed(2000 + batch + seq + in_dim + out_dim)
    x = torch.randn((batch, seq, in_dim), device="cuda", dtype=torch.bfloat16)
    wk = torch.randn((out_dim, in_dim), device="cuda", dtype=torch.bfloat16) / in_dim**0.5
    wv = torch.randn((out_dim, in_dim), device="cuda", dtype=torch.bfloat16) / in_dim**0.5
    bk = torch.randn((out_dim,), device="cuda", dtype=torch.bfloat16)
    bv = torch.randn((out_dim,), device="cuda", dtype=torch.bfloat16)
    wcat = torch.cat([wk, wv], dim=0).contiguous()
    bcat = torch.cat([bk, bv], dim=0).contiguous()

    def separate_kv():
        return F.linear(x, wk, bk), F.linear(x, wv, bv)

    def concat_kv():
        y = F.linear(x, wcat, bcat)
        return y.split(out_dim, dim=-1)

    sep_out, sep_times = time_cuda(separate_kv, repeats, warmup)
    cat_out, cat_times = time_cuda(concat_kv, repeats, warmup)
    sep_avg = sum(sep_times) / len(sep_times)
    cat_avg = sum(cat_times) / len(cat_times)
    return {
        "shape": {"batch": batch, "seq": seq, "in_dim": in_dim, "out_dim": out_dim},
        "separate_kv": stats(sep_times),
        "concat_kv": stats(cat_times),
        "speedup": sep_avg / cat_avg,
        **compare_tuple(sep_out, cat_out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-kv-concat-microbench/result.json")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cases = [
        ("v2a_stage1_video_context", 3, 15810, 4096, 2048),
        ("v2a_stage2_video_context", 1, 63240, 4096, 2048),
        ("a2v_stage1_audio_context", 3, 251, 2048, 2048),
        ("a2v_stage2_audio_context", 1, 251, 2048, 2048),
    ]
    payload = {"device": torch.cuda.get_device_name(0), "results": {}}
    for case in cases:
        payload["results"][case[0]] = bench_case(*case, repeats=args.repeats, warmup=args.warmup)
        torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
