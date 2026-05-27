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


def bench_case(name, batch, seq, hidden, heads, repeats, warmup):
    torch.manual_seed(1000 + batch + seq + hidden)
    x = torch.randn((batch, seq, hidden), device="cuda", dtype=torch.bfloat16)
    wq = torch.randn((hidden, hidden), device="cuda", dtype=torch.bfloat16) / hidden**0.5
    wk = torch.randn((hidden, hidden), device="cuda", dtype=torch.bfloat16) / hidden**0.5
    wv = torch.randn((hidden, hidden), device="cuda", dtype=torch.bfloat16) / hidden**0.5
    bq = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    bk = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    bv = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    wcat = torch.cat([wq, wk, wv], dim=0).contiguous()
    bcat = torch.cat([bq, bk, bv], dim=0).contiguous()

    wg = torch.randn((heads, hidden), device="cuda", dtype=torch.bfloat16) / hidden**0.5
    bg = torch.randn((heads,), device="cuda", dtype=torch.bfloat16)
    wcat_g = torch.cat([wq, wk, wv, wg], dim=0).contiguous()
    bcat_g = torch.cat([bq, bk, bv, bg], dim=0).contiguous()

    def separate_qkv():
        return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))

    def concat_qkv():
        y = F.linear(x, wcat, bcat)
        return y.split(hidden, dim=-1)

    def separate_qkvg():
        return (
            F.linear(x, wq, bq),
            F.linear(x, wk, bk),
            F.linear(x, wv, bv),
            F.linear(x, wg, bg),
        )

    def concat_qkvg():
        y = F.linear(x, wcat_g, bcat_g)
        q, k, v, g = y.split([hidden, hidden, hidden, heads], dim=-1)
        return q, k, v, g

    sep_out, sep_times = time_cuda(separate_qkv, repeats, warmup)
    cat_out, cat_times = time_cuda(concat_qkv, repeats, warmup)
    sepg_out, sepg_times = time_cuda(separate_qkvg, repeats, warmup)
    catg_out, catg_times = time_cuda(concat_qkvg, repeats, warmup)
    sep_avg = sum(sep_times) / len(sep_times)
    cat_avg = sum(cat_times) / len(cat_times)
    sepg_avg = sum(sepg_times) / len(sepg_times)
    catg_avg = sum(catg_times) / len(catg_times)
    result = {
        "shape": {"batch": batch, "seq": seq, "hidden": hidden, "heads": heads},
        "separate_qkv": stats(sep_times),
        "concat_qkv": stats(cat_times),
        "qkv_speedup": sep_avg / cat_avg,
        **compare_tuple(sep_out, cat_out),
        "separate_qkvg": stats(sepg_times),
        "concat_qkvg": stats(catg_times),
        "qkvg_speedup": sepg_avg / catg_avg,
        **{f"gate_{k}": v for k, v in compare_tuple(sepg_out, catg_out).items()},
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-qkv-concat-microbench/result.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--which", choices=["video", "audio", "all"], default="video")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cases = []
    if args.which in ("video", "all"):
        cases.extend([
            ("stage1_video", 3, 15810, 4096, 32),
            ("stage2_video", 1, 63240, 4096, 32),
        ])
    if args.which in ("audio", "all"):
        cases.extend([
            ("stage1_audio", 3, 251, 2048, 32),
            ("stage2_audio", 1, 251, 2048, 32),
        ])
    payload = {"device": torch.cuda.get_device_name(0), "results": {}}
    for case in cases:
        print("running", case[0], flush=True)
        payload["results"][case[0]] = bench_case(*case, repeats=args.repeats, warmup=args.warmup)
        torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
