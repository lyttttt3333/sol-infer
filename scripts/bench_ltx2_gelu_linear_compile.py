import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def gelu_linear(x, w, b):
    return F.linear(F.gelu(x, approximate='tanh'), w, b)


def time_fn(fn, args, repeats=10, warmup=3):
    for _ in range(warmup):
        y = fn(*args)
        torch.cuda.synchronize()
        del y
    times=[]
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        y=fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del y
    return times


def bench(name, m, inner=16384, hidden=4096, repeats=10):
    torch.manual_seed(123)
    x=torch.randn((m, inner), device='cuda', dtype=torch.bfloat16)
    w=torch.randn((hidden, inner), device='cuda', dtype=torch.bfloat16) / (inner ** 0.5)
    b=torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    args=(x,w,b)
    compiled=torch.compile(gelu_linear, mode='max-autotune-no-cudagraphs', dynamic=False, fullgraph=True)
    y0=gelu_linear(*args)
    torch.cuda.synchronize()
    y1=compiled(*args)
    torch.cuda.synchronize()
    diff=(y0.float()-y1.float()).abs()
    eager=time_fn(gelu_linear,args,repeats=repeats)
    comp=time_fn(compiled,args,repeats=repeats)
    return {
        'm': m,
        'inner': inner,
        'hidden': hidden,
        'max_abs_diff': float(diff.max().item()),
        'mean_abs_diff': float(diff.mean().item()),
        'eager_ms': eager,
        'compiled_ms': comp,
        'eager_avg_ms': sum(eager)/len(eager),
        'compiled_avg_ms': sum(comp)/len(comp),
        'speedup': (sum(eager)/len(eager))/(sum(comp)/len(comp)),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='outputs/ltx23-gelu-linear-compile-microbench/result.json')
    p.add_argument('--repeats', type=int, default=10)
    args=p.parse_args()
    torch.cuda.set_device(0)
    results={
        'stage1_b3_m47430': bench('stage1_b3_m47430', 3*15810, repeats=args.repeats),
        'stage2_b1_m63240': bench('stage2_b1_m63240', 63240, repeats=args.repeats),
    }
    out=Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload={
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
        'results': results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
