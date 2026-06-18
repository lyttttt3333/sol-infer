import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def ffn(x, w1, b1, w2, b2):
    y = F.linear(x, w1, b1)
    y = F.gelu(y, approximate='tanh')
    y = F.linear(y, w2, b2)
    return y


def time_fn(fn, args, repeats):
    torch.cuda.synchronize()
    # warmup actual kernels / compiled graph after initial compile
    for _ in range(2):
        y = fn(*args)
        torch.cuda.synchronize()
        del y
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times=[]
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        y = fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del y
    return times


def bench_shape(name, shape, hidden, inner, repeats):
    print(f'Running {name} shape={shape} hidden={hidden} inner={inner}', flush=True)
    device='cuda'
    torch.manual_seed(123)
    x = torch.randn(shape, device=device, dtype=torch.bfloat16)
    w1 = torch.randn((inner, hidden), device=device, dtype=torch.bfloat16) / (hidden ** 0.5)
    b1 = torch.randn((inner,), device=device, dtype=torch.bfloat16)
    w2 = torch.randn((hidden, inner), device=device, dtype=torch.bfloat16) / (inner ** 0.5)
    b2 = torch.randn((hidden,), device=device, dtype=torch.bfloat16)
    args=(x,w1,b1,w2,b2)
    eager = ffn
    compiled = torch.compile(ffn, mode='max-autotune-no-cudagraphs', dynamic=False, fullgraph=True)
    # Trigger compile separately; compile time intentionally excluded.
    y0 = eager(*args)
    torch.cuda.synchronize()
    y1 = compiled(*args)
    torch.cuda.synchronize()
    max_abs = (y0.float() - y1.float()).abs().max().item()
    mean_abs = (y0.float() - y1.float()).abs().mean().item()
    eager_times = time_fn(eager, args, repeats)
    compiled_times = time_fn(compiled, args, repeats)
    return {
        'shape': list(shape),
        'hidden': hidden,
        'inner': inner,
        'max_abs_diff': max_abs,
        'mean_abs_diff': mean_abs,
        'eager_ms': eager_times,
        'compiled_ms': compiled_times,
        'eager_avg_ms': sum(eager_times)/len(eager_times),
        'compiled_avg_ms': sum(compiled_times)/len(compiled_times),
        'speedup': (sum(eager_times)/len(eager_times))/(sum(compiled_times)/len(compiled_times)),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/ltx23-ffn-compile-microbench/result.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--which', choices=['stage1_video','stage2_video','audio','all'], default='stage1_video')
    args=parser.parse_args()
    todo=[]
    if args.which in ('stage1_video','all'):
        todo.append(('stage1_video_b3_t15810_h4096_i16384', (3,15810,4096), 4096, 16384))
    if args.which in ('stage2_video','all'):
        todo.append(('stage2_video_b1_t63240_h4096_i16384', (1,63240,4096), 4096, 16384))
    if args.which in ('audio','all'):
        todo.append(('stage1_audio_b3_t251_h2048_i8192', (3,251,2048), 2048, 8192))
        todo.append(('stage2_audio_b1_t251_h2048_i8192', (1,251,2048), 2048, 8192))
    results={}
    for item in todo:
        name, shape, hidden, inner = item
        results[name] = bench_shape(name, shape, hidden, inner, args.repeats)
        torch.cuda.empty_cache()
    out=Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
