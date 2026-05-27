import argparse
import json
import os
from pathlib import Path

import torch


def gelu_tanh_inplace(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.gelu_(x, approximate="tanh")


def time_fn(fn, x, repeats: int):
    torch.cuda.synchronize()
    # Warmup
    for _ in range(3):
        y = fn(x.clone())
        del y
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        inp = x.clone()
        torch.cuda.synchronize()
        start.record()
        y = fn(inp)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del y, inp
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/ltx23-ffn-gelu-microbench/result.json')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda')
    shapes = {
        'stage1_video_b3_t15810_d16384': (3, 15810, 16384),
        'stage2_video_b1_t63240_d16384': (1, 63240, 16384),
        'stage1_audio_b3_t251_d8192': (3, 251, 8192),
        'stage2_audio_b1_t251_d8192': (1, 251, 8192),
    }
    results = {}
    for name, shape in shapes.items():
        print(f'Running {name} {shape}', flush=True)
        x = torch.randn(shape, device=device, dtype=torch.bfloat16)
        ref = torch.nn.functional.gelu(x, approximate='tanh')
        cand_in = x.clone()
        cand = gelu_tanh_inplace(cand_in)
        torch.cuda.synchronize()
        max_abs = (ref.float() - cand.float()).abs().max().item()
        mean_abs = (ref.float() - cand.float()).abs().mean().item()
        torch_times = time_fn(lambda t: torch.nn.functional.gelu(t, approximate='tanh'), x, args.repeats)
        triton_times = time_fn(gelu_tanh_inplace, x, args.repeats)
        results[name] = {
            'shape': list(shape),
            'max_abs_diff': max_abs,
            'mean_abs_diff': mean_abs,
            'torch_ms': torch_times,
            'triton_inplace_ms': triton_times,
            'torch_avg_ms': sum(torch_times) / len(torch_times),
            'triton_inplace_avg_ms': sum(triton_times) / len(triton_times),
            'speedup': (sum(torch_times) / len(torch_times)) / (sum(triton_times) / len(triton_times)),
        }
        del x, ref, cand, cand_in
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
