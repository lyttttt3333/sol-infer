import argparse
import json
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.triton.ltx2_qknorm import (
    ltx2_qknorm_split_rope_pair,
)
from sglang.jit_kernel.diffusion.triton.ltx2_rotary import (
    apply_ltx2_split_rotary_emb,
)


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
    return {'avg_ms': sum(times) / len(times), 'min_ms': min(times), 'ms': times}


def bench(batch: int, seq: int, hidden: int, heads: int, repeats: int, warmup: int):
    torch.manual_seed(123)
    half = hidden // heads // 2
    q = torch.randn((batch, seq, hidden), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, seq, hidden), device='cuda', dtype=torch.bfloat16)
    qw = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    kw = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    cos = torch.randn((batch, heads, seq, half), device='cuda', dtype=torch.bfloat16)
    sin = torch.randn((batch, heads, seq, half), device='cuda', dtype=torch.bfloat16)
    qn = torch.nn.RMSNorm(hidden, eps=1e-6, device='cuda', dtype=torch.bfloat16)
    kn = torch.nn.RMSNorm(hidden, eps=1e-6, device='cuda', dtype=torch.bfloat16)
    qn.weight.data.copy_(qw)
    kn.weight.data.copy_(kw)

    def base():
        return (
            apply_ltx2_split_rotary_emb(qn(q), cos, sin),
            apply_ltx2_split_rotary_emb(kn(k), cos, sin),
        )

    def fused():
        return ltx2_qknorm_split_rope_pair(q, k, qw, kw, cos, sin, cos, sin, 1e-6)

    (q_base, k_base), base_times = time_cuda(base, repeats, warmup)
    (q_fused, k_fused), fused_times = time_cuda(fused, repeats, warmup)
    return {
        'shape': {'batch': batch, 'seq': seq, 'hidden': hidden, 'heads': heads},
        'base': stats(base_times),
        'fused': stats(fused_times),
        'speedup': (sum(base_times) / len(base_times)) / (sum(fused_times) / len(fused_times)),
        'q_max_abs_diff': float((q_base.float() - q_fused.float()).abs().max().item()),
        'k_max_abs_diff': float((k_base.float() - k_fused.float()).abs().max().item()),
        'q_mean_abs_diff': float((q_base.float() - q_fused.float()).abs().mean().item()),
        'k_mean_abs_diff': float((k_base.float() - k_fused.float()).abs().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/ltx23-qknorm-rope-triton-microbench/result.json')
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    payload = {
        'device': torch.cuda.get_device_name(0),
        'results': {
            'stage1_video': bench(3, 15810, 4096, 32, args.repeats, args.warmup),
            'stage2_video': bench(1, 63240, 4096, 32, args.repeats, args.warmup),
            'stage1_audio': bench(3, 251, 2048, 32, args.repeats, args.warmup),
            'stage2_audio': bench(1, 251, 2048, 32, args.repeats, args.warmup),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
