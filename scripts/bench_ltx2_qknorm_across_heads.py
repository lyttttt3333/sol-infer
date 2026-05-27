import argparse
import json
from pathlib import Path

import torch

from sglang.jit_kernel.norm import fused_inplace_qknorm_across_heads


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


def bench_case(name: str, batch: int, seq: int, hidden: int, repeats: int, warmup: int):
    torch.manual_seed(123)
    q0 = torch.randn((batch, seq, hidden), device='cuda', dtype=torch.bfloat16)
    k0 = torch.randn((batch, seq, hidden), device='cuda', dtype=torch.bfloat16)
    q_weight = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    k_weight = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    q_norm = torch.nn.RMSNorm(hidden, eps=1e-6, device='cuda', dtype=torch.bfloat16)
    k_norm = torch.nn.RMSNorm(hidden, eps=1e-6, device='cuda', dtype=torch.bfloat16)
    q_norm.weight.data.copy_(q_weight)
    k_norm.weight.data.copy_(k_weight)

    def torch_pair():
        return q_norm(q0), k_norm(k0)

    def jit_pair():
        q = q0.reshape(-1, hidden).clone()
        k = k0.reshape(-1, hidden).clone()
        fused_inplace_qknorm_across_heads(q, k, q_weight, k_weight, eps=1e-6)
        return q.reshape(batch, seq, hidden), k.reshape(batch, seq, hidden)

    q_work = q0.reshape(-1, hidden).clone()
    k_work = k0.reshape(-1, hidden).clone()

    def jit_pair_inplace_only():
        fused_inplace_qknorm_across_heads(q_work, k_work, q_weight, k_weight, eps=1e-6)
        return q_work.reshape(batch, seq, hidden), k_work.reshape(batch, seq, hidden)

    (q_ref, k_ref), torch_times = time_cuda(torch_pair, repeats, warmup)
    (q_jit, k_jit), jit_times = time_cuda(jit_pair, repeats, warmup)
    (_, _), jit_inplace_times = time_cuda(jit_pair_inplace_only, repeats, warmup)
    q_diff = (q_ref.float() - q_jit.float()).abs()
    k_diff = (k_ref.float() - k_jit.float()).abs()
    torch_avg = sum(torch_times) / len(torch_times)
    jit_avg = sum(jit_times) / len(jit_times)
    jit_inplace_avg = sum(jit_inplace_times) / len(jit_inplace_times)
    return {
        'shape': {'batch': batch, 'seq': seq, 'hidden': hidden, 'tokens': batch * seq},
        'torch_ms': torch_times,
        'torch_avg_ms': torch_avg,
        'torch_min_ms': min(torch_times),
        'jit_with_clone_ms': jit_times,
        'jit_with_clone_avg_ms': jit_avg,
        'jit_with_clone_min_ms': min(jit_times),
        'jit_with_clone_speedup': torch_avg / jit_avg,
        'jit_inplace_ms': jit_inplace_times,
        'jit_inplace_avg_ms': jit_inplace_avg,
        'jit_inplace_min_ms': min(jit_inplace_times),
        'jit_inplace_speedup': torch_avg / jit_inplace_avg,
        'q_max_abs_diff': float(q_diff.max().item()),
        'q_mean_abs_diff': float(q_diff.mean().item()),
        'k_max_abs_diff': float(k_diff.max().item()),
        'k_mean_abs_diff': float(k_diff.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/ltx23-qknorm-across-heads-microbench/result.json')
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=4)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cases = [
        ('stage1_video', 3, 15810, 4096),
        ('stage2_video', 1, 63240, 4096),
        ('stage1_audio', 3, 251, 2048),
        ('stage2_audio', 1, 251, 2048),
    ]
    payload = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
        'results': {},
    }
    for case in cases:
        print(f'Running {case}', flush=True)
        payload['results'][case[0]] = bench_case(*case, repeats=args.repeats, warmup=args.warmup)
        torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
