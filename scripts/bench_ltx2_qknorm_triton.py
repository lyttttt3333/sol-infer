import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.diffusion.triton.rmsnorm_onepass import (
    triton_one_pass_rms_norm,
)


@triton.jit
def _qknorm_pair_inplace_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_n)
    mask = cols < n_cols
    base = row * n_cols + cols

    q = tl.load(q_ptr + base, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + base, mask=mask, other=0.0).to(tl.float32)
    qw = tl.load(q_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    kw = tl.load(k_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    q_var = tl.sum(q * q, axis=0) / n_cols
    k_var = tl.sum(k * k, axis=0) / n_cols
    q_rstd = tl.rsqrt(q_var + eps)
    k_rstd = tl.rsqrt(k_var + eps)

    tl.store(q_ptr + base, q * q_rstd * qw, mask=mask)
    tl.store(k_ptr + base, k * k_rstd * kw, mask=mask)


def triton_qknorm_pair_inplace(q, k, q_weight, k_weight, eps=1e-6, num_warps=8):
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError('q and k must be flattened to [tokens, hidden]')
    if q.shape != k.shape:
        raise ValueError(f'q/k shape mismatch: {q.shape} vs {k.shape}')
    tokens, hidden = q.shape
    block_n = triton.next_power_of_2(hidden)
    _qknorm_pair_inplace_kernel[(tokens,)](
        q,
        k,
        q_weight,
        k_weight,
        hidden,
        eps,
        block_n,
        num_warps=num_warps,
    )
    return q, k


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
    return {
        'ms': times,
        'avg_ms': sum(times) / len(times),
        'min_ms': min(times),
    }


def bench_case(name: str, batch: int, seq: int, hidden: int, repeats: int, warmup: int, warps: list[int]):
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

    def triton_separate_pair():
        return (
            triton_one_pass_rms_norm(q0, q_weight, eps=1e-6),
            triton_one_pass_rms_norm(k0, k_weight, eps=1e-6),
        )

    (q_ref, k_ref), torch_times = time_cuda(torch_pair, repeats, warmup)
    (q_sep, k_sep), sep_times = time_cuda(triton_separate_pair, repeats, warmup)

    result = {
        'shape': {'batch': batch, 'seq': seq, 'hidden': hidden, 'tokens': batch * seq},
        'torch': stats(torch_times),
        'triton_separate': stats(sep_times),
        'triton_separate_speedup': (sum(torch_times) / len(torch_times)) / (sum(sep_times) / len(sep_times)),
        'triton_separate_q_max_abs_diff': float((q_ref.float() - q_sep.float()).abs().max().item()),
        'triton_separate_k_max_abs_diff': float((k_ref.float() - k_sep.float()).abs().max().item()),
        'fused': {},
    }

    for num_warps in warps:
        def fused_with_clone():
            q = q0.reshape(-1, hidden).clone()
            k = k0.reshape(-1, hidden).clone()
            triton_qknorm_pair_inplace(q, k, q_weight, k_weight, eps=1e-6, num_warps=num_warps)
            return q.reshape(batch, seq, hidden), k.reshape(batch, seq, hidden)

        q_work = q0.reshape(-1, hidden).clone()
        k_work = k0.reshape(-1, hidden).clone()

        def fused_inplace_only():
            triton_qknorm_pair_inplace(q_work, k_work, q_weight, k_weight, eps=1e-6, num_warps=num_warps)
            return q_work.reshape(batch, seq, hidden), k_work.reshape(batch, seq, hidden)

        (q_fused, k_fused), fused_clone_times = time_cuda(fused_with_clone, repeats, warmup)
        (_, _), fused_inplace_times = time_cuda(fused_inplace_only, repeats, warmup)
        fused_clone_avg = sum(fused_clone_times) / len(fused_clone_times)
        fused_inplace_avg = sum(fused_inplace_times) / len(fused_inplace_times)
        torch_avg = sum(torch_times) / len(torch_times)
        result['fused'][str(num_warps)] = {
            'with_clone': stats(fused_clone_times),
            'inplace_only': stats(fused_inplace_times),
            'with_clone_speedup': torch_avg / fused_clone_avg,
            'inplace_only_speedup': torch_avg / fused_inplace_avg,
            'q_max_abs_diff': float((q_ref.float() - q_fused.float()).abs().max().item()),
            'q_mean_abs_diff': float((q_ref.float() - q_fused.float()).abs().mean().item()),
            'k_max_abs_diff': float((k_ref.float() - k_fused.float()).abs().max().item()),
            'k_mean_abs_diff': float((k_ref.float() - k_fused.float()).abs().mean().item()),
        }
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/ltx23-qknorm-triton-microbench/result.json')
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--warps', default='4,8,16')
    args = parser.parse_args()
    warps = [int(x) for x in args.warps.split(',') if x]
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
        'triton': triton.__version__,
        'device': torch.cuda.get_device_name(0),
        'warps': warps,
        'results': {},
    }
    for case in cases:
        print(f'Running {case}', flush=True)
        payload['results'][case[0]] = bench_case(*case, repeats=args.repeats, warmup=args.warmup, warps=warps)
        torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
