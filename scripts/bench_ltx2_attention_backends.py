import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    flash_attn_varlen_func_op,
)


def time_cuda(fn, warmup=5, iters=20):
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    last = None
    for _ in range(iters):
        start.record()
        last = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return last, sum(times) / len(times), min(times), max(times)


def run_shape(name, batch, seq, heads, dim, dtype, iters):
    torch.manual_seed(1234)
    scale = dim ** -0.5
    q = torch.randn((batch, seq, heads, dim), device='cuda', dtype=dtype)
    k = torch.randn((batch, seq, heads, dim), device='cuda', dtype=dtype)
    v = torch.randn((batch, seq, heads, dim), device='cuda', dtype=dtype)

    results = []

    def fa4():
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
            return_softmax_lse=False,
            ver=4,
        )

    ref = None
    try:
        out, avg, mn, mx = time_cuda(fa4, iters=iters)
        ref = out.detach()
        results.append({
            'backend': 'fa4',
            'avg_ms': avg,
            'min_ms': mn,
            'max_ms': mx,
            'maxdiff_vs_fa4': 0.0,
        })
    except Exception as exc:
        results.append({'backend': 'fa4', 'error': repr(exc)})

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)

    backends = [
        ('sdpa_default', None),
        ('sdpa_cudnn', [SDPBackend.CUDNN_ATTENTION]),
        ('sdpa_flash', [SDPBackend.FLASH_ATTENTION]),
        ('sdpa_efficient', [SDPBackend.EFFICIENT_ATTENTION]),
    ]
    if seq <= 20000:
        backends.append(('sdpa_math', [SDPBackend.MATH]))

    for backend_name, backend_list in backends:
        def sdpa_call():
            ctx = sdpa_kernel(backend_list) if backend_list is not None else nullcontext()
            with ctx:
                return F.scaled_dot_product_attention(
                    q_sdpa,
                    k_sdpa,
                    v_sdpa,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                ).transpose(1, 2)

        from contextlib import nullcontext
        try:
            out, avg, mn, mx = time_cuda(sdpa_call, iters=iters)
            maxdiff = None
            meandiff = None
            if ref is not None:
                diff = (out - ref).abs()
                maxdiff = float(diff.max().item())
                meandiff = float(diff.float().mean().item())
            results.append({
                'backend': backend_name,
                'avg_ms': avg,
                'min_ms': mn,
                'max_ms': mx,
                'maxdiff_vs_fa4': maxdiff,
                'meandiff_vs_fa4': meandiff,
            })
        except Exception as exc:
            results.append({'backend': backend_name, 'error': repr(exc)})

    del q, k, v, q_sdpa, k_sdpa, v_sdpa
    if ref is not None:
        del ref
    torch.cuda.empty_cache()
    return {
        'name': name,
        'shape': {'batch': batch, 'seq': seq, 'heads': heads, 'dim': dim},
        'dtype': str(dtype),
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--iters', type=int, default=20)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    dtype = torch.bfloat16
    shapes = [
        ('stage1_video_self_b1', 1, 15810, 32, 128),
        ('stage1_video_self_b4', 4, 15810, 32, 128),
        ('stage2_video_self_b1', 1, 63240, 32, 128),
    ]
    out = {
        'pid': os.getpid(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
        'results': [run_shape(*shape, dtype=dtype, iters=args.iters) for shape in shapes],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
