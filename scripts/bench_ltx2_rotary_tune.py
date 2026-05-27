import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import apply_split_rotary_emb


@triton.jit
def _rotary_kernel(
    out_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    stride_cos_b: tl.constexpr,
    stride_cos_h: tl.constexpr,
    stride_cos_t: tl.constexpr,
    stride_sin_b: tl.constexpr,
    stride_sin_h: tl.constexpr,
    stride_sin_t: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    head_block = tl.program_id(1)
    batch = pid_bt // seq_len
    token = pid_bt - batch * seq_len
    heads = head_block * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    offsets = tl.arange(0, BLOCK_HALF)
    mask = (heads[:, None] < num_heads) & (offsets[None, :] < half_dim)

    x_base = ((batch * seq_len + token) * num_heads + heads[:, None]) * head_dim
    cos_base = batch * stride_cos_b + heads[:, None] * stride_cos_h + token * stride_cos_t
    sin_base = batch * stride_sin_b + heads[:, None] * stride_sin_h + token * stride_sin_t

    x_first = tl.load(x_ptr + x_base + offsets[None, :], mask=mask, other=0.0)
    x_second = tl.load(x_ptr + x_base + half_dim + offsets[None, :], mask=mask, other=0.0)
    cos = tl.load(cos_ptr + cos_base + offsets[None, :], mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_base + offsets[None, :], mask=mask, other=0.0)

    out_first = (x_first * cos).to(tl.bfloat16).to(tl.float32) + (-x_second.to(tl.float32) * sin.to(tl.float32))
    out_second = (x_second * cos).to(tl.bfloat16).to(tl.float32) + (x_first.to(tl.float32) * sin.to(tl.float32))

    tl.store(out_ptr + x_base + offsets[None, :], out_first, mask=mask)
    tl.store(out_ptr + x_base + half_dim + offsets[None, :], out_second, mask=mask)


def tuned_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_heads: int) -> torch.Tensor:
    batch, seq_len, inner_dim = x.shape
    _, num_heads, _, half_dim = cos.shape
    head_dim = half_dim * 2
    out = torch.empty_like(x)
    block_half = triton.next_power_of_2(half_dim)
    grid = (batch * seq_len, triton.cdiv(num_heads, block_heads))
    num_warps = min(8, max(1, block_heads))
    _rotary_kernel[grid](
        out,
        x,
        cos,
        sin,
        seq_len,
        num_heads,
        head_dim,
        half_dim,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        BLOCK_HEADS=block_heads,
        BLOCK_HALF=block_half,
        num_warps=num_warps,
    )
    return out


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


def bench_shape(name: str, batch: int, seq: int, heads: int, head_dim: int, block_heads_values: list[int], repeats: int, warmup: int):
    torch.manual_seed(123)
    half_dim = head_dim // 2
    x = torch.randn((batch, seq, heads * head_dim), device='cuda', dtype=torch.bfloat16)
    cos = torch.randn((batch, heads, seq, half_dim), device='cuda', dtype=torch.bfloat16)
    sin = torch.randn((batch, heads, seq, half_dim), device='cuda', dtype=torch.bfloat16)

    ref, ref_times = time_cuda(lambda: apply_split_rotary_emb(x, (cos, sin)), repeats, warmup)
    result = {
        'shape': {'batch': batch, 'seq': seq, 'heads': heads, 'head_dim': head_dim},
        'default_ms': ref_times,
        'default_avg_ms': sum(ref_times) / len(ref_times),
        'default_min_ms': min(ref_times),
        'variants': {},
    }
    for block_heads in block_heads_values:
        try:
            out, times = time_cuda(lambda bh=block_heads: tuned_rotary(x, cos, sin, bh), repeats, warmup)
            diff = (out.float() - ref.float()).abs()
            avg = sum(times) / len(times)
            result['variants'][str(block_heads)] = {
                'ms': times,
                'avg_ms': avg,
                'min_ms': min(times),
                'speedup_vs_default': result['default_avg_ms'] / avg,
                'max_abs_diff': float(diff.max().item()),
                'mean_abs_diff': float(diff.mean().item()),
            }
        except Exception as exc:
            result['variants'][str(block_heads)] = {'error': repr(exc)}
        torch.cuda.empty_cache()
    del x, cos, sin, ref
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/ltx23-rotary-tune-microbench/result.json')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--block-heads', default='1,2,4,8,16,32')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    block_heads = [int(x) for x in args.block_heads.split(',') if x.strip()]
    payload = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
        'block_heads': block_heads,
        'results': {
            'stage1_video': bench_shape('stage1_video', 3, 15810, 32, 128, block_heads, args.repeats, args.warmup),
            'stage2_video': bench_shape('stage2_video', 1, 63240, 32, 128, block_heads, args.repeats, args.warmup),
            'stage1_audio': bench_shape('stage1_audio', 3, 251, 32, 64, block_heads, args.repeats, args.warmup),
            'stage2_audio': bench_shape('stage2_audio', 1, 251, 32, 64, block_heads, args.repeats, args.warmup),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
