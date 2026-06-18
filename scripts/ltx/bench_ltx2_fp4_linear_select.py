import argparse
import json
import statistics
from pathlib import Path

import flashinfer
import torch
import torch.nn.functional as F

from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant


DTYPE = torch.bfloat16
GLOBAL_SCALE = 512.0
ALPHA_VALUE = 1.0 / (GLOBAL_SCALE * GLOBAL_SCALE)


def _time_cuda(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        y = fn()
        torch.cuda.synchronize()
        del y

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        y = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del y
    return times


def _scale_shape(m: int, k: int) -> tuple[int, int]:
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_k = k // 16
    rounded_scale_k = ((scale_k + 4 - 1) // 4) * 4
    return rounded_m, rounded_scale_k


def _build_fp4_weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp4 = torch.randint(0, 256, (n, k // 2), device='cuda', dtype=torch.uint8)
    weight_sf = torch.ones(_scale_shape(n, k), device='cuda', dtype=torch.float8_e4m3fn)
    return weight_fp4, weight_sf


def _stats(times: list[float], m: int, n: int, k: int) -> dict[str, float | list[float]]:
    median = float(statistics.median(times))
    return {
        'median_ms': median,
        'avg_ms': float(sum(times) / len(times)),
        'min_ms': float(min(times)),
        'max_ms': float(max(times)),
        'tflops': float((2 * m * n * k) / (median / 1e3) / 1e12),
        'samples_ms': times,
    }


def _bench_shape(name: str, m: int, k: int, n: int, repeats: int, warmup: int) -> dict:
    torch.manual_seed(20260527 + m + k + n)
    x = torch.randn((m, k), device='cuda', dtype=DTYPE)
    weight = torch.randn((n, k), device='cuda', dtype=DTYPE) / (k ** 0.5)
    bias = torch.randn((n,), device='cuda', dtype=DTYPE)
    weight_fp4, weight_sf = _build_fp4_weight(n, k)
    alpha = torch.tensor(ALPHA_VALUE, device='cuda', dtype=torch.float32)
    global_scale = torch.tensor(GLOBAL_SCALE, device='cuda', dtype=torch.float32)

    def bf16_linear():
        return F.linear(x, weight, bias)

    def fp4_cudnn_flashinfer_quant():
        x_fp4, x_sf = flashinfer.fp4_quantize(x, global_scale)
        if x_sf.dtype == torch.uint8:
            x_sf = x_sf.view(torch.float8_e4m3fn)
        y = flashinfer.mm_fp4(
            x_fp4,
            weight_fp4.T,
            x_sf,
            weight_sf.T,
            alpha,
            DTYPE,
            backend='cudnn',
        )
        return y + bias

    def fp4_cudnn_sglang_quant():
        x_fp4, x_sf = scaled_fp4_quant(x, global_scale)
        y = flashinfer.mm_fp4(
            x_fp4,
            weight_fp4.T,
            x_sf,
            weight_sf.T,
            alpha,
            DTYPE,
            backend='cudnn',
        )
        return y + bias

    def fp4_sglang_cutlass():
        x_fp4, x_sf = scaled_fp4_quant(x, global_scale)
        y = cutlass_scaled_fp4_mm(x_fp4, weight_fp4, x_sf, weight_sf, alpha, DTYPE)
        return y + bias

    providers = {
        'bf16_torch_linear': bf16_linear,
        'fp4_cudnn_flashinfer_quant': fp4_cudnn_flashinfer_quant,
        'fp4_cudnn_sglang_quant': fp4_cudnn_sglang_quant,
        'fp4_sglang_cutlass': fp4_sglang_cutlass,
    }

    results = {}
    for provider, fn in providers.items():
        torch.cuda.empty_cache()
        try:
            times = _time_cuda(fn, repeats=repeats, warmup=warmup)
            results[provider] = {'ok': True, **_stats(times, m, n, k)}
        except Exception as exc:
            results[provider] = {'ok': False, 'error': repr(exc)}
        finally:
            torch.cuda.synchronize()

    bf16 = results['bf16_torch_linear']['median_ms'] if results['bf16_torch_linear'].get('ok') else None
    fp4_candidates = {
        k: v['median_ms']
        for k, v in results.items()
        if k.startswith('fp4_') and v.get('ok')
    }
    best_provider = min(fp4_candidates, key=fp4_candidates.get) if fp4_candidates else None
    best_fp4 = fp4_candidates[best_provider] if best_provider else None
    speedup = (bf16 / best_fp4) if bf16 and best_fp4 else None
    return {
        'shape': {'name': name, 'm': m, 'k': k, 'n': n},
        'best_fp4_provider': best_provider,
        'bf16_median_ms': bf16,
        'best_fp4_median_ms': best_fp4,
        'speedup_vs_bf16': speedup,
        'positive': bool(speedup is not None and speedup > 1.0),
        'providers': results,
    }


def _shape_cases() -> list[tuple[str, int, int, int]]:
    stage1_video_m = 3 * 15810
    stage2_video_m = 63240
    stage1_audio_m = 3 * 251
    stage2_audio_m = 251
    return [
        ('video_attn_4096_stage1', stage1_video_m, 4096, 4096),
        ('video_attn_4096_stage2', stage2_video_m, 4096, 4096),
        ('video_ffn_proj_in_stage1', stage1_video_m, 4096, 16384),
        ('video_ffn_proj_in_stage2', stage2_video_m, 4096, 16384),
        ('video_ffn_proj_out_stage1', stage1_video_m, 16384, 4096),
        ('video_ffn_proj_out_stage2', stage2_video_m, 16384, 4096),
        ('video_gate_logits_stage1', stage1_video_m, 4096, 32),
        ('video_gate_logits_stage2', stage2_video_m, 4096, 32),
        ('audio_attn_2048_stage1', stage1_audio_m, 2048, 2048),
        ('audio_attn_2048_stage2', stage2_audio_m, 2048, 2048),
        ('audio_ffn_proj_in_stage1', stage1_audio_m, 2048, 8192),
        ('audio_ffn_proj_in_stage2', stage2_audio_m, 2048, 8192),
        ('audio_ffn_proj_out_stage1', stage1_audio_m, 8192, 2048),
        ('audio_ffn_proj_out_stage2', stage2_audio_m, 8192, 2048),
        ('a2v_video_query_stage1', stage1_video_m, 4096, 2048),
        ('a2v_video_query_stage2', stage2_video_m, 4096, 2048),
        ('a2v_video_out_stage1', stage1_video_m, 2048, 4096),
        ('a2v_video_out_stage2', stage2_video_m, 2048, 4096),
        ('v2a_audio_query_stage1', stage1_audio_m, 2048, 2048),
        ('v2a_audio_query_stage2', stage2_audio_m, 2048, 2048),
        ('v2a_audio_kv_stage1', stage1_audio_m, 4096, 2048),
        ('v2a_audio_kv_stage2', stage2_audio_m, 4096, 2048),
        ('adaln_video_linear_stage1_like', 3, 4096, 36864),
        ('adaln_audio_linear_stage1_like', 3, 2048, 18432),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/ltx23-fp4-linear-select/result.json')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    payload = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
        'global_scale': GLOBAL_SCALE,
        'results': {},
    }
    for name, m, k, n in _shape_cases():
        print(f'benchmarking {name}: m={m} k={k} n={n}', flush=True)
        payload['results'][name] = _bench_shape(name, m, k, n, args.repeats, args.warmup)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
