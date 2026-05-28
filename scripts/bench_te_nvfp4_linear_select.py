import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import fp8_autocast


DTYPE = torch.bfloat16


def _time_cuda(fn, repeats: int, warmup: int) -> dict:
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
    median = float(statistics.median(times))
    return {
        "median_ms": median,
        "avg_ms": float(sum(times) / len(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "samples_ms": times,
    }


def _shape_cases() -> list[tuple[str, int, int, int]]:
    stage1_video_m = 3 * 15810
    stage2_video_m = 63240
    stage1_audio_m = 3 * 251
    stage2_audio_m = 251
    return [
        ("video_attn_4096_stage1", stage1_video_m, 4096, 4096),
        ("video_attn_4096_stage2", stage2_video_m, 4096, 4096),
        ("video_ffn_proj_in_stage1", stage1_video_m, 4096, 16384),
        ("video_ffn_proj_in_stage2", stage2_video_m, 4096, 16384),
        ("video_ffn_proj_out_stage1", stage1_video_m, 16384, 4096),
        ("video_ffn_proj_out_stage2", stage2_video_m, 16384, 4096),
        ("video_gate_logits_stage1", stage1_video_m, 4096, 32),
        ("video_gate_logits_stage2", stage2_video_m, 4096, 32),
        ("audio_attn_2048_stage1", stage1_audio_m, 2048, 2048),
        ("audio_attn_2048_stage2", stage2_audio_m, 2048, 2048),
        ("audio_ffn_proj_in_stage1", stage1_audio_m, 2048, 8192),
        ("audio_ffn_proj_in_stage2", stage2_audio_m, 2048, 8192),
        ("audio_ffn_proj_out_stage1", stage1_audio_m, 8192, 2048),
        ("audio_ffn_proj_out_stage2", stage2_audio_m, 8192, 2048),
        ("a2v_video_query_stage1", stage1_video_m, 4096, 2048),
        ("a2v_video_query_stage2", stage2_video_m, 4096, 2048),
        ("a2v_video_out_stage1", stage1_video_m, 2048, 4096),
        ("a2v_video_out_stage2", stage2_video_m, 2048, 4096),
        ("v2a_audio_query_stage1", stage1_audio_m, 2048, 2048),
        ("v2a_audio_query_stage2", stage2_audio_m, 2048, 2048),
        ("v2a_audio_kv_stage1", stage1_audio_m, 4096, 2048),
        ("v2a_audio_kv_stage2", stage2_audio_m, 4096, 2048),
        ("adaln_video_linear_stage1_like", 3, 4096, 36864),
        ("adaln_audio_linear_stage1_like", 3, 2048, 18432),
    ]


def _make_te_linear(k: int, n: int, weight: torch.Tensor, bias: torch.Tensor) -> te.Linear:
    layer = te.Linear(k, n, bias=True, params_dtype=DTYPE, device="cuda")
    layer.eval()
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer


def _bench_shape(name: str, m: int, k: int, n: int, repeats: int, warmup: int, recipe) -> dict:
    torch.manual_seed(20260528 + m + k + n)
    x = torch.randn((m, k), device="cuda", dtype=DTYPE)
    weight = torch.randn((n, k), device="cuda", dtype=DTYPE) / (k ** 0.5)
    bias = torch.randn((n,), device="cuda", dtype=DTYPE)
    te_layer = _make_te_linear(k, n, weight, bias)

    def torch_bf16():
        return F.linear(x, weight, bias)

    def te_bf16():
        with fp8_autocast(enabled=False):
            return te_layer(x)

    def te_nvfp4():
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            return te_layer(x)

    results = {}
    for provider, fn in (
        ("torch_bf16_linear", torch_bf16),
        ("te_bf16_linear", te_bf16),
        ("te_nvfp4_linear", te_nvfp4),
    ):
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                results[provider] = {"ok": True, **_time_cuda(fn, repeats, warmup)}
        except Exception as exc:
            results[provider] = {"ok": False, "error": repr(exc)}
        finally:
            torch.cuda.synchronize()

    torch_bf16_ms = results.get("torch_bf16_linear", {}).get("median_ms")
    te_bf16_ms = results.get("te_bf16_linear", {}).get("median_ms")
    te_nvfp4_ms = results.get("te_nvfp4_linear", {}).get("median_ms")
    return {
        "shape": {"name": name, "m": m, "k": k, "n": n},
        "torch_bf16_median_ms": torch_bf16_ms,
        "te_bf16_median_ms": te_bf16_ms,
        "te_nvfp4_median_ms": te_nvfp4_ms,
        "speedup_te_nvfp4_vs_torch_bf16": (
            torch_bf16_ms / te_nvfp4_ms if torch_bf16_ms and te_nvfp4_ms else None
        ),
        "speedup_te_nvfp4_vs_te_bf16": (
            te_bf16_ms / te_nvfp4_ms if te_bf16_ms and te_nvfp4_ms else None
        ),
        "providers": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-te-nvfp4-linear-select/result.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--disable-rht", action="store_true")
    parser.add_argument("--disable-stochastic-rounding", action="store_true")
    parser.add_argument("--disable-2d-quantization", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    recipe = NVFP4BlockScaling(
        disable_rht=args.disable_rht,
        disable_stochastic_rounding=args.disable_stochastic_rounding,
        disable_2d_quantization=args.disable_2d_quantization,
    )
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "transformer_engine": getattr(te, "__version__", None),
        "recipe": {
            "name": recipe.__class__.__name__,
            "disable_rht": args.disable_rht,
            "disable_stochastic_rounding": args.disable_stochastic_rounding,
            "disable_2d_quantization": args.disable_2d_quantization,
        },
        "results": {},
    }
    for name, m, k, n in _shape_cases():
        print(f"benchmarking {name}: m={m} k={k} n={n}", flush=True)
        payload["results"][name] = _bench_shape(name, m, k, n, args.repeats, args.warmup, recipe)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
