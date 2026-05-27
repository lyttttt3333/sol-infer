import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _time_cuda(fn, repeats: int, warmup: int) -> tuple[torch.Tensor, list[float]]:
    last = None
    for _ in range(warmup):
        last = fn()
    torch.cuda.synchronize()

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        last = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    assert last is not None
    return last, times


def _linear_flat(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = x.shape[:-1]
    y = F.linear(x.reshape(-1, x.shape[-1]), w, b)
    return y.reshape(*shape, w.shape[0])


def _linear_pretrans_addmm(
    x: torch.Tensor, wt: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    shape = x.shape[:-1]
    y = torch.addmm(b, x.reshape(-1, x.shape[-1]), wt)
    return y.reshape(*shape, wt.shape[1])


def _linear_pretrans_matmul_add(
    x: torch.Tensor, wt: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    shape = x.shape[:-1]
    y = x.reshape(-1, x.shape[-1]) @ wt
    y = y + b
    return y.reshape(*shape, wt.shape[1])


def _linear_m_padded(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    m_multiple: int,
) -> torch.Tensor:
    shape = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    m = x2d.shape[0]
    padded_m = math.ceil(m / m_multiple) * m_multiple
    if padded_m == m:
        y = F.linear(x2d, w, b)
    else:
        x_pad = x2d.new_zeros((padded_m, x2d.shape[-1]))
        x_pad[:m] = x2d
        y = F.linear(x_pad, w, b)[:m]
    return y.reshape(*shape, w.shape[0])


def _bench_case(
    name: str,
    shape: tuple[int, int, int],
    out_features: int,
    repeats: int,
    warmup: int,
    multiples: list[int],
) -> dict:
    torch.manual_seed(123)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(
        (out_features, shape[-1]), device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(shape[-1])
    b = torch.randn((out_features,), device="cuda", dtype=torch.bfloat16)
    wt = w.t().contiguous()

    def direct():
        return F.linear(x, w, b)

    def flat():
        return _linear_flat(x, w, b)

    def pretrans_addmm():
        return _linear_pretrans_addmm(x, wt, b)

    def pretrans_matmul_add():
        return _linear_pretrans_matmul_add(x, wt, b)

    direct_out, direct_times = _time_cuda(direct, repeats=repeats, warmup=warmup)
    flat_out, flat_times = _time_cuda(flat, repeats=repeats, warmup=warmup)
    flat_diff = (flat_out.float() - direct_out.float()).abs()
    pretrans_addmm_out, pretrans_addmm_times = _time_cuda(
        pretrans_addmm, repeats=repeats, warmup=warmup
    )
    pretrans_addmm_diff = (pretrans_addmm_out.float() - direct_out.float()).abs()
    pretrans_matmul_add_out, pretrans_matmul_add_times = _time_cuda(
        pretrans_matmul_add, repeats=repeats, warmup=warmup
    )
    pretrans_matmul_add_diff = (
        pretrans_matmul_add_out.float() - direct_out.float()
    ).abs()
    direct_avg = sum(direct_times) / len(direct_times)
    flat_avg = sum(flat_times) / len(flat_times)
    pretrans_addmm_avg = sum(pretrans_addmm_times) / len(pretrans_addmm_times)
    pretrans_matmul_add_avg = sum(pretrans_matmul_add_times) / len(
        pretrans_matmul_add_times
    )

    results: dict[str, object] = {
        "shape": list(shape),
        "m": int(shape[0] * shape[1]),
        "in_features": int(shape[-1]),
        "out_features": int(out_features),
        "direct_ms": direct_times,
        "direct_avg_ms": direct_avg,
        "direct_min_ms": min(direct_times),
        "flat_ms": flat_times,
        "flat_avg_ms": flat_avg,
        "flat_min_ms": min(flat_times),
        "flat_speedup": direct_avg / flat_avg,
        "flat_max_abs_diff": float(flat_diff.max().item()),
        "flat_mean_abs_diff": float(flat_diff.mean().item()),
        "pretrans_addmm_ms": pretrans_addmm_times,
        "pretrans_addmm_avg_ms": pretrans_addmm_avg,
        "pretrans_addmm_min_ms": min(pretrans_addmm_times),
        "pretrans_addmm_speedup": direct_avg / pretrans_addmm_avg,
        "pretrans_addmm_max_abs_diff": float(pretrans_addmm_diff.max().item()),
        "pretrans_addmm_mean_abs_diff": float(pretrans_addmm_diff.mean().item()),
        "pretrans_matmul_add_ms": pretrans_matmul_add_times,
        "pretrans_matmul_add_avg_ms": pretrans_matmul_add_avg,
        "pretrans_matmul_add_min_ms": min(pretrans_matmul_add_times),
        "pretrans_matmul_add_speedup": direct_avg / pretrans_matmul_add_avg,
        "pretrans_matmul_add_max_abs_diff": float(
            pretrans_matmul_add_diff.max().item()
        ),
        "pretrans_matmul_add_mean_abs_diff": float(
            pretrans_matmul_add_diff.mean().item()
        ),
        "padded": {},
    }

    for multiple in multiples:
        def padded():
            return _linear_m_padded(x, w, b, multiple)

        try:
            padded_out, padded_times = _time_cuda(
                padded, repeats=repeats, warmup=warmup
            )
            diff = (padded_out.float() - direct_out.float()).abs()
            avg = sum(padded_times) / len(padded_times)
            results["padded"][str(multiple)] = {
                "padded_m": int(math.ceil(results["m"] / multiple) * multiple),
                "extra_rows": int(
                    math.ceil(results["m"] / multiple) * multiple - results["m"]
                ),
                "ms": padded_times,
                "avg_ms": avg,
                "min_ms": min(padded_times),
                "speedup": direct_avg / avg,
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
            }
        except Exception as exc:
            results["padded"][str(multiple)] = {"error": repr(exc)}
        torch.cuda.empty_cache()

    del (
        x,
        w,
        b,
        wt,
        direct_out,
        flat_out,
        pretrans_addmm_out,
        pretrans_matmul_add_out,
    )
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="outputs/ltx23-linear-m-padding-microbench/result.json"
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--multiples", default="16,32,64,128")
    parser.add_argument(
        "--which",
        choices=["video", "audio", "all"],
        default="video",
    )
    args = parser.parse_args()

    torch.cuda.set_device(0)
    multiples = [int(item) for item in args.multiples.split(",") if item.strip()]
    cases: list[tuple[str, tuple[int, int, int], int]] = []
    if args.which in ("video", "all"):
        cases.extend(
            [
                ("stage1_attn_proj", (3, 15810, 4096), 4096),
                ("stage1_attn_gate", (3, 15810, 4096), 32),
                ("stage1_ffn_proj_in", (3, 15810, 4096), 16384),
                ("stage1_ffn_proj_out", (3, 15810, 16384), 4096),
                ("stage2_attn_proj", (1, 63240, 4096), 4096),
                ("stage2_attn_gate", (1, 63240, 4096), 32),
                ("stage2_ffn_proj_in", (1, 63240, 4096), 16384),
                ("stage2_ffn_proj_out", (1, 63240, 16384), 4096),
            ]
        )
    if args.which in ("audio", "all"):
        cases.extend(
            [
                ("stage1_audio_attn_proj", (3, 251, 2048), 2048),
                ("stage1_audio_attn_gate", (3, 251, 2048), 32),
                ("stage1_audio_ffn_proj_in", (3, 251, 2048), 8192),
                ("stage1_audio_ffn_proj_out", (3, 251, 8192), 2048),
                ("stage2_audio_attn_proj", (1, 251, 2048), 2048),
                ("stage2_audio_attn_gate", (1, 251, 2048), 32),
                ("stage2_audio_ffn_proj_in", (1, 251, 2048), 8192),
                ("stage2_audio_ffn_proj_out", (1, 251, 8192), 2048),
            ]
        )

    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "multiples": multiples,
        "results": {},
    }
    for name, shape, out_features in cases:
        print(f"Running {name} shape={shape} out={out_features}", flush=True)
        payload["results"][name] = _bench_case(
            name,
            shape,
            out_features,
            repeats=args.repeats,
            warmup=args.warmup,
            multiples=multiples,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
