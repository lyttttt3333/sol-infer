import argparse
import json
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

from bench_te_nvfp4_linear_select import _bench_shape, _shape_cases


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ltx23-te-nvfp4-linear-select-padded/result.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--pad-m-to", type=int, default=16)
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
        "pad_m_to": args.pad_m_to,
        "recipe": {
            "name": recipe.__class__.__name__,
            "disable_rht": args.disable_rht,
            "disable_stochastic_rounding": args.disable_stochastic_rounding,
            "disable_2d_quantization": args.disable_2d_quantization,
        },
        "results": {},
    }
    for name, m, k, n in _shape_cases():
        padded_m = _round_up(m, args.pad_m_to)
        print(f"benchmarking {name}: original_m={m} padded_m={padded_m} k={k} n={n}", flush=True)
        item = _bench_shape(name, padded_m, k, n, args.repeats, args.warmup, recipe)
        item["shape"]["original_m"] = m
        item["shape"]["padded_m"] = padded_m
        item["shape"]["pad_rows"] = padded_m - m
        payload["results"][name] = item

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
