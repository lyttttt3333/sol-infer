#!/usr/bin/env python3
from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from sglang.jit_kernel.nvfp4 import scaled_fp4_quant
from sglang.jit_kernel.diffusion.triton.ltx2_gelu import ltx2_bias_gelu_tanh_inplace
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    modelopt_fp4_apply_linear_bias_gelu,
    modelopt_fp4_apply_quantized_linear,
    modelopt_fp4_quantize_activation,
)


def main() -> None:
    torch.manual_seed(2026)
    dtype = torch.bfloat16
    device = "cuda"
    m, k, n = 512, 4096, 16384
    x = torch.randn((1, m, k), device=device, dtype=dtype) * 0.5
    weight_bf16 = torch.randn((n, k), device=device, dtype=dtype) * 0.5
    bias = torch.randn((n,), device=device, dtype=dtype) * 0.1
    one = torch.tensor([1.0], device=device, dtype=torch.float32)
    w_fp4, w_sf = scaled_fp4_quant(weight_bf16, one)
    layer = SimpleNamespace(
        weight=w_fp4,
        weight_scale_interleaved=w_sf,
        alpha=one,
        input_scale_inv=one,
        output_size_per_partition=n,
        weights_padding_cols=0,
    )

    x_fp4, x_sf, input_shape, output_dtype = modelopt_fp4_quantize_activation(x, layer.input_scale_inv)
    baseline = modelopt_fp4_apply_quantized_linear(
        layer, x_fp4, x_sf, input_shape, output_dtype, bias=None
    )
    baseline = ltx2_bias_gelu_tanh_inplace(baseline.contiguous(), bias)
    fused = modelopt_fp4_apply_linear_bias_gelu(layer, x, bias)
    if fused is None:
        raise RuntimeError("modelopt_fp4_apply_linear_bias_gelu returned None")
    torch.cuda.synchronize()
    diff = (baseline.float() - fused.float()).abs()
    print(json.dumps({
        "shape": {"m": m, "n": n, "k": k},
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "fused_shape": list(fused.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
