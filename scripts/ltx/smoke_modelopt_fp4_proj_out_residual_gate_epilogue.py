#!/usr/bin/env python3
from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from sglang.jit_kernel.diffusion.triton.ltx2_gelu import ltx2_bias_residual_gate
from sglang.jit_kernel.nvfp4 import scaled_fp4_quant
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    modelopt_fp4_apply_linear_per_col_residual_gate,
    modelopt_fp4_apply_quantized_linear,
    modelopt_fp4_quantize_activation,
)


def main() -> None:
    torch.manual_seed(2027)
    dtype = torch.bfloat16
    device = "cuda"
    m, k, n = 512, 16384, 4096
    x = torch.randn((1, m, k), device=device, dtype=dtype) * 0.5
    residual = torch.randn((1, m, n), device=device, dtype=dtype) * 0.5
    gate = torch.randn((1, n), device=device, dtype=dtype) * 0.1
    weight_bf16 = torch.randn((n, k), device=device, dtype=dtype) * 0.5
    bias = torch.randn((n,), device=device, dtype=dtype) * 0.1
    alpha = torch.tensor([0.75], device=device, dtype=torch.float32)
    input_scale_inv = torch.tensor([1.0], device=device, dtype=torch.float32)
    w_fp4, w_sf = scaled_fp4_quant(weight_bf16, input_scale_inv)
    layer = SimpleNamespace(
        weight=w_fp4,
        weight_scale_interleaved=w_sf,
        alpha=alpha,
        input_scale_inv=input_scale_inv,
        output_size_per_partition=n,
        weights_padding_cols=0,
    )

    x_fp4, x_sf, input_shape, output_dtype = modelopt_fp4_quantize_activation(
        x, layer.input_scale_inv
    )
    update = modelopt_fp4_apply_quantized_linear(
        layer, x_fp4, x_sf, input_shape, output_dtype, bias=None
    )
    baseline = ltx2_bias_residual_gate(update.contiguous(), residual, gate, bias)
    fused = modelopt_fp4_apply_linear_per_col_residual_gate(
        layer, x, residual, gate, bias
    )
    if fused is None:
        raise RuntimeError("modelopt_fp4_apply_linear_per_col_residual_gate returned None")
    torch.cuda.synchronize()
    diff = (baseline.float() - fused.float()).abs()
    print(json.dumps({
        "shape": {"m": m, "n": n, "k": k},
        "alpha": float(alpha.item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "fused_shape": list(fused.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
