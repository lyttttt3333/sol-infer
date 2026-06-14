# SPDX-License-Identifier: Apache-2.0
#
# Selective W4A4 NVFP4 for SANA-Video attention GEMMs (env-gated, separable
# toggle). Only the attention q/k/v/out projections are GEMMs worth quantizing;
# the FFN (GLUMBTempConv) is convolutional, not a matmul, so it is left alone.
# Weights are quantized once; activations are quantized dynamically per forward
# (true W4A4). Uses the jit_cutlass nvfp4 kernels with the standard
# global-scale / alpha convention (see jit_kernel tests).
#
# Toggle (OFF == baseline):
#   SGLANG_SANA_NVFP4=1            enable
#   SGLANG_SANA_NVFP4_LAYERS=0-19 which blocks (default all)
#   SGLANG_SANA_NVFP4_MODULES=attn1,attn2  which attentions (default both)

from __future__ import annotations

import os

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_FP4_MAX = 6.0
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return (_FP8_MAX * _FP4_MAX / x.abs().amax().clamp_min(1e-6)).to(torch.float32)


class Fp4Linear(nn.Module):
    """W4A4 NVFP4 drop-in for nn.Linear (weight pre-quantized, activations
    quantized per-forward). in_features must be a multiple of 16."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        from sglang.jit_kernel.nvfp4 import scaled_fp4_quant

        w = lin.weight.data.to(torch.bfloat16)
        self.out_features, self.in_features = w.shape
        assert self.in_features % 16 == 0, f"nvfp4 needs in_features%16==0, got {self.in_features}"
        w_gs = _global_scale(w)
        w_fp4, w_scale = scaled_fp4_quant(w, w_gs)
        # Plain attrs (not params/buffers) so a later model.to(bf16) won't touch
        # the uint8 fp4 / fp8 scale tensors. Created on the weight's device.
        self.w_fp4 = w_fp4
        self.w_scale = w_scale
        self.w_gs = w_gs
        self.bias = lin.bias  # original Parameter (or None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant

        shp = x.shape
        x2 = x.reshape(-1, self.in_features).to(torch.bfloat16)
        a_gs = _global_scale(x2)
        x_fp4, x_scale = scaled_fp4_quant(x2, a_gs)
        alpha = (1.0 / (a_gs * self.w_gs)).to(torch.float32)
        out = cutlass_scaled_fp4_mm(
            x_fp4, self.w_fp4, x_scale, self.w_scale, alpha, torch.bfloat16
        )
        out = out.reshape(*shp[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class Fp4Conv1x1(nn.Module):
    """W4A4 NVFP4 drop-in for a 1x1 nn.Conv2d (== GEMM over channels). Only the
    conv-FFN `conv_inverted` (high N/K) is worth this — microbench: fp4 2.26x bare
    vs bf16, whereas conv_point/attn are net-negative (low N/K, quant overhead)."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        from sglang.jit_kernel.nvfp4 import scaled_fp4_quant

        w = conv.weight.data.to(torch.bfloat16)  # [out, in, 1, 1]
        self.out_ch, self.in_ch = w.shape[0], w.shape[1]
        assert self.in_ch % 16 == 0, f"nvfp4 needs in%16==0, got {self.in_ch}"
        w2 = w.reshape(self.out_ch, self.in_ch)
        self.w_gs = _global_scale(w2)
        self.w_fp4, self.w_scale = scaled_fp4_quant(w2, self.w_gs)
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant

        N, _, H, W = x.shape
        x2 = x.permute(0, 2, 3, 1).reshape(-1, self.in_ch).to(torch.bfloat16)
        a_gs = _global_scale(x2)
        x_fp4, x_scale = scaled_fp4_quant(x2, a_gs)
        alpha = (1.0 / (a_gs * self.w_gs)).to(torch.float32)
        out = cutlass_scaled_fp4_mm(x_fp4, self.w_fp4, x_scale, self.w_scale, alpha, torch.bfloat16)
        out = out.reshape(N, H, W, self.out_ch).permute(0, 3, 1, 2)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype).view(1, -1, 1, 1)
        return out


class Fp8Conv1x1(nn.Module):
    """W8A8 fp8-e4m3 drop-in for a 1x1 nn.Conv2d (microbench: fp8 1.38x on
    conv_inverted; safer than fp4)."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        w = conv.weight.data.to(torch.bfloat16)
        self.out_ch, self.in_ch = w.shape[0], w.shape[1]
        w2 = w.reshape(self.out_ch, self.in_ch)
        self.w_s = (w2.abs().amax() / _FP8_MAX).clamp_min(1e-6).to(torch.float32)
        self.w_fp8 = (w2 / self.w_s).to(torch.float8_e4m3fn)  # [out, in]
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x.shape
        x2 = x.permute(0, 2, 3, 1).reshape(-1, self.in_ch).to(torch.bfloat16)
        a_s = (x2.abs().amax() / _FP8_MAX).clamp_min(1e-6).to(torch.float32)
        xq = (x2 / a_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(xq, self.w_fp8.t(), scale_a=a_s, scale_b=self.w_s, out_dtype=torch.bfloat16)
        out = out.reshape(N, H, W, self.out_ch).permute(0, 3, 1, 2)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype).view(1, -1, 1, 1)
        return out


def maybe_swap_ffn_lowprec(transformer) -> int:
    """Swap each block's conv-FFN `conv_inverted` (1x1, the high-N/K GEMM) to
    fp4/fp8 per SGLANG_SANA_FFN_LP in {fp4, fp8}. OFF == baseline."""
    mode = os.environ.get("SGLANG_SANA_FFN_LP", "").strip().lower()
    if mode not in ("fp4", "fp8"):
        return 0
    cls = Fp4Conv1x1 if mode == "fp4" else Fp8Conv1x1
    n = 0
    for blk in transformer.transformer_blocks:
        ff = getattr(blk, "ff", None)
        ci = getattr(ff, "conv_inverted", None) if ff is not None else None
        if isinstance(ci, nn.Conv2d):
            ff.conv_inverted = cls(ci)
            n += 1
    logger.info(f"[SANA-Video FFN {mode}] swapped {n} conv_inverted 1x1 convs")
    return n


def _parse_layers(spec: str, n: int) -> set[int]:
    if not spec or spec.strip().lower() in ("", "all"):
        return set(range(n))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def maybe_swap_attn_to_fp4(transformer) -> int:
    """Selectively swap attn q/k/v/out Linears to W4A4 Fp4Linear per env. Returns
    the number of Linears swapped (0 == OFF / baseline)."""
    if os.environ.get("SGLANG_SANA_NVFP4", "0") not in ("1", "true", "True"):
        return 0
    blocks = transformer.transformer_blocks
    layers = _parse_layers(os.environ.get("SGLANG_SANA_NVFP4_LAYERS", "all"), len(blocks))
    mods = [m.strip() for m in os.environ.get("SGLANG_SANA_NVFP4_MODULES", "attn1,attn2").split(",") if m.strip()]
    n_swapped = 0
    for i, blk in enumerate(blocks):
        if i not in layers:
            continue
        for mname in mods:
            attn = getattr(blk, mname, None)
            if attn is None:
                continue
            for proj in ("to_q", "to_k", "to_v"):
                lin = getattr(attn, proj, None)
                if isinstance(lin, nn.Linear):
                    setattr(attn, proj, Fp4Linear(lin))
                    n_swapped += 1
            if hasattr(attn, "to_out") and isinstance(attn.to_out[0], nn.Linear):
                attn.to_out[0] = Fp4Linear(attn.to_out[0])
                n_swapped += 1
    logger.info(
        f"[SANA-Video NVFP4] W4A4 swapped {n_swapped} attn Linears "
        f"(layers={sorted(layers)[:3]}..{max(layers) if layers else '-'}, modules={mods})"
    )
    return n_swapped
