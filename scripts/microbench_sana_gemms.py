#!/usr/bin/env python3
"""Bare-operator low-precision microbench for the SANA-Video DiT GEMMs.

Gate for the "fp8/nvfp4 on the conv-FFN" experiment: time the ISOLATED matmuls
at the real shapes (incl. the per-call activation-quant cost, which is what made
W4A4 net-negative on the small attn projections) in bf16 vs fp8-e4m3 vs nvfp4.
Apply low precision in the model only where the bare op is significantly faster.

Shapes are Linear-style: x[M,K] @ W[N,K]^T -> [M,N]. M = post-patch tokens
(32760 for 480p/81f, unbatched CFG; 65520 = CFG-batched, informs experiment #1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.bfloat16
FP8 = torch.float8_e4m3fn
FP8MAX = float(torch.finfo(FP8).max)  # 448
FP4MAX = 6.0

# (name, M, K, N)
SHAPES = [
    ("conv_inverted FFN expand+GLU", 32760, 2240, 13440),
    ("conv_point    FFN project",    32760, 6720, 2240),
    ("attn to_q/k/v proj",           32760, 2240, 2240),
    ("attn to_out proj",             32760, 2240, 2240),
    ("proj_out (final)",             32760, 2240, 192),
    ("conv_inverted CFG-batched x2", 65520, 2240, 13440),
]


def bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def main():
    torch.manual_seed(0)
    print(f"{'op':32s} {'M':>6s} {'K':>5s} {'N':>6s} | "
          f"{'bf16ms':>7s} {'TFLOP/s':>8s} | {'fp8ms':>7s} {'fp8x':>5s} | {'fp4ms':>7s} {'fp4x':>5s}")
    try:
        from sglang.jit_kernel.nvfp4 import scaled_fp4_quant, cutlass_scaled_fp4_mm
        have_fp4 = True
    except Exception as ex:
        print("nvfp4 import failed:", ex)
        have_fp4 = False

    def gs(x):
        return (FP8MAX * FP4MAX / x.abs().amax().clamp_min(1e-6)).to(torch.float32)

    for name, M, K, N in SHAPES:
        A = torch.randn(M, K, device=DEV, dtype=DT)
        W = torch.randn(N, K, device=DEV, dtype=DT)
        flops = 2.0 * M * K * N

        t_bf = bench(lambda: F.linear(A, W))

        # fp8 e4m3 W8A8 (per-tensor); time includes per-call activation quant
        try:
            w_s = (W.abs().amax() / FP8MAX).to(torch.float32)
            Wq = (W / w_s).to(FP8)
            b = Wq.t()  # [K,N] for _scaled_mm (column-major)
            a_s = (A.abs().amax() / FP8MAX).to(torch.float32)

            def fp8_run():
                aq = (A / a_s).to(FP8)
                return torch._scaled_mm(aq, b, scale_a=a_s, scale_b=w_s, out_dtype=DT)

            fp8_run()
            t_fp8 = bench(fp8_run)
        except Exception as ex:
            t_fp8 = float("nan")
            print(f"  [fp8 err {name}: {ex}]")

        # nvfp4 W4A4; time includes per-call activation fp4 quant
        t_fp4 = float("nan")
        if have_fp4 and K % 16 == 0:
            try:
                w_gs = gs(W)
                w_fp4, w_scale = scaled_fp4_quant(W, w_gs)

                def fp4_run():
                    a_gs = gs(A)
                    a_fp4, a_scale = scaled_fp4_quant(A, a_gs)
                    alpha = (1.0 / (a_gs * w_gs)).to(torch.float32)
                    return cutlass_scaled_fp4_mm(a_fp4, w_fp4, a_scale, w_scale, alpha, DT)

                fp4_run()
                t_fp4 = bench(fp4_run)
            except Exception as ex:
                print(f"  [fp4 err {name}: {ex}]")

        print(f"{name:32s} {M:>6d} {K:>5d} {N:>6d} | "
              f"{t_bf:7.3f} {flops/t_bf/1e9:8.1f} | "
              f"{t_fp8:7.3f} {t_bf/t_fp8:5.2f} | {t_fp4:7.3f} {t_bf/t_fp4:5.2f}")

    print("\nGuide: fp8x/fp4x = speedup vs bf16 (incl. act-quant). >1.3x on the big "
          "conv-FFN GEMMs => worth wiring in; ~1x or <1 => abandon (like the attn).")


if __name__ == "__main__":
    main()
