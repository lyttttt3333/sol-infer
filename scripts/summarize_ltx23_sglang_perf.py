#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--variant', required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    perf_path = out_dir / 'perf.json'
    data = json.loads(perf_path.read_text())
    steps = {item['name']: item['duration_ms'] for item in data.get('steps', [])}
    denoise_s = steps.get('LTX2AVDenoisingStage', 0.0) / 1000.0
    refine_s = steps.get('LTX2RefinementStage', 0.0) / 1000.0
    summary = {
        'output_dir': str(out_dir),
        'variant': args.variant,
        'total_s': data.get('total_duration_ms', 0.0) / 1000.0,
        'denoise_s': denoise_s,
        'refine_s': refine_s,
        'dit_s': denoise_s + refine_s,
        'decode_s': steps.get('LTX2AVDecodingStage', 0.0) / 1000.0,
        'piecewise_sparsity': os.environ.get('SGLANG_PIECEWISE_ATTN_SPARSITY'),
        'piecewise_block_size': os.environ.get('SGLANG_PIECEWISE_ATTN_BLOCK_SIZE'),
        'piecewise_only_video_self': os.environ.get('SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF'),
        'piecewise_approx_remainder': os.environ.get('SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER'),
        'piecewise_route_mode': os.environ.get('SGLANG_PIECEWISE_ATTN_ROUTE_MODE'),
        'fp4_quantize_backend': os.environ.get('SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND'),
        'fp4_gemm_backend': os.environ.get('SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND'),
        'fp4_fused_proj_in_bias_gelu': os.environ.get('SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU'),
        'fp4_fused_proj_out_bias_gate': os.environ.get('SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE'),
        'fp4_fused_attn_to_out_bias_gate': os.environ.get('SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE'),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
