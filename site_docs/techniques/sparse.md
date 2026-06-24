# Sparse attention

Sparse attention targets model-level redundancy in long video token sequences. Many spatiotemporal attention interactions contribute little to the final output, so structured sparse patterns can reduce attention cost.

## In Sol-Engine

LTX-2.3 uses PISA-style sparse video self-attention in selected stage-2 refinement work. It is combined with cache, fusion, NVFP4, and token pruning in the full optimization stack.

Implemented entries:

- `python/sglang/multimodal_gen/runtime/efficiency/transforms/sparse_attention.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/sparse_video_gen_2_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/sparse_linear_attn.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/backends/block_sparse_attn.py`

## Practical notes

- Sparse settings should be validated visually.
- The value of sparse attention depends on sequence length and stage placement.
- Report both denoise time and end-to-end time; decode and offload overhead can hide attention savings.
