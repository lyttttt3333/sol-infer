# Quantization

Quantization targets kernel-level redundancy and bandwidth pressure by lowering precision where the model tolerates it.

## In Sol-Engine

| Pipeline | Quantization path | Notes |
|---|---|---|
| Cosmos3-Super | NVFP4 | First and last denoising steps stay dense |
| LTX-2.3 | NVFP4 video FFN | Used inside the full optimization stack |

NVFP4 requires Blackwell GPUs and TransformerEngine. On older GPUs, the code falls back to BF16 with a warning.

## Why boundary steps stay dense

Early and late denoising steps are more sensitive to numerical error. Sol-Engine keeps those steps dense and applies NVFP4 where the speed/quality tradeoff is better.
