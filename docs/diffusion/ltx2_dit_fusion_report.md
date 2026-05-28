# LTX2 DiT 融合优化报告

本文记录当前 LTX-2.3 1080p/10s 单卡最佳实现的 DiT 内部结构、保留的优化方法、每个融合点替代了什么原始子图，以及为什么这些优化不改变算法语义。

## 结论

当前最佳复现脚本：

```bash
scripts/run_ltx23_best_1080p_single_gpu.sh
```

当前最佳 profile：

```text
outputs/ltx23-dev-1080p10s-speed-resident-prefix-qknorm-rope-dualmod-adavalues-all9-residual-ffn-gateout-audioqkvg-tiledvae-decode-profile/perf.json
```

性能结果：

| 口径 | Baseline | 当前最佳 | 加速比 |
| --- | ---: | ---: | ---: |
| Diffusers no-compile，到 video VAE decode 完成 | `119.811s` | `54.915s` | `2.182x` |
| Diffusers compile，到 video VAE decode 完成 | `88.052s` | `54.915s` | `1.603x` |
| Diffusers no-compile，对 full measured request | `119.811s` | `59.332s` | `2.019x` |
| Diffusers compile，对 full measured request | `88.052s` | `59.332s` | `1.484x` |

当前保留的所有优化都是 runtime/kernel/compile 层面的等价变换。没有修改：

- 分辨率：仍是 `1088x1920`。
- 帧数和 FPS：仍是 `241` frames, `24` fps。
- Stage 1 steps：仍是 `30`。
- Stage 2：仍是官方 3-sigma refinement：`0.909375`, `0.725`, `0.421875`。
- CFG：仍是 `guidance_scale=3.0`。
- LoRA：仍是官方 LTX-2.3 distilled LoRA，在 stage 2 合并使用。
- Scheduler、采样语义、attention 语义、量化策略、剪枝策略。

## 无损性检查结论

结论：当前最终版是算法层面无损，不是 bitwise 完全一致。

这里的“无损”按本项目讨论中的口径理解：不改变采样算法、调度器、step 数、CFG/STG 语义、LoRA 权重使用、attention 语义、分辨率或帧数；允许 kernel 内部因为 bf16 cast、fused multiply-add、GEMM tiling、Inductor/CuTeDSL/Triton 执行顺序不同而产生最后几位的数值差异。

已检查到的证据：

| 优化项 | 算法层面是否无损 | bitwise 是否完全一致 | 证据 |
| --- | --- | --- | --- |
| 官方参数保持 | 是 | 不适用 | 最佳脚本仍使用 `1088x1920`, `241` frames, `30` steps, `guidance_scale=3.0`, 官方 stage-2 sigmas 和官方 distilled LoRA。 |
| AdaLN fusion | 是 | 对 actual fused combo 记录为 exact；一般 bf16 reference 会有舍入差异 | `outputs/ltx23-adaln-from-ada-microbench/result_actual_residual_combo.json` 中 actual/combo 输出多项 `max_abs_diff=0.0`。 |
| Q/K RMSNorm + RoPE fusion | 是 | 否 | `outputs/ltx23-qknorm-rope-triton-microbench/result.json`：stage2 video `q_max_abs_diff=0.125`, `q_mean_abs_diff=1.85e-08`；这是 bf16/kernel 顺序差异，不是公式变化。 |
| Dual modulation fusion | 是 | 否 | `outputs/ltx23-dual-modulate-microbench-bf16round/result.json`：video/audio mean diff 约 `1e-8` 到 `1e-7`，max diff 最大 `0.0625`。 |
| Ada values all9 fusion | 是 | 是 | `outputs/ltx23-ada-values-all9-microbench/result.json`：所有检查到的 `max_abs_diff=0.0`、`mean_abs_diff=0.0`。 |
| Residual gate fusion | 是 | 否 | `torch.addcmul(residual, update, gate)` 等价于 `residual + update * gate`；microbench 有 bf16 fused op 差异，最大到 `0.125`。 |
| FFN `proj_in + GELU` fusion | 是 | 是 | `outputs/ltx23-ffn-gelu-microbench/result-v2.json`：stage1/stage2 video/audio 全部 `max_abs_diff=0.0`。 |
| Attention `gate_to_out` compile | 是 | 否 | 编译的是同一子图：sigmoid、scale、reshape、linear；`outputs/ltx23-gate-to-out-compile-microbench/result.json` 最大 diff `0.0625`，mean diff 约 `0.0024`。 |
| Audio Q/K/V/Gate projection concat | 是 | 理论上不保证 bitwise | 同一组 Q/K/V/gate weight 和 bias concat 后一次 `F.linear` 再 split；数学上等价于四个独立 linear。 |
| Block-0 self-attention sharing | 是 | 应为等价输出复用 | 代码只在 block0、无 mask/perturbation、无 skip、无 SP replicated 且 branch 输入等价时共享，之后 expand 回完整 batch。 |
| Guidance prefix sharing | 是 | 应为等价输出复用 | `_ltx2_guidance_prefix_share_plan` 只在 perturbed branch 第一次 divergence 之前裁掉冗余 branch，并在 divergence block 前恢复完整 batch。 |
| Tiled VAE decoder compile | 是 | 不保证 bitwise | 编译同一个 tiled decoder graph；只改变 Inductor kernel/调度方式。 |

因此，如果“无损”指算法层面和生成配置层面：是，无损。
如果“无损”指逐元素 bitwise 完全一致：否，部分 fused/compiled kernel 会有 bf16 和 kernel 执行顺序带来的数值差异。

## 当前 DiT 内部结构

代码入口是：

```text
python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py
```

主模型类是 `LTX2VideoTransformer3DModel`。一次 DiT forward 的主流程是：

1. 准备 video/audio/cross-attention RoPE 坐标和 cos/sin embedding。
2. 对 video latent 和 audio latent 做 patchify projection。
3. 根据 timestep 构造 video/audio 的 AdaLN modulation tensor。
4. 构造 prompt cross-attention 和 audio-video cross-attention 的 modulation tensor。
5. 顺序执行 `LTX2TransformerBlock` 堆叠。
6. 执行 final LayerNorm + output scale/shift。
7. 通过 output projection 回到 latent patch channel。
8. unpatchify 回 video/audio latent layout。

目标 1080p/10s run 的关键固定 shape：

- Stage 1 video tokens：`31 * 17 * 30 = 15810`。
- Stage 2 video tokens：`31 * 34 * 60 = 63240`。
- Audio tokens：`251`。
- Video hidden size：`4096`。
- Audio hidden size：`2048`。
- Video self-attention：`32` heads, `128` dim/head。
- Audio self-attention：`32` heads, `64` dim/head。

每个 `LTX2TransformerBlock` 是 video/audio 双流结构，内部包含：

1. `attn1`：video self-attention。
2. `audio_attn1`：audio self-attention。
3. `attn2`：video prompt cross-attention。
4. `audio_attn2`：audio prompt cross-attention。
5. `audio_to_video_attn`：audio-to-video cross-attention。
6. `video_to_audio_attn`：video-to-audio cross-attention。
7. `ff`：video feed-forward network。
8. `audio_ff`：audio feed-forward network。
9. 多组 scale/shift/gate modulation table，覆盖 MSA、MLP、prompt cross-attention、A2V/V2A cross-attention。

一个 block 的主要计算顺序可以概括为：

```text
video/audio MSA AdaLN
video/audio self-attention
prompt cross-attention AdaLN
video/audio prompt cross-attention
A2V/V2A cross-attention modulation
A2V cross-attention
V2A cross-attention
video/audio MLP AdaLN
video/audio FFN
residual gate updates
```

因此 DiT 内部耗时主要来自：

- attention projection：Q/K/V/gate/to_out。
- attention core。
- FFN `proj_in`/`proj_out`。
- 大量围绕 attention/FFN 的 RMSNorm、scale/shift、residual gate、Ada value 生成。

当前优化重点不是改 attention core 算法，而是把这些高频小算子融合到更少的 kernel 或已存在的 GEMM epilogue/compiled subgraph 里，减少 launch overhead 和中间 tensor 的读写。

## 最终保留的开关

当前最佳脚本实际开启：

```bash
SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=1
SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=1
SGLANG_LTX2_FUSED_ADALN=1
SGLANG_LTX2_FUSED_QKNORM_ROPE=1
SGLANG_LTX2_FUSED_DUAL_MODULATE=1
SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1
SGLANG_LTX2_FUSED_RESIDUAL_GATE=1
SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1
SGLANG_LTX2_COMPILE_GATE_TO_OUT=1
SGLANG_LTX2_FUSED_AUDIO_QKVG=1
SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=1
SGLANG_LTX2_VAE_COMPILE_MODE=max-autotune-no-cudagraphs
```

下面逐项解释。

## 1. AdaLN 融合

原始 eager 子图有两种常见形式。

纯 modulation：

```text
normed = rms_norm(x)
y = normed * (1 + scale) + shift
```

带 residual gate：

```text
residual_out = residual + x * gate
normed = rms_norm(residual_out)
y = normed * (1 + scale) + shift
```

保留实现：

- 开关：`SGLANG_LTX2_FUSED_ADALN=1`。
- 调用点：`_ltx2_norm_scale_shift`、`_ltx2_residual_norm_scale_shift`。
- Kernel 文件：`python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py`。
- Kernel 入口：`fused_norm_scale_shift`、`fused_scale_residual_norm_scale_shift`。

怎么融合：

- 一个 CTA 处理一个 `[batch, token]` row。
- 从 global memory 读取 `x`、可选 `residual`、可选 `gate`、`scale`、`shift`。
- 在 register 中完成 `residual + x * gate`。
- 在同一个 kernel 内做 RMSNorm。
- 继续在同一个 kernel 内做 `normed * (1 + scale) + shift`。
- 最后只写一次输出；如果调用者需要 residual_out，也在同一个 kernel 内写出。

收益来源：

- 避免单独的 multiply/add/RMSNorm/scale/shift 多个 kernel launch。
- 避免 materialize `residual_out`、`normed` 等中间 tensor 后再读回。
- 这些子图在每个 block 的 MSA、prompt cross-attention、MLP 前后高频出现。

语义：数学操作相同，只改变 kernel 内执行位置和内存访问路径。

## 2. Q/K RMSNorm + RoPE 融合

原始 attention projection 后的 eager 子图：

```text
q = to_q(x)
k = to_k(context)
q = q_norm(q)
k = k_norm(k)
q = apply_split_rotary_emb(q, cos, sin)
k = apply_split_rotary_emb(k, k_cos, k_sin)
```

保留实现：

- 开关：`SGLANG_LTX2_FUSED_QKNORM_ROPE=1`。
- 调用点：`_ltx2_try_fused_qknorm_split_rope`。
- Kernel 文件：`python/sglang/jit_kernel/diffusion/triton/ltx2_qknorm.py`。
- Kernel 入口：`ltx2_qknorm_split_rope_pair`。

怎么融合：

- 输入 Q/K 是 `[B, S, D]`，flatten 成 token row。
- 每个 Triton program 处理一个 row。
- Kernel 同时 load Q 和 K 的前半/后半 channel。
- 计算 Q/K 各自的 RMS variance。
- 应用 q_norm/k_norm weight。
- 直接在同一个 kernel 中使用 cos/sin 做 split RoPE。
- 最终写出已经 norm + RoPE 完成的 Q/K。

收益来源：

- 不再 materialize normalized Q/K。
- 不再分别 launch Q norm、K norm、Q RoPE、K RoPE。
- 对 LTX2 的固定 hidden/head shape 做了专门路径。

触发条件：bf16、contiguous、TP world size 为 1、split RoPE layout 为 `[B, heads, tokens, half_dim]`。

## 3. Dual Modulation 融合

A2V/V2A cross-attention 前，会从同一个 hidden state 产生两份 modulated input。

原始 eager 子图：

```text
normed = rms_norm(hidden)
a2v_input = normed * (1 + a2v_scale) + a2v_shift
v2a_input = normed * (1 + v2a_scale) + v2a_shift
```

保留实现：

- 开关：`SGLANG_LTX2_FUSED_DUAL_MODULATE=1`。
- 调用点：`_ltx2_try_fused_rmsnorm_dual_modulate`、`_ltx2_try_fused_rmsnorm_ca_dual_modulate`。
- Kernel 文件：`python/sglang/jit_kernel/diffusion/triton/ltx2_dual_modulate.py`。
- Kernel 入口：`ltx2_rmsnorm_dual_modulate`、`ltx2_rmsnorm_ca_dual_modulate_from_temb`。

怎么融合：

- 对同一个 row 只做一次 RMSNorm。
- 同一个 kernel 中生成两份输出：A2V 输入和 V2A 输入。
- cross-attention 版本还把 `scale_shift_table + temb` 的计算也合进来。

收益来源：避免对同一 hidden state 重复 RMSNorm，减少两组 scale/shift 的 elementwise kernel，并减少中间 `normed` tensor 的写回和读回。

## 4. Ada Values All9 融合

LTX2.3 block 中经常需要一次性生成 9 个 Ada 参数，例如：

```text
shift_msa, scale_msa, gate_msa
shift_mlp, scale_mlp, gate_mlp
shift_q, scale_q, gate_q
```

原始 eager 子图：

```text
ada = scale_shift_table[indices].unsqueeze(0).unsqueeze(0)
ada = ada + timestep.reshape(B, S, P, D)[:, :, indices, :]
values = ada.unbind(dim=2)
values = [v.squeeze(2) for v in values]
```

保留实现：

- 开关：`SGLANG_LTX2_FUSED_ADA_VALUES_ALL=1`。
- 调用点：`_ltx2_try_fused_ada_values9`。
- Kernel 文件：`python/sglang/jit_kernel/diffusion/triton/ltx2_ada_values.py`。
- Kernel 入口：`ltx2_ada_values9`。

怎么融合：

- Triton kernel 读取 9 行 `scale_shift_table`。
- 同时读取 `timestep` 中对应的 9 组 hidden slice。
- 在一个 launch 中写出 9 个 `[B, S, D]` tensor。

收益来源：去掉 reshape/unbind/squeeze 的小算子链，并避免多次 table add kernel。

## 5. Residual Gate 融合

原始 eager 子图：

```text
out = residual + update * gate
```

保留实现：

- 开关：`SGLANG_LTX2_FUSED_RESIDUAL_GATE=1`。
- 调用点：`_ltx2_residual_gate_add`。
- 实现方式：`torch.addcmul(residual, update, gate)`。

怎么融合：用 PyTorch fused add-multiply primitive 替代单独 multiply 和 add。该操作出现在 attention/FFN 后的 residual update 上。

## 6. FFN `proj_in + GELU` 融合

原始 FFN：

```text
x = proj_in(x)
x = GELU(x, approximate="tanh")
x = proj_out(x)
```

保留实现：

- 开关：`SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=1`。
- 调用点：`_ltx2_try_fused_ffn_proj_in_gelu`。
- 实现方式：`torch.ops.aten._addmm_activation.default(..., use_gelu=True)`。

怎么融合：

- 将 `proj_in` 的 GEMM、bias add、GELU 激活放到 ATen fused addmm activation 路径。
- `proj_out` 仍保留原来的 `RowParallelLinear`。

收益来源：避免 `proj_in` 输出后单独 launch GELU，避免 `proj_in` output 被完整写出后再读入 GELU。

## 7. Attention `gate_to_out` 编译

LTX2 gated attention 的 output 子图：

```text
out = attention(q, k, v)
scale = 2 * sigmoid(gate_logits)
out = out * scale.unsqueeze(-1)
out = out.reshape(B, T, heads * head_dim)
out = to_out(out)
```

保留实现：

- 开关：`SGLANG_LTX2_COMPILE_GATE_TO_OUT=1`。
- 调用点：`_try_compiled_gate_to_out`。
- 编译函数：`_ltx2_gate_to_out_impl`。
- 编译模式：`torch.compile(..., mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=True)`。

怎么融合：

- Inductor 编译 sigmoid、scale、reshape、`F.linear` 这个固定 shape 子图。
- 当前只保留 video self-attention 的正收益 shape：`query_dim = inner_dim = 4096`，`heads = 32`，`dim_head = 128`。

没有保留的相关候选：A2V gate-to-output compile 和 stage-2 audio gate-to-output compile 都测过，但 full run 负收益。

## 8. Audio Q/K/V/Gate Projection 融合

Audio self-attention 原始 projection：

```text
q = to_q(x)
k = to_k(x)
v = to_v(x)
gate_logits = to_gate_logits(x)
```

保留实现：

- 开关：`SGLANG_LTX2_FUSED_AUDIO_QKVG=1`。
- 调用点：`_try_fused_audio_qkvg_projection`。
- 实现方式：把 Q/K/V/gate 的 weight 和 bias concat 成一个大 linear。

怎么融合：

```text
fused_weight = cat([q_weight, k_weight, v_weight, gate_weight], dim=0)
fused_bias = cat([q_bias, k_bias, v_bias, gate_bias], dim=0)
fused = F.linear(x, fused_weight, fused_bias)
q, k, v, gate = split(fused)
```

代码会用参数 tensor 的 data pointer、shape、stride、dtype、device、version 构造 signature；signature 变化时重新构造 fused weight/bias。

为什么只保留 audio：当前正收益条件限定为 audio self-attention shape：`query_dim = context_dim = inner_dim = 2048`。video/global QKV、KV、Q+gate 等 projection fusion 都测过，full run 或 microbench 不如当前 cuBLASLt 路径。

## 9. Block-0 Self-Attention Sharing

CFG/STG 会把多个 guidance branch 合成 batch 运行。某些分支在最开始还没有发生 perturbation，输入完全等价，但 eager 路径仍重复算 self-attention。

保留实现：

- 开关：`SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=1`。
- 调用点：`LTX2TransformerBlock.forward` 中的 `share_video_self_attn` 和 `share_audio_self_attn`。

怎么做：

- 只在 block `0` 生效。
- 检查 batch size、mask、perturbation mask、skip flag、sequence parallel 状态。
- 条件满足时，只对代表性 branch 运行 video/audio self-attention。
- 结果用 `expand(...).contiguous()` 扩回完整 guidance batch。

语义：只共享“输入和 perturbation 行为都相同”的 branch，在分支真正产生差异前共享计算。

## 10. Guidance Prefix Sharing

Block-0 sharing 只覆盖第一个 block 的 self-attention。`SHARE_GUIDANCE_PREFIX` 更进一步：在 STG perturbed branch 第一次真正 diverge 之前，整个 block prefix 都可以少跑一个冗余 branch。

保留实现：

- 开关：`SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=1`。
- 规划函数：`_ltx2_guidance_prefix_share_plan`。
- batch 裁剪函数：`_ltx2_index_batch_dim`。
- RoPE 裁剪函数：`_ltx2_index_rotary_emb`。

怎么做：

- 检查 `perturbation_configs`，找出第一个 self-attention skip/perturb block。
- 在这个 block 之前，把冗余 perturbed branch 从以下 tensor 中裁掉：video/audio hidden states、prompt hidden states、temb、prompt temb、A2V/V2A modulation tensor、RoPE cos/sin、attention masks、perturbation state maps。
- 到第一个 divergence block 时，把 compact batch 扩回完整 guidance batch，并恢复完整输入。

语义：只在 perturbed branch 与 conditional branch 等价的 prefix 中少算；分支真正需要不同结果之前已经恢复完整 batch。

## 11. Tiled Video VAE Decoder Compile

这不属于 DiT block 内部，但属于最终保留的端到端正收益项，因为 strict 口径是“到 video VAE decode 完成”。

保留实现：

- 开关：`SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=1`。
- 编译模式：`SGLANG_LTX2_VAE_COMPILE_MODE=max-autotune-no-cudagraphs`。

怎么做：

- 1080p/10s 的 tiled VAE decode shape 固定。
- 对 tiled decoder 进行 Inductor compile。
- 实测 `max-autotune-no-cudagraphs` 最快，其他 mode 不保留。

结果：当前 video VAE decode 是 `1.496s`；当前完整 decode stage 是 `5.913s`，其中 video postprocess 不计入 strict 口径。

## 12. NVFP4 叠加实验更新

本节记录在上述 fully optimized SGLang pipeline 上继续叠加 selective NVFP4 的结果。这里的 baseline 仍然是同一套 1080p/10s、单卡、two-stage official 配置：`1088x1920`, `241` frames, stage1 `30` steps, stage2 `3` steps, `guidance_scale=3.0`。

当前端到端最好结果：

| 方案 | total | denoise | refine | decode | vs `59.332s` |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 fully optimized baseline | `59.332s` | `44.421s` | `8.705s` | `5.913s` | `1.000x` |
| selective NVFP4 video attn+FFN, previous best | `53.360s` | `41.399s` | `8.605s` | `3.194s` | `1.112x` |
| selective NVFP4 video attn+FFN + FP4 bias in-place add | `52.938s` | `41.025s` | `8.618s` | `3.113s` | `1.121x` |
| + FP4 `proj_in` bias+GELU + FP4 `proj_out` bias/residual/gate | `50.519s` | `38.729s` | `8.418s` | `3.204s` | `1.174x` |

当前最好证据文件：

```text
outputs/ltx23-dev-1080p10s-nvfp4-video-attn-ffn-flashinfer-fp4biasgelu-projoutgate-best-pipeline/perf.json
```

距离目标：`59.332 / 1.2 = 49.443s`，当前 `50.519s` 还差约 `1.076s`。因此 NVFP4 已经带来额外收益，但还没有达到 1.2x 目标。

### 当前保留的 NVFP4 方案

量化 checkpoint 使用 selective NVFP4，只替换 video transformer 中 full run 证明确认为正收益的线性层家族：

```text
transformer_blocks.*.ff.net.0.proj
transformer_blocks.*.ff.net.2
transformer_blocks.*.attn1.to_q / to_k / to_v / to_out.0
transformer_blocks.*.attn2.to_q / to_k / to_v / to_out.0
```

stage 2 使用 LoRA 合并后的对应 selective NVFP4 transformer，因此运行时不再做 stage-2 LoRA switch。当前最快 backend 组合是：

```bash
SGLANG_DIFFUSION_FP4_QUANTIZE_BACKEND=flashinfer
SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn
```

FP4 Linear 的实际执行流程是：

```text
activation bf16
-> FP4 activation quantize
-> pad activation if needed
-> flashinfer.mm_fp4(..., backend="cudnn")
-> slice padded output if needed
-> bias add
```

本轮保留的正收益改动包括 FP4 linear 输出 bias 的 in-place add、FP4 `proj_in` 的 bias+GELU Triton epilogue，以及 FP4 `proj_out` 后的 bias+residual/gate Triton epilogue。bias in-place add 是把最后一步从：

```python
out = out + bias
```

改成：

```python
out.add_(bias)
```

这一步不是 GEMM epilogue fusion，bias 仍然是 GEMM 后的独立 elementwise kernel；收益来自避免为 `out + bias` 再分配一个完整输出 tensor。`proj_in` 的 bias+GELU 和 `proj_out` 的 bias+residual/gate 则是 GEMM 后紧接的 Triton epilogue fusion，减少中间 tensor 读写和小 kernel launch，但不把 GEMM 本体融合进同一个 FP4 GEMM kernel。上述改动不改变 FP4 路径的算法语义。

### 算法误差口径

NVFP4 本身相对 BF16 baseline 是量化方案，因此不是 BF16 算法层面的无损替换。

但本轮新增的 FP4 runtime/fusion 改动不引入额外算法变化：

- FP4 bias in-place add 与 `out + bias` 数学等价。
- FP4 `proj_in` bias+GELU 等价于 `proj_in(x) + bias` 后接 `GELU(approximate="tanh")`。
- FP4 `proj_out` bias+residual/gate 等价于 `residual + (proj_out(x) + bias) * gate`。
- shared activation quant 的 QKV 实验只复用同一 input scale 下的 activation quant；没有被纳入 best。
- packed QKV 因真实 checkpoint 中 Q/K/V `weight_scale_2` 不一致，不能直接用单 scalar alpha 无损合并成一次 GEMM。

因此当前 best 的额外误差来源仍然是 selective NVFP4 量化本身，不是 bias in-place add。

### 被验证但不保留的 NVFP4 候选

| 候选 | 结果 | 结论 |
| --- | ---: | --- |
| FP4 shared-QKV activation quant | `53.822s` | 只省 activation quant，仍是 3 个 GEMM，full run 慢于 best。 |
| packed QKV equal-alpha microbench | stage1 `1.028x` vs shared, stage2 `0.979x` vs shared | 理论收益很小，且真实权重 alpha 不一致。 |
| packed QKV + output rescale microbench | stage1 `0.655x`, stage2 `0.628x` vs shared | 为修正不同 alpha 需要大输出 rescale，负收益。 |
| Triton in-place GELU | microbench `1.33x`，full `52.982s` | 单独开不超过 bias-only；作为 FP4 `proj_in` bias+GELU 子路径后纳入 best。 |
| attention gate scaling in-place | full `54.337s` | 负收益，已撤回。 |
| GELU + gate in-place | full `54.035s` | 旧 in-place 组合负收益。 |
| FP4 attention `to_out` bias/residual/gate epilogue | full `51.424s` | 慢于 `50.519s` best，不保留默认开启。 |
| FlashInfer activation quant `backend="cuda"` | full `53.690s` | 慢于默认 FlashInfer quant。 |
| FlashInfer activation quant `backend="cute-dsl"` | full `58.327s` | 负收益。 |
| repo-local `sglang.jit_kernel.nvfp4.scaled_fp4_quant` | full `53.620s` | 慢于默认 FlashInfer quant。 |
| FlashInfer FP4 GEMM `backend="trtllm"` | full `58.365s` | 负收益。 |

### 下一步需要的真正 1.2x 空间

当前剩余 1.2x 差距约 `1.076s`。从已有 profile 看，attention core 仍有约 `15s+`，不在 FP4 Linear GEMM 范围内；video QKV/FFN FP4 GEMM 周围的小 kernel 优化只能再提供亚秒级收益。要达到 `49.44s`，更可能需要下面之一：

- FP4 GEMM 原生支持 bias/GELU/gate epilogue，而不是 GEMM 后单独 kernel。
- 支持 per-output/per-group alpha 的 packed QKV FP4 GEMM，避免真实 Q/K/V scale 不一致导致的输出 rescale。
- attention core 本身的 fixed-shape kernel 调优或近似/算法级改动；后者会超出当前“只做实现层优化”的无损范围。


## 13. Sparse Attention 叠加更新

本节记录从 `origin/ltx-sparse-attn-bringup` 接入 B200 `piecewise_attn` sparse attention backend 后，在当前 selective NVFP4 + fusion 最好版本上的结果。这里开始属于算法层面有损优化，不再属于前面 fusion-only 的“算法无损”口径。

### 接入方式

新增 backend 文件：

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py
```

注册点：

```text
python/sglang/multimodal_gen/runtime/platforms/interface.py
python/sglang/multimodal_gen/runtime/platforms/cuda.py
python/sglang/multimodal_gen/configs/models/dits/base.py
```

复现脚本：

```bash
scripts/slurm_ltx23_best_nvfp4_piecewise_1080p10s.sh
```

脚本仍然使用同一套 1080p/10s 官方生成配置：`1088x1920`, `241` frames, stage1 `30` steps, stage2 `3` steps, `guidance_scale=3.0`，并加载当前 selective NVFP4 stage1/stage2 transformer：

```text
outputs/ltx23-selective-nvfp4-video-attn-ffn-transformer-mat
outputs/ltx23-selective-nvfp4-video-attn-ffn-stage2-lora-transformer-mat
```

### Sparse Kernel 行为

从 sparse branch 接入的 `piecewise_attn` 原始路径做三步：

1. 对 Q/K/V 按 block 做 chunk reduce，得到 `qc/kc/vc` 和 K variance proxy。
2. 用 `qc @ kc` 加上 variance proxy 做 top-k block routing。
3. 对 top-k block 做 exact attention；对未选中的 block，用 `kc/vc` centroid 做 piecewise 近似项。

当前实现又保留了几个固定 shape 下的可控开关：

```bash
SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER=false
SGLANG_PIECEWISE_ATTN_ROUTE_MODE=score|local
SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE=0|1
```

- `approx_remainder=true`：保留 sparse branch 原始 piecewise 近似，即未选中 block 仍用 centroid 参与 softmax。
- `approx_remainder=false`：更激进的 exact-selected-only sparse attention，只保留选中的 exact block，完全丢弃未选中 block。
- `route_mode=score`：继续使用 top-k score routing。
- `route_mode=local`：直接选同位置 local block，不再计算 score/top-k routing；这是当前速度最快的路径，但它比 score routing 更偏离原 sparse branch 语义。
- `SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE=1`：打开 selected FP4 attention `to_out + bias + gate` 融合。这个开关在 dense attention 上是负收益，但叠加 local sparse `b=32` 后变成正收益。

当前脚本默认保留的最快实测配置是：

```bash
SGLANG_PIECEWISE_ATTN_SPARSITY=0.999
SGLANG_PIECEWISE_ATTN_BLOCK_SIZE=32
SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER=false
SGLANG_PIECEWISE_ATTN_ROUTE_MODE=local
SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF=true
SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE=1
```

它只作用在 video self-attention (`attn1`) 且 Q/K/V 序列长度相同的路径；cross-attention 和 audio self-attention 回落到 dense FlashAttention/SDPA fallback。

### 性能结果

| 方案 | total | denoise | refine | decode | vs `50.519s` | vs `59.332s` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVFP4 + fusion best, dense attention | `50.519s` | `38.729s` | `8.418s` | `3.204s` | `1.000x` | `1.174x` |
| piecewise original, `s=0.99,b=64,approx=true` | `38.510s` | `31.643s` | `3.857s` | `2.842s` | `1.312x` | `1.541x` |
| piecewise original, `s=0.999,b=64,approx=true` | `37.976s` | `31.369s` | `3.660s` | `2.778s` | `1.330x` | `1.562x` |
| exact-selected-only, `s=0.999,b=64` | `36.137s` | `29.930s` | `3.307s` | `2.741s` | `1.398x` | `1.642x` |
| exact-selected-only, `s=0.999,b=128` | `35.583s` | `29.465s` | `3.196s` | `2.758s` | `1.420x` | `1.667x` |
| local exact, `s=0.999,b=16` | `35.821s` | `29.641s` | `3.153s` | `2.861s` | `1.410x` | `1.656x` |
| local exact, `s=0.999,b=32` | `35.455s` | `29.423s` | `3.123s` | `2.749s` | `1.425x` | `1.673x` |
| local exact + FP4 attn `to_out+bias+gate`, `s=0.999,b=32` | `34.861s` | `28.848s` | `3.051s` | `2.790s` | `1.449x` | `1.702x` |
| local exact, `s=0.999,b=64` | `36.343s` | `30.071s` | `3.188s` | `2.898s` | `1.390x` | `1.633x` |
| local exact, `s=0.999,b=128` | `36.101s` | `29.901s` | `3.182s` | `2.849s` | `1.399x` | `1.643x` |

当前最快证据文件：

```text
outputs/ltx23-dev-1080p10s-nvfp4-video-attn-ffn-piecewise-localexact-attnoutgate-s0999-b32-fp4biasgelu-projoutgate-pipeline/perf.json
outputs/ltx23-dev-1080p10s-nvfp4-video-attn-ffn-piecewise-localexact-attnoutgate-s0999-b32-fp4biasgelu-projoutgate-pipeline/summary.json
```

保守 score-routed exact-only 对照证据文件：

```text
outputs/ltx23-dev-1080p10s-nvfp4-video-attn-ffn-piecewise-exactonly-s0999-b128-fp4biasgelu-projoutgate-pipeline/perf.json
outputs/ltx23-dev-1080p10s-nvfp4-video-attn-ffn-piecewise-exactonly-s0999-b128-fp4biasgelu-projoutgate-pipeline/summary.json
```

按 full measured total 算，当前最快配置是 `50.519 / 34.861 = 1.449x`，还没有达到“在 50s 基础上进一步 1.5x”的 `33.679s` 目标；距离目标还差 `1.182s`。按旧 BF16 fully optimized baseline `59.332s` 算是 `1.702x`。

decode-profile 严格口径也没有达到目标：

- profile 文件：`outputs/ltx23-dev-1080p10s-nvfp4-video-attn-ffn-piecewise-exactonly-s0999-b128-decode-profile/perf.json`
- `DIT + video_vae_decode = 30.058s + 3.254s + 1.686s = 34.998s`
- `total - video_postprocess = 36.347s - 0.875s = 35.472s`
- 两个口径都高于 `33.679s`。

### 有损范围

Sparse attention 的有损范围按 stage/layer 是：

| Stage | Layer | 有损内容 |
| --- | --- | --- |
| Stage 1 denoise | `transformer_blocks.*.attn1` video self-attention | local exact sparse attention；当前最快 `s=0.999,b=32` 下每个 query block 只看同位置 local KV block，丢弃其他 KV block。 |
| Stage 2 refine | `transformer_blocks.*.attn1` video self-attention | 同上。 |
| Stage 1/2 | `attn2` prompt cross-attention、A2V/V2A cross-attention、audio self-attention | 当前默认不做 sparse，走 dense fallback。 |
| Stage 1/2 | selected video `attn1/attn2` projection 和 FFN linear | 继续沿用 selective NVFP4，属于低精度量化误差。 |
| Stage 1/2 | selected FP4 attention `to_out` projection | 当前 fastest 默认融合 `to_out + bias + gate`，属于 FP4 selected linear 低精度路径的一部分。 |

因此当前最快 sparse 版本的算法误差来自两处：

- selective NVFP4：weight/activation FP4 quantization。
- local exact sparse attention：直接固定 local block，不再保留原始 piecewise centroid approximation，也不再做 score/top-k routing。

### 不保留的 Sparse 候选

| 候选 | 结果 | 结论 |
| --- | ---: | --- |
| `approx=true,b=128,s=0.95` | `42.071s` | 比 b64 慢。 |
| `approx=true,s=0.9975,b=64` | `38.221s` | 比 `s=0.99`/`0.999` 波动更慢。 |
| exact-only `b=256` | `37.853s` | block 太大，exact attention 计算增加，负收益。 |
| exact-only `b=512` | 失败 | Triton 在 512x512 dot lowering 阶段失败，不保留。 |
| local exact `b=16` | `35.821s` | 比 local exact `b=32` 慢。 |
| local exact `b=32`, 不开 FP4 attn `to_out+bias+gate` | `35.455s` | 慢于当前 fastest `34.861s`。 |
| local exact `b=64/b=128` | `36.343s` / `36.101s` | 慢于 local exact `b=32`。 |
| cached-index local exact `b=32` | `35.586s` | 固定 shape 下缓存 block index 反而慢于动态生成。 |
| local-only custom forward kernel `b=32` | `35.834s` | 去掉 indices/lse 通路后仍慢于当前 Triton exact sparse kernel。 |


## 不计入提升或被拒绝的候选

以下项做过验证，但不进入最终版：

- 官方 two-stage/resident two-stage：这是 baseline 配置，不算作优化。
- `SGLANG_DIFFUSION_DECODE_PROFILE=1`：只是 instrumentation。
- Fast video postprocess：strict 口径在 video VAE decode 完成即停止，因此不计。
- Whole-model compile / CUDA graph：无明显收益或负收益。
- FlashAttention split tuning、SM margin tuning：不保留。
- 全局 video QKV、KV、Q+gate、standalone gate projection fusion：不保留。
- A2V gate-to-output compile、stage-2 audio gate-to-output compile：不保留。
- Cross-attention dual modulation：full run 负收益。
- Ada direct norm/residual fusion：full run 负收益。
- Video-only FFN 变体、FFN output projection + residual/gate epilogue：不保留。
- Final output LayerNorm + scale/shift fusion：microbench video 正收益，但 full run 负收益。
- RoPE embedding cache：不保留。
- VAE temporal padding direct slice：microbench 正收益，但 full run `67.873s` full / `63.552s` strict，负收益。
- DeepGEMM / FlashInfer GEMM replacement：当前环境中失败或慢于 cuBLASLt。

最终版原则是：只保留 full 1080p/10s 目标 workload 上有正收益的选项。

## 为什么当前优化集中在 fusion，而不是改算法

当前剩余 DiT 成本主要在：

- attention core kernel。
- 大型 cuBLASLt linear projection。
- FFN projection GEMM。

CODA-style roadmap 适合处理 GEMM 周围的 epilogue、norm、activation、residual、modulation/reduction。对 LTX2 当前代码来说，可稳定落地并有正收益的是：

- 把 RMSNorm、scale/shift、residual gate 融到一个 kernel。
- 把 Q/K norm 和 RoPE 融到一个 kernel。
- 把两个分支共用的 RMSNorm 结果复用。
- 把 Ada value 生成从多个小算子合成一个 kernel。
- 把 FFN `proj_in` 的 bias/GELU 放进 fused addmm activation。
- 对固定 shape 的 gate-to-out 和 tiled VAE decoder 使用 compile。
- 利用 CFG/STG branch 在 prefix 阶段的等价性减少重复 branch 计算。

对前面的 fusion-only 版本而言，没有采用 approximate attention、step skipping、LoRA 改写、scheduler 改写或 stage 设计改写；提升来自实现层面的 kernel/graph 调度优化。后续 sparse/NVFP4 叠加版本已经引入算法层面的低精度和 sparse attention 折中，需按第 12/13 节单独看待。
