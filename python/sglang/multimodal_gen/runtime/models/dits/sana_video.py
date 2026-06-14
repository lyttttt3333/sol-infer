# SPDX-License-Identifier: Apache-2.0
#
# SANA-Video 3D DiT (SanaVideoTransformer3DModel), ported natively into the
# sglang multimodal_gen runtime for parity with the diffusers reference
# (diffusers.models.transformers.transformer_sana_video). Architecture:
#   - Conv3d patch embed (patch (1,2,2))
#   - per block: adaLN-modulated LayerNorm -> linear self-attention (ReLU
#     feature map + 3D Wan RoPE) -> gate; softmax cross-attention to Gemma2
#     text; adaLN-modulated LayerNorm -> GLUMBTempConv (conv FFN w/ temporal
#     conv) -> gate.
#   - AdaLayerNormSingle timestep -> 6 modulation params; PixArt caption proj +
#     RMSNorm; SanaModulatedNorm output head; 3D unpatchify.
# Weight names match the diffusers checkpoint exactly for direct loading.
#
# Parity-critical bits (WanRotaryPosEmbed, the rotary-linear-attn normalization,
# GLUMBTempConv) are ported faithfully from the reference. Reuses the validated
# image-SANA SanaAdaLayerNormSingle (identical weights) and sglang RMSNorm.

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    get_1d_rotary_pos_embed,
)

from sglang.multimodal_gen.configs.models.dits.sana_video import SanaVideoConfig
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.sana import SanaAdaLayerNormSingle
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanRotaryPosEmbed(nn.Module):
    """3D rotary position embedding (frame/height/width), ported verbatim from
    diffusers SANA-Video so the freq layout matches the checkpoint exactly."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self.t_dim, self.h_dim, self.w_dim = t_dim, h_dim, w_dim

        # Lazily computed in forward (plain attrs, NOT registered buffers) so
        # the meta-device weight loader doesn't flag them as un-materialized.
        self._freqs_cos = None
        self._freqs_sin = None

    def _ensure_freqs(self, device: torch.device):
        if self._freqs_cos is not None and self._freqs_cos.device == device:
            return
        freqs_cos, freqs_sin = [], []
        for dim in [self.t_dim, self.h_dim, self.w_dim]:
            fc, fs = get_1d_rotary_pos_embed(
                dim,
                self.max_seq_len,
                self.theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=torch.float64,
            )
            freqs_cos.append(fc)
            freqs_sin.append(fs)
        self._freqs_cos = torch.cat(freqs_cos, dim=1).to(device)
        self._freqs_sin = torch.cat(freqs_sin, dim=1).to(device)

    def forward(self, hidden_states: torch.Tensor):
        self._ensure_freqs(hidden_states.device)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self._freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self._freqs_sin.split(split_sizes, dim=1)

        fcf = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        fch = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        fcw = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        fsf = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        fsh = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        fsw = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([fcf, fch, fcw], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([fsf, fsh, fsw], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        return freqs_cos, freqs_sin


def _apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class SanaVideoLinearAttention(nn.Module):
    """ReLU-feature-map linear self-attention with 3D RoPE. Ports the math of
    diffusers SanaLinearAttnProcessor3_0 exactly (RoPE on the KV-aggregation
    matmuls; normalizer uses the non-rotated relu features)."""

    def __init__(self, query_dim, num_heads, head_dim, qk_norm=True, bias=False):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )
        self.norm_q = RMSNorm(inner_dim) if qk_norm else None
        self.norm_k = RMSNorm(inner_dim) if qk_norm else None
        # diffusers casts the KV-aggregation to fp32 -> slow SIMT sgemm (~9% of the
        # DiT, profiled). bf16 tensor cores accumulate in fp32 anyway, so this env
        # keeps the aggregation in bf16 (tensor-core, compile-friendly). OFF==fp32.
        self._bf16_agg = os.environ.get("SGLANG_SANA_LINATTN_BF16", "0") in ("1", "true", "True")
        # Fused QKV projection (lossless: concat to_q/k/v weights -> one GEMM, 3
        # launches -> 1). Built in post_load_weights. OFF == separate projections.
        self._qkv_merge = os.environ.get("SGLANG_SANA_QKV_MERGE", "0") in ("1", "true", "True")
        self._qkv_w = None

    def build_qkv_merge(self):
        if not self._qkv_merge:
            return
        # to_q/k/v are bias-free (attention_bias=False) -> just concat the weights.
        self._qkv_w = torch.cat(
            [self.to_q.weight, self.to_k.weight, self.to_v.weight], dim=0
        ).detach()

    def forward(self, hidden_states, rotary_emb):
        original_dtype = hidden_states.dtype

        if self._qkv_w is not None:
            query, key, value = F.linear(hidden_states, self._qkv_w).chunk(3, dim=-1)
        else:
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # B, N, H, C
        query = query.unflatten(2, (self.num_heads, -1))
        key = key.unflatten(2, (self.num_heads, -1))
        value = value.unflatten(2, (self.num_heads, -1))

        query = F.relu(query)
        key = F.relu(key)

        query_rotate = _apply_rotary_emb(query, *rotary_emb)
        key_rotate = _apply_rotary_emb(key, *rotary_emb)

        # B, H, C, N
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        query_rotate = query_rotate.permute(0, 2, 3, 1)
        key_rotate = key_rotate.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)

        if not self._bf16_agg:
            query_rotate, key_rotate, value = (
                query_rotate.float(),
                key_rotate.float(),
                value.float(),
            )

        z = 1 / (key.sum(dim=-1, keepdim=True).transpose(-2, -1) @ query + 1e-15)
        scores = torch.matmul(value, key_rotate.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query_rotate)
        hidden_states = hidden_states * z

        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class SanaVideoCrossAttention(nn.Module):
    """Softmax (SDPA) cross-attention to Gemma2 text embeddings, with qk_norm."""

    def __init__(self, query_dim, cross_attention_dim, num_heads, head_dim, qk_norm=True):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=True)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )
        self.norm_q = RMSNorm(inner_dim) if qk_norm else None
        self.norm_k = RMSNorm(inner_dim) if qk_norm else None

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None):
        B, S, _ = hidden_states.shape
        T = encoder_hidden_states.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if encoder_attention_mask is not None:
            # encoder_attention_mask is an additive bias [B, 1, T] or bool [B, T]
            if encoder_attention_mask.dtype == torch.bool:
                attn_mask = encoder_attention_mask[:, None, None, :].expand(
                    B, self.num_heads, S, T
                )
            else:
                attn_mask = encoder_attention_mask.view(B, 1, -1, T).expand(
                    B, self.num_heads, S, T
                )

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(B, S, -1)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class GLUMBTempConv(nn.Module):
    """Gated conv FFN with temporal aggregation (diffusers GLUMBTempConv,
    norm_type=None, residual_connection=False for the SANA-Video checkpoint)."""

    def __init__(self, in_channels, out_channels, expand_ratio=3.0):
        super().__init__()
        hidden_channels = int(expand_ratio * in_channels)
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2
        )
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)
        self.conv_temp = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False
        )

    def forward(self, hidden_states):
        # hidden_states: [B, F, H, W, C]
        batch_size, num_frames, height, width, num_channels = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size * num_frames, height, width, num_channels
        ).permute(0, 3, 1, 2)

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)

        # Temporal aggregation
        hidden_states_temporal = hidden_states.view(
            batch_size, num_frames, num_channels, height * width
        ).permute(0, 2, 1, 3)
        hidden_states = hidden_states_temporal + self.conv_temp(hidden_states_temporal)
        hidden_states = hidden_states.permute(0, 2, 3, 1).view(
            batch_size, num_frames, height, width, num_channels
        )
        return hidden_states


class SanaVideoModulatedNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, hidden_states, temb, scale_shift_table):
        hidden_states = self.norm(hidden_states)
        shift, scale = (
            scale_shift_table[None, None] + temb[:, :, None].to(scale_shift_table.device)
        ).unbind(dim=2)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class SanaVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        num_cross_attention_heads,
        cross_attention_head_dim,
        cross_attention_dim,
        mlp_ratio,
        norm_eps,
        qk_norm=True,
        attention_bias=False,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = SanaVideoLinearAttention(
            query_dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qk_norm=qk_norm,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn2 = SanaVideoCrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
            qk_norm=qk_norm,
        )

        self.ff = GLUMBTempConv(dim, dim, expand_ratio=mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        frames,
        height,
        width,
        rotary_emb,
        encoder_attention_mask=None,
    ):
        batch_size = hidden_states.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None]
            + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
        ).unbind(dim=2)

        # 1. Self attention (linear + RoPE)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)
        attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = hidden_states + gate_msa * attn_output

        # 2. Cross attention (softmax)
        attn_output = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward (conv FFN with temporal conv)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_hidden_states = norm_hidden_states.unflatten(1, (frames, height, width))
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(1, 3)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


class _TaylorSeerState:
    """Finite-difference Taylor forecast of a feature across denoise steps
    (ported from cache_dit TaylorSeerState). order = n_derivatives + 1.

    On computed steps it maintains a difference table dY (dY[0]=Y,
    dY[i+1]=(dY[i]-dY_prev[i])/window); on skipped steps it forecasts via the
    Taylor series Y_hat = sum_i dY[i] * elapsed^i / i!. n_derivatives=1 = linear
    extrapolation, 2 = quadratic, vs the order-0 'hold constant' of plain reuse.
    """

    def __init__(self, n_derivatives=1, max_warmup_steps=3, skip_interval_steps=2):
        self.n_derivatives = n_derivatives
        self.order = n_derivatives + 1
        self.max_warmup_steps = max_warmup_steps
        self.skip_interval_steps = max(1, skip_interval_steps)
        self.current_step = -1
        self.last_non_approximated_step = -1
        self.dY_prev = [None] * self.order
        self.dY_current = [None] * self.order

    def mark_step_begin(self):
        self.current_step += 1

    def should_compute(self):
        s = self.current_step
        return (
            s < self.max_warmup_steps
            or (s - self.max_warmup_steps + 1) % self.skip_interval_steps == 0
        )

    def _derivative(self, Y):
        dY = [None] * self.order
        dY[0] = Y
        window = max(1, self.current_step - self.last_non_approximated_step)
        for i in range(self.n_derivatives):
            if self.dY_prev[i] is not None and self.current_step > 1:
                dY[i + 1] = (dY[i] - self.dY_prev[i]) / window
            else:
                break
        return dY

    def update(self, Y):
        self.dY_prev = self.dY_current
        self.dY_current = self._derivative(Y)
        self.last_non_approximated_step = self.current_step

    def approximate(self):
        elapsed = self.current_step - self.last_non_approximated_step
        out = None
        for i, d in enumerate(self.dY_current):
            if d is None:
                break
            term = d if i == 0 else (1.0 / math.factorial(i)) * d * (elapsed**i)
            out = term if out is None else out + term
        return out


class SanaVideoTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):

    _fsdp_shard_conditions = [
        lambda n, m: isinstance(m, SanaVideoTransformerBlock),
    ]
    _compile_conditions = [
        lambda n, m: isinstance(m, SanaVideoTransformerBlock),
    ]
    param_names_mapping = SanaVideoConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = {}

    def __init__(self, config: SanaVideoConfig, hf_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config or {}, **kwargs)

        arch = config.arch_config
        self.out_channels = arch.out_channels
        self.patch_size = tuple(arch.patch_size)
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim

        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.out_channels

        # 1. Patch + position embedding
        self.rope = WanRotaryPosEmbed(
            arch.attention_head_dim, self.patch_size, arch.rope_max_seq_len
        )
        self.patch_embedding = nn.Conv3d(
            arch.in_channels, self.inner_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        # 2. Timestep / caption embeddings
        self.time_embed = SanaAdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=arch.caption_channels, hidden_size=self.inner_dim
        )
        self.caption_norm = RMSNorm(self.inner_dim)

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SanaVideoTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=arch.num_attention_heads,
                    attention_head_dim=arch.attention_head_dim,
                    num_cross_attention_heads=arch.num_cross_attention_heads,
                    cross_attention_head_dim=arch.cross_attention_head_dim,
                    cross_attention_dim=arch.cross_attention_dim,
                    mlp_ratio=arch.mlp_ratio,
                    norm_eps=arch.norm_eps,
                    qk_norm=arch.qk_norm is not None,
                    attention_bias=arch.attention_bias,
                )
                for _ in range(arch.num_layers)
            ]
        )

        # 4. Output
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.norm_out = SanaVideoModulatedNorm(self.inner_dim, eps=arch.norm_eps)
        self.proj_out = nn.Linear(self.inner_dim, math.prod(self.patch_size) * self.out_channels)

        self.layer_names = ["transformer_blocks"]

        # --- TeaCache toggle (timestep-embedding-aware step skipping) ---
        # thresh<=0 => OFF == byte-identical baseline; higher => more skips (跳步幅度).
        # All per-step cache state is keyed by CFG branch (0=first call of a step,
        # 1=second) because sglang runs CFG unbatched (2 forward calls/step); cond
        # and uncond must not contaminate each other's rel-L1 / cached residual.
        self._tc_thresh = float(os.environ.get("SGLANG_SANA_TEACACHE_THRESH", "0") or 0.0)
        _co = os.environ.get("SGLANG_SANA_TEACACHE_COEFFS", "")
        # Calibrated input-relL1 -> output-relL1 polynomial (highest-degree-first,
        # Horner). Default [1,0]=identity == UNcalibrated; fit per model via the
        # SGLANG_SANA_TEACACHE_CALIB collection pass + scripts/fit_sana_teacache.py.
        self._tc_coeffs = [float(x) for x in _co.split(",")] if _co else [1.0, 0.0]
        # Late-step skip (fine-grained, composition-preserving): compute steps
        # [0, skip_from) per branch, reuse the residual for the rest. 0 = disabled.
        # Keeps early/mid steps (which set the layout) so the output stays close to
        # the dense sample; only late refinement is dropped.
        self._tc_skip_from = int(os.environ.get("SGLANG_SANA_SKIP_FROM_STEP", "0") or 0)
        # Calibration-collection: when set to a path, run dense and append
        # "branch input_relL1 output_relL1" per step for the offline polynomial fit.
        self._tc_calib = os.environ.get("SGLANG_SANA_TEACACHE_CALIB", "")
        if self._tc_calib:
            open(self._tc_calib, "w").close()  # truncate once at build
        self._tc_prev_mod = {}
        self._tc_prev_residual = {}  # per-branch: calibration + first-compute check
        self._tc_prev_b0 = {}  # calib: first-block OUTPUT (FBCache-style signal)
        # Shared (common-mode) residual reused on skipped steps. CFG is unbatched
        # and both branches get the SAME latent x_t, so reusing one shared residual
        # makes the guidance term s*(R-R)=0 vanish on skipped steps -- riding the
        # established composition. A per-branch residual would instead re-inject a
        # 6x-amplified STALE guidance term -> high-freq artifacts. Shared is correct.
        self._tc_reuse_residual = None
        self._tc_accum = {}
        self._tc_step = {}
        self._tc_prev_t = None
        self._cache_prev_t = None  # CFG-branch detection (shared-timestep)
        # TaylorSeer: forecast the block-stack residual on skipped steps via an
        # order-N finite-difference Taylor expansion (0 = off). Per-branch state.
        # Compute the first WARMUP steps then every INTERVAL-th step; forecast rest.
        self._ts_order = int(os.environ.get("SGLANG_SANA_TAYLORSEER_ORDER", "0") or 0)
        self._ts_warmup = int(os.environ.get("SGLANG_SANA_TAYLORSEER_WARMUP", "3") or 3)
        self._ts_interval = int(os.environ.get("SGLANG_SANA_TAYLORSEER_INTERVAL", "2") or 2)
        self._ts_states = {}
        # EasyCache: calibration-free adaptive skip (ported from the LTX-2 stage1
        # cache core). Measures the online input->output transformation rate K, then
        # estimates per-step relative output change ~= K * input_change / out_norm,
        # accumulates it, and skips while the accumulator stays below threshold.
        # CFG is unbatched and both branches get the SAME latent x_t -> the decision
        # is identical, so decide once on branch 0 and reuse the shared residual.
        self._ec_thresh = float(os.environ.get("SGLANG_SANA_EASYCACHE_THRESH", "0") or 0.0)
        self._ec_warmup = int(os.environ.get("SGLANG_SANA_EASYCACHE_WARMUP", "3") or 3)
        self._ec_sub = int(os.environ.get("SGLANG_SANA_EASYCACHE_SUBSAMPLE", "8") or 8)
        self._ec_x_prev = None
        self._ec_out_prev = None
        self._ec_out_prev_norm = None
        self._ec_rate = None
        self._ec_cumulative = 0.0
        self._ec_run = True
        self._ec_debug = bool(os.environ.get("SGLANG_SANA_EASYCACHE_DEBUG", ""))
        # One-shot component profiler (eager): SGLANG_SANA_PROFILE=1 -> torch.profiler
        # the block stack on the first call and print the per-op CUDA-time breakdown.
        self._profile = bool(os.environ.get("SGLANG_SANA_PROFILE", ""))
        self._profiled = False

    def _tc_reset_if_new_gen(self, t_scalar):
        # New generation when the timestep jumps back up (e.g. warmup -> real run).
        if self._tc_prev_t is None or t_scalar > self._tc_prev_t + 1e-4:
            self._tc_prev_mod = {}
            self._tc_prev_residual = {}
            self._tc_prev_b0 = {}
            self._tc_reuse_residual = None
            self._tc_accum = {}
            self._tc_step = {}
            self._ts_states = {}
            self._ec_x_prev = None
            self._ec_out_prev = None
            self._ec_out_prev_norm = None
            self._ec_rate = None
            self._ec_cumulative = 0.0
            self._ec_run = True
            self._cache_prev_t = None
        self._tc_prev_t = t_scalar

    def _tc_thr_decide(self, branch, mod_inp):
        """Threshold mode (per CFG branch): accumulate the calibrated rescale of
        the block-0 modulated-input rel-L1; recompute when it crosses the
        threshold, else reuse this branch's cached residual."""
        prev = self._tc_prev_mod.get(branch)
        self._tc_prev_mod[branch] = mod_inp.detach()
        if prev is None:
            self._tc_accum[branch] = 0.0
            return True
        diff = (mod_inp - prev).abs().mean()
        denom = prev.abs().mean().clamp_min(1e-9)
        rel_l1 = float((diff / denom).item())
        rescaled = 0.0
        for c in self._tc_coeffs:  # horner, highest-degree-first
            rescaled = rescaled * rel_l1 + c
        acc = self._tc_accum.get(branch, 0.0) + rescaled
        if acc >= self._tc_thresh:
            self._tc_accum[branch] = 0.0
            return True
        self._tc_accum[branch] = acc
        return False

    def _tc_calib_record(self, branch, mod_inp, block0_out, residual):
        """Calibration: append (branch, input_relL1, block0out_relL1, output_relL1)
        for this step vs the previous computed step of the same branch. Runs under
        dense so the rows describe the true per-step drift. Two candidate cheap
        signals are logged: block-0 *input* (TeaCache) and block-0 *output*
        (FBCache) -- the fit script reports which actually predicts output drift."""

        def rel(a, b):
            return float(((a - b).abs().mean() / b.abs().mean().clamp_min(1e-9)).item())

        pm = self._tc_prev_mod.get(branch)
        pb = self._tc_prev_b0.get(branch)
        pr = self._tc_prev_residual.get(branch)
        if pm is not None and pb is not None and pr is not None:
            with open(self._tc_calib, "a") as f:
                f.write(f"{branch} {rel(mod_inp, pm):.8f} {rel(block0_out, pb):.8f} {rel(residual, pr):.8f}\n")
        self._tc_prev_mod[branch] = mod_inp.detach()
        self._tc_prev_b0[branch] = block0_out.detach()

    def _ec_decide(self, step, x):
        """EasyCache skip decision (shared per step). Returns run_blocks: compute
        until warmup/rate is established, then accumulate the estimated relative
        output change (K * input_change / out_norm) and compute once it crosses the
        threshold; otherwise skip and reuse the shared residual."""
        if (
            step < self._ec_warmup
            or self._ec_x_prev is None
            or self._ec_rate is None
            or self._ec_out_prev_norm is None
        ):
            return True
        cur = x[:, :: self._ec_sub].float()
        input_change = (cur - self._ec_x_prev).abs().mean()
        approx = float((self._ec_rate * input_change / max(self._ec_out_prev_norm, 1e-6)).item())
        self._ec_cumulative += approx
        return self._ec_cumulative >= self._ec_thresh

    def _ec_update(self, x, out):
        """EasyCache state update on a computed step (branch-0/cond signal): refresh
        the online transformation rate K = out_change / input_change, the previous
        subsampled tensors, the output norm, and reset the accumulator."""
        cur_x = x[:, :: self._ec_sub].float()
        cur_o = out[:, :: self._ec_sub].float()
        if self._ec_x_prev is not None and self._ec_out_prev is not None:
            inc = (cur_x - self._ec_x_prev).abs().mean()
            outc = (cur_o - self._ec_out_prev).abs().mean()
            if float(inc.item()) > 1e-12:
                self._ec_rate = float((outc / inc).item())
        self._ec_x_prev = cur_x.detach()
        self._ec_out_prev = cur_o.detach()
        self._ec_out_prev_norm = float(out.float().abs().mean().item())
        self._ec_cumulative = 0.0

    # --- compiled hot path vs. eager cache control ----------------------------
    # sglang compiles the WHOLE DiT forward (denoising.py: module.compile()). To
    # let a skip-cache coexist with torch.compile, the only thing dynamo should
    # trace is the transformer-block stack (_run_blocks, stable shapes -> compiles
    # once, fuses, stays cached). Every cache decision / Python-state mutation /
    # .item() sync lives in @torch.compiler.disable methods below, so the skip
    # schedule is invisible to dynamo and never triggers guard-failure recompiles.

    def _run_blocks(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_mod,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        rotary_emb,
        encoder_attention_mask,
    ):
        args = (encoder_hidden_states, timestep_mod, post_patch_num_frames,
                post_patch_height, post_patch_width, rotary_emb)
        if self._profile and not self._profiled:
            return self._profile_blocks(hidden_states, args, encoder_attention_mask)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, *args, encoder_attention_mask=encoder_attention_mask)
        return hidden_states

    @torch.compiler.disable
    def _profile_blocks(self, hidden_states, args, encoder_attention_mask):
        import torch.profiler as tp

        self._profiled = True
        with tp.profile(activities=[tp.ProfilerActivity.CUDA], record_shapes=True) as prof:
            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, *args, encoder_attention_mask=encoder_attention_mask)
            torch.cuda.synchronize()
        print("==== SANA-Video block-stack component profile (20 layers, eager) ====", flush=True)
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=35), flush=True)
        print("==== END component profile ====", flush=True)
        return hidden_states

    @torch.compiler.disable
    def _run_blocks_capture0(self, *args):
        # Calibration only (eager, never stacked with compile): also return the
        # first block's output for the FBCache-signal probe.
        hidden_states = args[0]
        block0_out = None
        for bi, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
                encoder_attention_mask=args[7],
            )
            if bi == 0:
                block0_out = hidden_states
        return hidden_states, block0_out

    @torch.compiler.disable
    def _cache_decide(self, timestep, hidden_states, timestep_mod, batch_size):
        """Eager (dynamo-opaque) per-step skip decision + state bookkeeping."""
        ts_on = self._ts_order > 0
        thr_on = self._tc_thresh > 0.0
        late_on = self._tc_skip_from > 0
        ec_on = self._ec_thresh > 0.0
        calib_on = bool(self._tc_calib)
        t_now = float(timestep.flatten()[0].item())
        self._tc_reset_if_new_gen(t_now)
        # CFG is unbatched: cond & uncond of a step share the timestep.
        branch = 1 if (self._cache_prev_t is not None and abs(t_now - self._cache_prev_t) < 1e-4) else 0
        self._cache_prev_t = t_now
        step = self._tc_step.get(branch, 0)
        self._tc_step[branch] = step + 1

        run_blocks = True
        ts_state = None
        mod_inp = None
        if ts_on:
            ts_state = self._ts_states.get(branch)
            if ts_state is None:
                ts_state = _TaylorSeerState(self._ts_order, self._ts_warmup, self._ts_interval)
                self._ts_states[branch] = ts_state
            ts_state.mark_step_begin()
            run_blocks = ts_state.should_compute()
        elif late_on:
            run_blocks = (step < self._tc_skip_from) or (self._tc_reuse_residual is None)
        elif ec_on:
            if branch == 0:
                run_blocks = self._ec_decide(step, hidden_states)
                self._ec_run = run_blocks
                if self._ec_debug:
                    print(f"EC step={step} run={int(run_blocks)} "
                          f"cum={self._ec_cumulative:.4f} K={self._ec_rate}", flush=True)
            else:
                run_blocks = self._ec_run
        elif thr_on or calib_on:
            b0 = self.transformer_blocks[0]
            shift0, scale0 = (
                b0.scale_shift_table[None, None]
                + timestep_mod.reshape(batch_size, timestep_mod.shape[1], 6, -1)
            ).unbind(dim=2)[:2]
            mod_inp = b0.norm1(hidden_states) * (1 + scale0) + shift0
            run_blocks = True if calib_on else self._tc_thr_decide(branch, mod_inp)
        return {
            "run_blocks": run_blocks, "branch": branch, "ts_state": ts_state,
            "mod_inp": mod_inp, "ts_on": ts_on, "ec_on": ec_on, "calib_on": calib_on,
        }

    @torch.compiler.disable
    def _cache_after_compute(self, decision, hidden_before, hidden_states, block0_out):
        """Eager: update cache state after a computed step (residual / rate / forecast)."""
        block_residual = hidden_states - hidden_before
        if decision["ts_on"]:
            # TaylorSeer forecasts the residual (order-N generalization of hold).
            decision["ts_state"].update(block_residual)
            return
        branch = decision["branch"]
        if decision["calib_on"] and decision["mod_inp"] is not None:
            self._tc_calib_record(branch, decision["mod_inp"], block0_out, block_residual)
        if decision["ec_on"] and branch == 0:
            self._ec_update(hidden_before, hidden_states)
        rdet = block_residual.detach()
        self._tc_prev_residual[branch] = rdet  # per-branch (calib / drift)
        self._tc_reuse_residual = rdet  # shared common-mode residual for reuse

    @torch.compiler.disable
    def _cache_reuse(self, decision, hidden_before, hidden_states):
        """Eager: skipped step -> reuse cached transformation (CFG common-mode)."""
        if decision["ts_on"]:
            return hidden_before + decision["ts_state"].approximate()
        return hidden_states + self._tc_reuse_residual

    def post_load_weights(self) -> None:
        # SANA-Video runs the whole transformer in bf16 (matching diffusers).
        # sglang's loader can leave Conv params in fp32 while Linears are bf16,
        # which breaks conv(bf16 input, fp32 weight); unify everything to bf16.
        self.to(torch.bfloat16)
        # Optional selective W4A4 NVFP4 on attn GEMMs + fp4/fp8 on the conv-FFN
        # conv_inverted 1x1 (the one high-N/K GEMM that wins at low precision).
        # All env-gated, OFF == baseline.
        from sglang.multimodal_gen.runtime.models.dits.sana_video_nvfp4 import (
            maybe_swap_attn_to_fp4,
            maybe_swap_ffn_lowprec,
        )
        maybe_swap_attn_to_fp4(self)
        maybe_swap_ffn_lowprec(self)
        # Lossless QKV merge for self-attention (env-gated, built after bf16 cast).
        for blk in self.transformer_blocks:
            attn1 = getattr(blk, "attn1", None)
            if attn1 is not None and hasattr(attn1, "build_qkv_merge"):
                attn1.build_qkv_merge()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        guidance: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("SANA-Video forward requires encoder_hidden_states")

        if isinstance(encoder_hidden_states, (list, tuple)):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = encoder_attention_mask[0]

        # convert encoder_attention_mask (1=keep,0=discard) -> bias, like diffusers
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        timestep_mod, embedded_timestep = self.time_embed(
            timestep, hidden_dtype=hidden_states.dtype
        )
        timestep_mod = timestep_mod.view(batch_size, -1, timestep_mod.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.shape[-1]
        )
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        any_cache = (
            self._ts_order > 0
            or self._tc_thresh > 0.0
            or self._tc_skip_from > 0
            or self._ec_thresh > 0.0
            or bool(self._tc_calib)
        )
        if not any_cache:
            # Dense / fusion-only: the block stack is the whole compiled hot path,
            # with no cache machinery in the graph.
            hidden_states = self._run_blocks(
                hidden_states,
                encoder_hidden_states,
                timestep_mod,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                rotary_emb,
                encoder_attention_mask,
            )
        else:
            # Cache control is eager (dynamo-opaque); _run_blocks stays compiled.
            decision = self._cache_decide(timestep, hidden_states, timestep_mod, batch_size)
            hidden_before = hidden_states
            if decision["run_blocks"]:
                if decision["calib_on"]:
                    hidden_states, block0_out = self._run_blocks_capture0(
                        hidden_states,
                        encoder_hidden_states,
                        timestep_mod,
                        post_patch_num_frames,
                        post_patch_height,
                        post_patch_width,
                        rotary_emb,
                        encoder_attention_mask,
                    )
                else:
                    hidden_states = self._run_blocks(
                        hidden_states,
                        encoder_hidden_states,
                        timestep_mod,
                        post_patch_num_frames,
                        post_patch_height,
                        post_patch_width,
                        rotary_emb,
                        encoder_attention_mask,
                    )
                    block0_out = None
                self._cache_after_compute(decision, hidden_before, hidden_states, block0_out)
            else:
                hidden_states = self._cache_reuse(decision, hidden_before, hidden_states)

        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return output


EntryClass = SanaVideoTransformer3DModel
