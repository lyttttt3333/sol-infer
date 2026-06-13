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

    def forward(self, hidden_states, rotary_emb):
        original_dtype = hidden_states.dtype

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
        self._tc_thresh = float(os.environ.get("SGLANG_SANA_TEACACHE_THRESH", "0") or 0.0)
        _co = os.environ.get("SGLANG_SANA_TEACACHE_COEFFS", "")
        self._tc_coeffs = [float(x) for x in _co.split(",")] if _co else [1.0, 0.0]
        self._tc_prev_mod = None
        self._tc_prev_residual = None
        self._tc_accum = 0.0
        self._tc_prev_t = None

    def _tc_should_calc(self, mod_inp, t_scalar):
        """Accumulate rescaled rel-L1 of the block-0 modulated input; skip (reuse
        cached residual) while under threshold. Resets per generation when the
        timestep jumps back up (e.g. warmup run -> real run)."""
        if self._tc_prev_t is None or t_scalar > self._tc_prev_t + 1e-4:
            self._tc_prev_mod = None
            self._tc_prev_residual = None
            self._tc_accum = 0.0
        self._tc_prev_t = t_scalar
        if self._tc_prev_mod is None:
            self._tc_prev_mod = mod_inp.detach()
            return True
        diff = (mod_inp - self._tc_prev_mod).abs().mean()
        denom = self._tc_prev_mod.abs().mean().clamp_min(1e-9)
        rel_l1 = float((diff / denom).item())
        self._tc_prev_mod = mod_inp.detach()
        rescaled = 0.0
        for c in self._tc_coeffs:  # horner, coeffs highest-degree-first (np.poly1d order)
            rescaled = rescaled * rel_l1 + c
        self._tc_accum += rescaled
        if self._tc_accum >= self._tc_thresh:
            self._tc_accum = 0.0
            return True
        return False

    def post_load_weights(self) -> None:
        # SANA-Video runs the whole transformer in bf16 (matching diffusers).
        # sglang's loader can leave Conv params in fp32 while Linears are bf16,
        # which breaks conv(bf16 input, fp32 weight); unify everything to bf16.
        self.to(torch.bfloat16)
        # Optional selective W4A4 NVFP4 on attn GEMMs (env-gated, OFF==baseline).
        from sglang.multimodal_gen.runtime.models.dits.sana_video_nvfp4 import (
            maybe_swap_attn_to_fp4,
        )
        maybe_swap_attn_to_fp4(self)

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

        tc_on = self._tc_thresh > 0.0
        should_calc = True
        if tc_on:
            b0 = self.transformer_blocks[0]
            shift0, scale0 = (
                b0.scale_shift_table[None, None]
                + timestep_mod.reshape(batch_size, timestep_mod.shape[1], 6, -1)
            ).unbind(dim=2)[:2]
            mod_inp = b0.norm1(hidden_states) * (1 + scale0) + shift0
            should_calc = self._tc_should_calc(mod_inp, float(timestep.flatten()[0].item()))

        if should_calc:
            hidden_before = hidden_states
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_mod,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )
            if tc_on:
                self._tc_prev_residual = (hidden_states - hidden_before).detach()
        else:
            hidden_states = hidden_states + self._tc_prev_residual

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
