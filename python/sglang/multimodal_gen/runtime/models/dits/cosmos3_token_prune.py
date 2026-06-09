# Copyright 2025 SGLang authors
#
# Feature-norm-based GEN-token pruning for Cosmos3.
#
# Adapted (migrated) from the LTX2 stage-1 token-pruning method on the
# `clean_full_opt` branch:
#   python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py
#     _ltx2_stage1_prune_ratio / _ltx2_stage1_prune_method /
#     _ltx2_stage1_prune_steps / _ltx2_stage1_prune_compensation /
#     _ltx2_stage2_midpoint_keep_indices (scoring + top-K selection)
#
# Method: at selected denoising steps, score each GEN (video patch) token,
# keep the top-K = round(N * r) by score, run the transformer GEN layers on
# ONLY those K tokens, then scatter the K-token result back to full N and fill
# the dropped tokens with a compensation hidden state (the previous step's, or
# zero). Default OFF (keep_ratio >= 1.0) == byte-identical to baseline.
#
# Cosmos3 adaptation vs the LTX2 reference:
#   * LTX2 prunes in the denoising STAGE and DISABLES under SP>1. Cosmos3
#     tokenizes + shards inside the transformer forward, so pruning lives in
#     the forward and selection is PER-RANK-LOCAL: every SP rank prunes its own
#     `local_seq_len` shard to the SAME K' = round(local_seq_len * r), which
#     keeps the USP all-to-all shards balanced. On 1 GPU this reduces to a
#     global top-K (identical to the reference on the full sequence).
#   * The 'prev' compensation is applied at the post-loop hidden level (before
#     the per-token norm_moe_gen + proj_out), which is exactly equivalent to the
#     reference's per-token previous-velocity fill.

from __future__ import annotations

import os

import torch

# --- env-config readers (mirror the LTX2 SGLANG_LTX2_STAGE1_* knobs) ---


def prune_ratio() -> float:
    """Keep-ratio r in (0, 1) for GEN-token pruning.

    Returns 1.0 (feature OFF, byte-identical to baseline) when the env var is
    unset, malformed, or outside (0, 1).
    """
    raw = os.environ.get("SGLANG_COSMOS3_PRUNE_RATIO")
    if raw is None:
        return 1.0
    try:
        ratio = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not (0.0 < ratio < 1.0):
        return 1.0
    return ratio


def prune_method() -> str:
    """Token-selection method (default 'feat_norm').

    Hidden-state-based (no extra plumbing): 'feat_norm'/'feat_l2', 'feat_l1',
    'feat_linf', 'feat_var', 'uniform', 'random'.
    """
    return os.environ.get("SGLANG_COSMOS3_PRUNE_METHOD", "feat_norm").strip().lower()


def prune_steps() -> set[int] | None:
    """Optional comma/range list of step indices to prune (e.g. "5-30", "0,2,4").

    Returns None when unset, in which case the caller prunes ALL steps EXCEPT
    step 0 (step 0 always runs full so the 'prev' buffer gets seeded). Malformed
    input yields an empty set (prune nothing) rather than raising.
    """
    raw = os.environ.get("SGLANG_COSMOS3_PRUNE_STEPS")
    if raw is None:
        return None
    steps: set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, _, hi = tok.partition("-")
            try:
                lo_i, hi_i = int(lo.strip()), int(hi.strip())
            except ValueError:
                continue
            if hi_i < lo_i:
                lo_i, hi_i = hi_i, lo_i
            steps.update(range(lo_i, hi_i + 1))
        else:
            try:
                steps.add(int(tok))
            except ValueError:
                continue
    return steps


def prune_compensation() -> str:
    """How dropped GEN tokens get filled on a pruned step.

    'prev' (DEFAULT): reuse the previous step's full post-loop hidden for the
        dropped tokens (per-token equivalent to reusing the previous velocity).
        On the first pruned step (no prev buffer) the caller runs FULL to seed.
    'zero': dropped tokens get a zero post-loop hidden (near-frozen update).
    """
    return os.environ.get("SGLANG_COSMOS3_PRUNE_COMPENSATION", "prev").strip().lower()


# --- scoring + top-K selection (ported from _ltx2_stage2_midpoint_keep_indices) ---


def _uniform_indices(
    num_tokens: int, keep: int, device: torch.device
) -> torch.Tensor:
    """Deterministic ascending uniform subset: idx[i] = floor(i * N / K)."""
    arange = torch.arange(keep, device=device, dtype=torch.long)
    idx = (arange * num_tokens) // keep
    return idx.clamp_(max=num_tokens - 1)


def keep_indices(
    method: str,
    num_tokens: int,
    keep_ratio: float,
    hidden_states: torch.Tensor,
    prev_velocity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ascending kept GEN-token indices.

    Content-aware methods score each token (averaged over the batch dim so one
    1-D index set applies to all batch elements) then take the top-K; uniform /
    random are content-blind. Ascending order is preserved so the attention
    sequence stays monotone. ``hidden_states`` is the [B, S, C] local shard;
    ``prev_velocity`` (same shape, optional) backs the 'velocity' method.
    """
    device = hidden_states.device
    keep = int(round(num_tokens * keep_ratio))
    keep = max(1, min(num_tokens, keep))
    if keep >= num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)

    if method in ("velocity", "vel") and prev_velocity is not None:
        scores = prev_velocity.float().pow(2).sum(dim=-1).mean(dim=0)  # [S]
    elif method in ("feat_norm", "feat", "norm", "feat_l2"):
        scores = hidden_states.float().pow(2).sum(dim=-1).mean(dim=0)  # [S]
    elif method in ("feat_l1",):
        scores = hidden_states.float().abs().sum(dim=-1).mean(dim=0)
    elif method in ("feat_linf", "feat_max"):
        scores = hidden_states.float().abs().amax(dim=-1).mean(dim=0)
    elif method in ("feat_var",):
        scores = hidden_states.float().var(dim=-1).mean(dim=0)
    elif method in ("random", "rand"):
        gen = torch.Generator(device=device).manual_seed(42)
        perm = torch.randperm(num_tokens, generator=gen, device=device)
        return torch.sort(perm[:keep]).values
    else:  # uniform / unknown -> content-blind even stride
        return _uniform_indices(num_tokens, keep, device)

    topk = torch.topk(scores, keep, largest=True).indices
    return torch.sort(topk).values
