# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Pyramid Attention Broadcast hooks.

Cosmos3 runs text UND attention once per prompt, so this PAB hook targets the
step-wise GEN cross-attention output inside each generation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, value, default)
        return default


@dataclass(frozen=True)
class Cosmos3PABConfig:
    enabled: bool = True
    cross_broadcast_window: int = 2
    warmup_steps: int = 0
    active_start_step: int | None = None
    active_end_step: int | None = None
    detach_on_store: bool = True
    clone_on_hit: bool = False


@dataclass
class Cosmos3PABStats:
    calls: int = 0
    computes: int = 0
    hits: int = 0
    stores: int = 0
    disabled: int = 0
    skipped_steps: list[int] = field(default_factory=list)
    computed_steps: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        eligible = self.hits + self.computes
        return 0.0 if eligible == 0 else self.hits / eligible

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "computes": self.computes,
            "hits": self.hits,
            "stores": self.stores,
            "disabled": self.disabled,
            "hit_rate": round(self.hit_rate, 4),
            "skipped_steps": sorted(set(self.skipped_steps)),
            "computed_steps": sorted(set(self.computed_steps)),
        }


@dataclass(frozen=True)
class Cosmos3PABKey:
    cache_key: str
    block_idx: int
    shape: tuple[int, ...]
    dtype: str


@dataclass
class Cosmos3PABEntry:
    output: torch.Tensor
    source_step: int


class Cosmos3PABCoordinator:
    def __init__(self, config: Cosmos3PABConfig) -> None:
        self.config = config
        self._cache: dict[Cosmos3PABKey, Cosmos3PABEntry] = {}
        self._stats: dict[int, Cosmos3PABStats] = {}
        self._step: int = 0
        self._num_steps: int | None = None
        self._cache_key: str = "default"

    def reset(self) -> None:
        self._cache.clear()

    def reset_stats(self) -> None:
        self._stats.clear()

    def begin_step(
        self,
        *,
        step: int,
        num_inference_steps: int | None,
        cache_key: str,
    ) -> None:
        self._step = int(step)
        self._num_steps = num_inference_steps
        self._cache_key = str(cache_key)
        if self._step == 0:
            self._cache.clear()

    def stats_summary(self) -> dict[str, Any]:
        return {
            f"gen_layer_{idx}": stats.as_dict()
            for idx, stats in sorted(self._stats.items())
        }

    def forward_cross_attention(
        self,
        attention: nn.Module,
        block_idx: int,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        stats = self._stats.setdefault(int(block_idx), Cosmos3PABStats())
        stats.calls += 1

        if not self._eligible():
            stats.disabled += 1
            return attention(hidden_states, *args, **kwargs)

        key = Cosmos3PABKey(
            cache_key=self._cache_key,
            block_idx=int(block_idx),
            shape=tuple(hidden_states.shape),
            dtype=str(hidden_states.dtype),
        )
        entry = self._cache.get(key)
        if (
            entry is not None
            and self._step - entry.source_step < self.config.cross_broadcast_window
        ):
            stats.hits += 1
            stats.skipped_steps.append(self._step)
            return entry.output.clone() if self.config.clone_on_hit else entry.output

        stats.computes += 1
        stats.computed_steps.append(self._step)
        output = attention(hidden_states, *args, **kwargs)
        self._cache[key] = Cosmos3PABEntry(
            output=output.detach() if self.config.detach_on_store else output,
            source_step=self._step,
        )
        stats.stores += 1
        return output

    def _eligible(self) -> bool:
        if not self.config.enabled:
            return False
        if self._step < self.config.warmup_steps:
            return False
        if self.config.active_start_step is not None and self._step < self.config.active_start_step:
            return False
        if self.config.active_end_step is not None and self._step >= self.config.active_end_step:
            return False
        if self._num_steps is not None and self._step >= self._num_steps - 1:
            return False
        return self.config.cross_broadcast_window > 1


def cosmos3_pab_config_from_env() -> Cosmos3PABConfig:
    return Cosmos3PABConfig(
        enabled=_env_flag("SGLANG_COSMOS3_PAB_ENABLED", False),
        cross_broadcast_window=_env_int("SGLANG_COSMOS3_PAB_CROSS_WINDOW", 2),
        warmup_steps=_env_int("SGLANG_COSMOS3_PAB_WARMUP", 0),
        active_start_step=(
            None
            if os.environ.get("SGLANG_COSMOS3_PAB_START", "") == ""
            else _env_int("SGLANG_COSMOS3_PAB_START", 0)
        ),
        active_end_step=(
            None
            if os.environ.get("SGLANG_COSMOS3_PAB_END", "") == ""
            else _env_int("SGLANG_COSMOS3_PAB_END", -1)
        ),
        detach_on_store=_env_flag("SGLANG_COSMOS3_PAB_DETACH_ON_STORE", True),
        clone_on_hit=_env_flag("SGLANG_COSMOS3_PAB_CLONE_ON_HIT", False),
    )


def install_cosmos3_pab(transformer: nn.Module) -> bool:
    config = cosmos3_pab_config_from_env()
    if not config.enabled:
        return False
    if getattr(transformer, "_cosmos3_pab_installed", False):
        return True

    coordinator = Cosmos3PABCoordinator(config)
    setattr(transformer, "_cosmos3_pab", coordinator)

    gen_layers = getattr(transformer, "gen_layers", None)
    if gen_layers is None:
        logger.warning("Cosmos3 PAB requested but transformer.gen_layers is missing")
        return False

    for idx, layer in enumerate(gen_layers):
        attention = getattr(layer, "cross_attention", None)
        if attention is None:
            continue
        layer._cosmos3_pab_original_cross_attention = attention
        layer_idx = int(idx)

        class _CrossAttentionWrapper(nn.Module):
            def __init__(self, original: nn.Module, pab: Cosmos3PABCoordinator, block_idx: int):
                super().__init__()
                self.original = original
                self.pab = pab
                self.block_idx = block_idx

            def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                return self.pab.forward_cross_attention(
                    self.original,
                    self.block_idx,
                    hidden_states,
                    *args,
                    **kwargs,
                )

        layer.cross_attention = _CrossAttentionWrapper(attention, coordinator, layer_idx)

    setattr(transformer, "_cosmos3_pab_installed", True)
    logger.info(
        "Cosmos3 PAB installed on %d GEN layers (cross_window=%d)",
        len(gen_layers),
        config.cross_broadcast_window,
    )
    return True


__all__ = [
    "Cosmos3PABConfig",
    "Cosmos3PABCoordinator",
    "cosmos3_pab_config_from_env",
    "install_cosmos3_pab",
]
