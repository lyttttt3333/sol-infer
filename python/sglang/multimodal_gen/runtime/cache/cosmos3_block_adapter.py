# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 BlockAdapter registration for cache-dit."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from cache_dit import BlockAdapter, ForwardPattern
    from cache_dit.caching.block_adapters import BlockAdapterRegister
except ImportError:  # pragma: no cover - cache-dit is optional
    logger.debug("cache-dit not installed; Cosmos3 BlockAdapter registration skipped.")
    BlockAdapter = None  # type: ignore[assignment]
    ForwardPattern = None  # type: ignore[assignment]
    BlockAdapterRegister = None  # type: ignore[assignment]


def _build_cosmos3_adapter(pipe, **kwargs):
    transformer = pipe.transformer
    return BlockAdapter(
        pipe=pipe,
        transformer=transformer,
        blocks=transformer.gen_layers,
        forward_pattern=ForwardPattern.Pattern_0,
        check_forward_pattern=False,
        **kwargs,
    )


if BlockAdapterRegister is not None:
    BlockAdapterRegister.register("Cosmos3")(_build_cosmos3_adapter)
    BlockAdapterRegister.register("FSDPCosmos3")(_build_cosmos3_adapter)
    logger.debug("Registered Cosmos3 BlockAdapter for cache-dit GEN layers.")


__all__ = ["_build_cosmos3_adapter"]
