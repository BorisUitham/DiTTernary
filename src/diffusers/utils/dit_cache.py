"""Utilities for DiT feature caching scaffolding.

This module provides lightweight cache primitives that allow models and
pipelines to share a common configuration object without committing to a
particular caching strategy yet. The implementation intentionally keeps the
behavior passive – unless caching is explicitly enabled the helpers are
essentially no-ops, which makes them safe to wire into existing code paths
without altering outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, MutableMapping, Optional


class CacheLevel(str, Enum):
    """Granularity levels that a cache implementation can target."""

    NONE = "none"
    BLOCK = "block"
    ATTN = "attn"

    @classmethod
    def from_value(cls, value: Optional[str]) -> "CacheLevel":
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        return cls(str(value))


class CachePolicy(str, Enum):
    """Policy names for cache eviction or bypass strategies."""

    DISABLED = "disabled"

    @classmethod
    def from_value(cls, value: Optional[str]) -> "CachePolicy":
        if value is None:
            return cls.DISABLED
        if isinstance(value, cls):
            return value
        return cls(str(value))


@dataclass
class CacheConfig:
    """Configuration describing how caching should behave."""

    enable: bool = False
    level: CacheLevel = CacheLevel.NONE
    policy: CachePolicy = CachePolicy.DISABLED

    @classmethod
    def from_flags(
        cls,
        enable: Any = False,
        level: Any = CacheLevel.NONE,
        policy: Any = CachePolicy.DISABLED,
    ) -> "CacheConfig":
        """Create a :class:`CacheConfig` from flag-style inputs."""

        def _to_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y"}:
                return True
            if text in {"0", "false", "no", "n", ""}:
                return False
            raise ValueError(f"Cannot convert {value!r} to bool")

        return cls(
            enable=_to_bool(enable),
            level=CacheLevel.from_value(level),
            policy=CachePolicy.from_value(policy),
        )

    @property
    def active(self) -> bool:
        """Return ``True`` if caching should be active."""

        return bool(self.enable and self.level is not CacheLevel.NONE and self.policy is not CachePolicy.DISABLED)


@dataclass
class CacheMetrics:
    """Bookkeeping metadata collected by :class:`FeatureCache`."""

    block_inputs: int = 0
    block_outputs: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cached_blocks: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "block_inputs": self.block_inputs,
            "block_outputs": self.block_outputs,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cached_blocks": self.cached_blocks,
        }


@dataclass
class FeatureCache:
    """In-memory cache container for transformer features."""

    config: CacheConfig = field(default_factory=CacheConfig)
    store: MutableMapping[str, Any] = field(default_factory=dict)
    metrics: CacheMetrics = field(default_factory=CacheMetrics)

    def reset(self) -> None:
        """Drop all cached values and metrics."""

        self.store.clear()
        self.metrics = CacheMetrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _block_key(self, block_index: int) -> str:
        return f"block:{block_index}"

    def _block_cache_active(self) -> bool:
        return self.config.active and self.config.level in {CacheLevel.BLOCK, CacheLevel.ATTN}

    # ------------------------------------------------------------------
    # Public hooks
    # ------------------------------------------------------------------
    def on_block_input(self, block_index: int, hidden_states: Any) -> Any:
        """Hook invoked before a block consumes its inputs."""

        if not self._block_cache_active():
            return hidden_states
        self.metrics.block_inputs += 1
        key = self._block_key(block_index)
        if key in self.store:
            self.metrics.cache_hits += 1
            return self.store[key]
        self.metrics.cache_misses += 1
        return hidden_states

    def on_block_output(self, block_index: int, hidden_states: Any) -> Any:
        """Hook invoked after a block produces its outputs."""

        if not self._block_cache_active():
            return hidden_states
        self.metrics.block_outputs += 1
        key = self._block_key(block_index)
        self.store[key] = hidden_states
        self.metrics.cached_blocks = len(self.store)
        return hidden_states


__all__ = [
    "CacheConfig",
    "CacheLevel",
    "CacheMetrics",
    "CachePolicy",
    "FeatureCache",
]
