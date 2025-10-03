"""Utilities for DiT feature caching scaffolding.

This module provides lightweight cache primitives that allow models and
pipelines to share a common configuration object while progressively layering
in richer cache behaviours.  The implementation intentionally keeps the
behavior passive – unless caching is explicitly enabled the helpers are
essentially no-ops, which makes them safe to wire into existing code paths
without altering outputs when disabled.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Deque, Dict, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)


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
    delta: int = 1
    alpha: float = 1.0
    cosine_threshold: float = 1.0
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if self.delta < 0:
            raise ValueError("cache delta must be non-negative")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("cache alpha must lie in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("cache warmup_steps must be non-negative")

    @classmethod
    def from_flags(
        cls,
        enable: Any = False,
        level: Any = CacheLevel.NONE,
        policy: Any = CachePolicy.DISABLED,
        delta: Any = 1,
        alpha: Any = 1.0,
        cosine_threshold: Any = 1.0,
        warmup_steps: Any = 0,
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

        def _to_int(value: Any, *, allow_none: bool = False) -> int:
            if value is None:
                if allow_none:
                    return 0
                raise ValueError("Integer flag may not be None")
            if isinstance(value, int):
                return value
            return int(str(value).strip())

        def _to_float(value: Any) -> float:
            if value is None:
                return 0.0
            if isinstance(value, float):
                return value
            if isinstance(value, int):
                return float(value)
            return float(str(value).strip())

        return cls(
            enable=_to_bool(enable),
            level=CacheLevel.from_value(level),
            policy=CachePolicy.from_value(policy),
            delta=max(0, _to_int(delta, allow_none=True)),
            alpha=_to_float(alpha),
            cosine_threshold=_to_float(cosine_threshold),
            warmup_steps=max(0, _to_int(warmup_steps, allow_none=True)),
        )

    @property
    def active(self) -> bool:
        """Return ``True`` if caching should be active."""

        return bool(
            self.enable
            and self.level is not CacheLevel.NONE
        )


@dataclass
class BlockTelemetry:
    """Per-block telemetry tracking acceptance rates and cosine stats."""

    attempts: int = 0
    accepts: int = 0
    cosine_sum: float = 0.0
    cosine_count: int = 0

    def record(self, cosine: Optional[float], accepted: bool) -> None:
        self.attempts += 1
        if cosine is not None:
            self.cosine_sum += float(cosine)
            self.cosine_count += 1
        if accepted:
            self.accepts += 1

    @property
    def acceptance_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.accepts / self.attempts

    @property
    def average_cosine(self) -> float:
        if self.cosine_count == 0:
            return 0.0
        return self.cosine_sum / self.cosine_count

    def as_dict(self) -> Dict[str, float]:
        return {
            "attempts": self.attempts,
            "acceptance_rate": self.acceptance_rate,
            "average_cosine": self.average_cosine,
        }


@dataclass
class CacheMetrics:
    """Bookkeeping metadata collected by :class:`FeatureCache`."""

    block_inputs: int = 0
    block_outputs: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cached_blocks: int = 0
    block_telemetry: Dict[int, BlockTelemetry] = field(default_factory=dict)
    latency_samples: Deque[float] = field(default_factory=deque)

    def record_block_telemetry(self, block_index: int, cosine: Optional[float], accepted: bool) -> None:
        telemetry = self.block_telemetry.setdefault(block_index, BlockTelemetry())
        telemetry.record(cosine, accepted)

    def record_latency(self, seconds: float) -> None:
        self.latency_samples.append(float(seconds))

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "block_inputs": self.block_inputs,
            "block_outputs": self.block_outputs,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cached_blocks": self.cached_blocks,
        }
        if self.block_telemetry:
            data["block_telemetry"] = {
                str(index): telemetry.as_dict()
                for index, telemetry in self.block_telemetry.items()
            }
        if self.latency_samples:
            count = len(self.latency_samples)
            average = sum(self.latency_samples) / count
            data["latency"] = {"count": count, "average_seconds": average}
        return data


@dataclass
class BlockCacheContext:
    """Context container passed to blocks while caching is active."""

    cache: "FeatureCache"
    block_index: int
    target_iteration: Optional[int]
    cached: Optional[Any]
    accepted: bool = False
    cosine: Optional[float] = None
    processed: bool = False

    def blend(self, hidden_states: Any) -> Any:
        self.processed = True
        return self.cache._blend_block_output(self, hidden_states)


@dataclass
class FeatureCache:
    """In-memory cache container for transformer features."""

    config: CacheConfig = field(default_factory=CacheConfig)
    store: MutableMapping[str, Deque[Tuple[int, Any]]] = field(default_factory=dict)
    metrics: CacheMetrics = field(default_factory=CacheMetrics)
    _iteration: int = 0
    _current_iteration: Optional[int] = None
    _current_timestep: Optional[int] = None

    def reset(self) -> None:
        """Drop all cached values and metrics."""

        self.store.clear()
        self.metrics = CacheMetrics()
        self._iteration = 0
        self._current_iteration = None
        self._current_timestep = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _block_key(self, block_index: int) -> str:
        return f"block:{block_index}"

    def _block_cache_active(self) -> bool:
        return self.config.active and self.config.level in {CacheLevel.BLOCK, CacheLevel.ATTN}

    def _history_length(self) -> int:
        return max(1, int(self.config.delta) + 1)

    def _get_block_store(self, block_index: int) -> Deque[Tuple[int, Any]]:
        key = self._block_key(block_index)
        if key not in self.store:
            self.store[key] = deque(maxlen=self._history_length())
        return self.store[key]

    def _retrieve_cached(self, block_index: int, target_iteration: Optional[int]) -> Optional[Any]:
        if target_iteration is None or target_iteration <= 0:
            return None
        key = self._block_key(block_index)
        block_store = self.store.get(key)
        if not block_store:
            return None
        for iteration, value in reversed(block_store):
            if iteration == target_iteration:
                return value
        return None

    def _blend_block_output(self, context: BlockCacheContext, hidden_states: Any) -> Any:
        if not self._block_cache_active():
            return hidden_states
        cached = context.cached
        block_index = context.block_index
        if cached is None:
            self.metrics.cache_misses += 1
            self.metrics.record_block_telemetry(block_index, None, False)
            context.accepted = False
            context.cosine = None
            return hidden_states
        try:
            import torch
            import torch.nn.functional as F
        except ModuleNotFoundError as exc:  # pragma: no cover - torch is required when caching is active
            raise RuntimeError("FeatureCache requires torch to blend hidden states") from exc

        candidate = cached
        if candidate.device != hidden_states.device:
            candidate = candidate.to(hidden_states.device)
        if candidate.dtype != hidden_states.dtype:
            candidate = candidate.to(hidden_states.dtype)

        current_flat = hidden_states.reshape(hidden_states.shape[0], -1)
        cached_flat = candidate.reshape(candidate.shape[0], -1)
        cosine_per_sample = F.cosine_similarity(current_flat, cached_flat, dim=1, eps=1e-6)
        cosine_value = float(cosine_per_sample.mean().detach().item())
        threshold = float(self.config.cosine_threshold)
        accepted = cosine_value >= threshold
        context.cosine = cosine_value
        context.accepted = accepted
        self.metrics.record_block_telemetry(block_index, cosine_value, accepted)
        logger.debug(
            "cache block %s iteration=%s cosine=%.4f threshold=%.4f accepted=%s",
            block_index,
            self._current_iteration,
            cosine_value,
            threshold,
            accepted,
        )
        if not accepted:
            self.metrics.cache_misses += 1
            return hidden_states

        self.metrics.cache_hits += 1
        alpha = float(self.config.alpha)
        if alpha >= 1.0:
            return hidden_states
        blended = hidden_states.mul(alpha).add(candidate, alpha=1 - alpha)
        return blended

    def _record_cached_block(self, block_index: int, hidden_states: Any) -> None:
        if not self._block_cache_active():
            return
        store = self._get_block_store(block_index)
        iteration = self._current_iteration
        if iteration is None:
            return
        store.append((iteration, hidden_states.detach()))
        self.metrics.cached_blocks = sum(1 for values in self.store.values() if values)

    # ------------------------------------------------------------------
    # Public hooks
    # ------------------------------------------------------------------
    def on_step_start(self, timestep: Optional[int] = None) -> None:
        if not self._block_cache_active():
            return
        self._iteration += 1
        self._current_iteration = self._iteration
        self._current_timestep = timestep

    def record_latency(self, seconds: float) -> None:
        if not self._block_cache_active():
            return
        self.metrics.record_latency(seconds)
        logger.info("cache latency %.3fms", seconds * 1000.0)

    def on_block_input(self, block_index: int, hidden_states: Any) -> Tuple[Any, Optional[BlockCacheContext]]:
        """Hook invoked before a block consumes its inputs."""

        if not self._block_cache_active():
            return hidden_states, None
        self.metrics.block_inputs += 1
        if self._current_iteration is None or self._current_iteration <= self.config.warmup_steps:
            context = BlockCacheContext(
                cache=self,
                block_index=block_index,
                target_iteration=None,
                cached=None,
            )
            return hidden_states, context
        target_iteration = self._current_iteration - int(self.config.delta)
        cached = self._retrieve_cached(block_index, target_iteration)
        context = BlockCacheContext(
            cache=self,
            block_index=block_index,
            target_iteration=target_iteration,
            cached=cached,
        )
        return hidden_states, context

    def on_block_output(
        self,
        block_index: int,
        hidden_states: Any,
        cache_context: Optional[BlockCacheContext] = None,
    ) -> Any:
        """Hook invoked after a block produces its outputs."""

        if not self._block_cache_active():
            return hidden_states
        self.metrics.block_outputs += 1
        result = hidden_states
        if cache_context is not None:
            if not cache_context.processed:
                result = cache_context.blend(hidden_states)
        else:
            # No context implies the block was not able to retrieve cached features.
            self.metrics.cache_misses += 1
        self._record_cached_block(block_index, result)
        return result

    def log_summary(self) -> None:
        """Log a summary of cache telemetry for debugging purposes."""

        if not self.metrics.block_telemetry:
            return
        for block_index, telemetry in sorted(self.metrics.block_telemetry.items()):
            logger.info(
                "cache block %s acceptance_rate=%.3f avg_cosine=%.4f attempts=%d",
                block_index,
                telemetry.acceptance_rate,
                telemetry.average_cosine,
                telemetry.attempts,
            )
        if self.metrics.latency_samples:
            count = len(self.metrics.latency_samples)
            average = sum(self.metrics.latency_samples) / count
            logger.info("cache latency avg=%.3fms samples=%d", average * 1000.0, count)


__all__ = [
    "CacheConfig",
    "CacheLevel",
    "CacheMetrics",
    "CachePolicy",
    "BlockTelemetry",
    "FeatureCache",
]
