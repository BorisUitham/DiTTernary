"""Minimal DiT pipeline extension that threads cache configuration."""
from __future__ import annotations

from typing import Any, Optional


class DiTPipeline:
    """Wrapper that forwards cache-related arguments to the transformer."""

    def __init__(self, transformer: Any):
        self.transformer = transformer

    def __call__(
        self,
        *args: Any,
        cache_config: Optional[Any] = None,
        feature_cache: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        if cache_config is not None and "cache_config" not in kwargs:
            kwargs["cache_config"] = cache_config
        if feature_cache is not None and "feature_cache" not in kwargs:
            kwargs["feature_cache"] = feature_cache
        return self.transformer(*args, **kwargs)


__all__ = ["DiTPipeline"]
