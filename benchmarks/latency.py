"""Latency microbenchmark harness for DiT models.

This script runs a light-weight DiT forward benchmark with optional cache
settings and reports detailed per-module timings.  It is intentionally small so
that it can execute in CI environments while still emitting telemetry that helps
track performance regressions.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from models import DiT_models

try:  # Optional dependency – only required when caching is enabled.
    from diffusers.utils.dit_cache import CacheConfig, FeatureCache
except ModuleNotFoundError:  # pragma: no cover - fallback for editable installs
    import importlib.util
    import sys

    _CACHE_PATH = Path(__file__).resolve().parents[1] / "src" / "diffusers" / "utils" / "dit_cache.py"
    if not _CACHE_PATH.exists():
        raise
    spec = importlib.util.spec_from_file_location("diffusers.utils.dit_cache", _CACHE_PATH)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    CacheConfig = module.CacheConfig  # type: ignore[attr-defined]
    FeatureCache = module.FeatureCache  # type: ignore[attr-defined]


@dataclass
class BenchmarkInputs:
    """Container describing a batch of latent inputs for benchmarking."""

    latents: torch.Tensor
    timesteps: Sequence[int]
    labels: torch.Tensor


class ModuleProfiler:
    """Utility that attaches forward hooks to modules and records timings."""

    def __init__(self, use_cuda: bool = False) -> None:
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.records: Dict[str, List[float]] = defaultdict(list)
        self._stack: List[Tuple[Optional[str], Optional[object]]] = []
        self._enabled: bool = True

    # ------------------------------------------------------------------
    # Hook plumbing
    # ------------------------------------------------------------------
    def attach(self, module: torch.nn.Module, name: str) -> Iterable[torch.utils.hooks.RemovableHandle]:
        """Attach profiling hooks to ``module`` using ``name`` for reporting."""

        def _pre_hook(_module: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...]) -> None:
            self._start(name)

        def _post_hook(
            _module: torch.nn.Module,
            _inputs: Tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            self._stop(name)

        return (
            module.register_forward_pre_hook(_pre_hook),
            module.register_forward_hook(_post_hook),
        )

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _start(self, name: str) -> None:
        if not self._enabled:
            self._stack.append((None, None))
            return
        if self.use_cuda:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(name)
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self._stack.append((name, start_event))
        else:
            self._stack.append((name, time.perf_counter()))

    def _stop(self, name: str) -> None:
        if not self._stack:
            return
        stack_name, token = self._stack.pop()
        if stack_name is None:
            return
        if stack_name != name:
            # Nested modules may share hooks; keep bookkeeping robust.
            name = stack_name
        if self.use_cuda:
            assert isinstance(token, torch.cuda.Event)
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            end_event.synchronize()
            torch.cuda.nvtx.range_pop()
            elapsed_ms = token.elapsed_time(end_event)
            duration = float(elapsed_ms) / 1000.0
        else:
            assert isinstance(token, (int, float)) or hasattr(token, "__float__")
            duration = float(time.perf_counter() - token)  # type: ignore[arg-type]
        self.records[name].append(duration)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for name, durations in sorted(self.records.items()):
            if not durations:
                continue
            mean = statistics.fmean(durations)
            stats[name] = {
                "count": float(len(durations)),
                "total_seconds": float(sum(durations)),
                "mean_seconds": float(mean),
                "p50_seconds": float(statistics.median(durations)),
                "p95_seconds": float(_percentile(durations, 0.95)),
            }
        return stats


def _percentile(data: Sequence[float], q: float) -> float:
    if not data:
        return 0.0
    ordered = sorted(data)
    index = min(len(ordered) - 1, max(0, int(math.ceil(len(ordered) * q)) - 1))
    return float(ordered[index])


def _resolve_device(device: str) -> torch.device:
    text = device.strip().lower()
    if text in {"auto", "default"}:
        text = "cuda" if torch.cuda.is_available() else "cpu"
    if text == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    return torch.device(text)


def _prepare_inputs(
    *,
    batch_size: int,
    image_size: int,
    num_classes: int,
    device: torch.device,
    num_steps: int,
) -> BenchmarkInputs:
    latent_size = image_size // 8
    latents = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
    timesteps = torch.linspace(0, 999, steps=num_steps, dtype=torch.long, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return BenchmarkInputs(latents=latents, timesteps=timesteps.tolist(), labels=labels)


def _load_model(
    model_name: str,
    *,
    image_size: int,
    num_classes: int,
    device: torch.device,
    cache_config: CacheConfig,
    checkpoint: Optional[Path] = None,
    random_weights: bool = False,
) -> torch.nn.Module:
    latent_size = image_size // 8
    model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        cache_config=cache_config,
    )
    if not random_weights:
        state_dict = None
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        else:
            from download import find_model  # Lazy import to avoid download when unused

            state_dict = find_model(f"{model_name.replace('/', '-')}-{image_size}x{image_size}.pt")
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def _register_profilers(model: torch.nn.Module, profiler: ModuleProfiler) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    handles.extend(profiler.attach(model, "model"))
    if hasattr(model, "x_embedder"):
        handles.extend(profiler.attach(model.x_embedder, "x_embedder"))
    if hasattr(model, "t_embedder"):
        handles.extend(profiler.attach(model.t_embedder, "t_embedder"))
    if hasattr(model, "y_embedder"):
        handles.extend(profiler.attach(model.y_embedder, "y_embedder"))
    if hasattr(model, "blocks"):
        for idx, block in enumerate(model.blocks):
            handles.extend(profiler.attach(block, f"block_{idx:02d}"))
            if hasattr(block, "attn"):
                handles.extend(profiler.attach(block.attn, f"block_{idx:02d}.attn"))
            if hasattr(block, "mlp"):
                handles.extend(profiler.attach(block.mlp, f"block_{idx:02d}.mlp"))
    if hasattr(model, "final_layer"):
        handles.extend(profiler.attach(model.final_layer, "final_layer"))
    return handles


def run_benchmark(
    *,
    model_name: str,
    image_size: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    num_steps: int,
    warmup: int,
    iterations: int,
    cache_config: CacheConfig,
    checkpoint: Optional[Path] = None,
    random_weights: bool = False,
) -> Dict[str, object]:
    inputs = _prepare_inputs(
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        device=device,
        num_steps=num_steps,
    )

    model = _load_model(
        model_name,
        image_size=image_size,
        num_classes=num_classes,
        device=device,
        cache_config=cache_config,
        checkpoint=checkpoint,
        random_weights=random_weights,
    )
    model.eval()

    profiler = ModuleProfiler(use_cuda=device.type == "cuda")
    handles = _register_profilers(model, profiler)

    results: Dict[str, object] = {
        "device": str(device),
        "model": model_name,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "cache": _cache_to_dict(cache_config),
        "random_weights": random_weights,
        "iterations": iterations,
        "warmup": warmup,
        "timings": {},
        "totals": [],
    }

    cache_metrics: List[Dict[str, object]] = []

    def _iteration(record: bool, iteration_id: int) -> None:
        feature_cache = FeatureCache(cache_config) if cache_config.active else None
        diffusion_timesteps = list(reversed(inputs.timesteps))
        latents = inputs.latents
        labels = inputs.labels
        for step_index, timestep in enumerate(diffusion_timesteps):
            if feature_cache is not None:
                feature_cache.on_step_start(timestep)
            timestep_tensor = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(
                    latents,
                    timestep_tensor,
                    labels,
                    cache_config=cache_config,
                    feature_cache=feature_cache,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if feature_cache is not None and record:
                feature_cache.record_latency(elapsed)
            if record:
                results.setdefault("per_step", []).append(
                    {
                        "iteration": iteration_id,
                        "step_index": step_index,
                        "timestep": timestep,
                        "seconds": elapsed,
                    }
                )
        if feature_cache is not None and record:
            cache_metrics.append(feature_cache.metrics.as_dict())

    profiler.set_enabled(False)
    for _ in range(warmup):
        _iteration(record=False, iteration_id=-1)
    profiler.set_enabled(True)

    totals: List[float] = []
    for iteration_id in range(iterations):
        start = time.perf_counter()
        _iteration(record=True, iteration_id=iteration_id)
        totals.append(time.perf_counter() - start)

    profiler_summary = profiler.summary()
    results["timings"] = profiler_summary
    results["totals"] = {
        "mean_seconds": statistics.fmean(totals) if totals else 0.0,
        "p50_seconds": statistics.median(totals) if totals else 0.0,
        "p95_seconds": _percentile(totals, 0.95) if totals else 0.0,
        "samples": totals,
    }
    if cache_metrics:
        results["cache_metrics"] = cache_metrics
    for handle in handles:
        handle.remove()
    return results


def _cache_to_dict(config: CacheConfig) -> Dict[str, object]:
    data: Dict[str, object] = dict(asdict(config))
    for key, value in list(data.items()):
        if hasattr(value, "value"):
            data[key] = getattr(value, "value")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DiT latency microbenchmark")
    parser.add_argument("--model", default="DiT-XL/2", help="DiT model variant to benchmark")
    parser.add_argument("--image-size", type=int, default=256, help="Rendered image size")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=4, help="Number of diffusion steps to profile")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before timing")
    parser.add_argument("--iterations", type=int, default=2, help="Recorded iterations")
    parser.add_argument("--device", default="auto", help="Device to run on (cpu, cuda, or auto)")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON results", default=None)
    parser.add_argument("--checkpoint", type=Path, help="Optional path to model checkpoint")
    parser.add_argument("--random-weights", action="store_true", help="Skip loading checkpoints and use random weights")
    parser.add_argument("--cache.enable", dest="cache_enable", default="false")
    parser.add_argument("--cache.level", dest="cache_level", default="none")
    parser.add_argument("--cache.policy", dest="cache_policy", default="disabled")
    parser.add_argument("--cache.delta", dest="cache_delta", default=1)
    parser.add_argument("--cache.alpha", dest="cache_alpha", default=1.0)
    parser.add_argument("--cache.cosine-threshold", dest="cache_cosine_threshold", default=1.0)
    parser.add_argument("--cache.warmup-steps", dest="cache_warmup_steps", default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(str(args.device))
    cache_config = CacheConfig.from_flags(
        enable=args.cache_enable,
        level=args.cache_level,
        policy=args.cache_policy,
        delta=args.cache_delta,
        alpha=args.cache_alpha,
        cosine_threshold=args.cache_cosine_threshold,
        warmup_steps=args.cache_warmup_steps,
    )
    results = run_benchmark(
        model_name=args.model,
        image_size=args.image_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        device=device,
        num_steps=args.num_steps,
        warmup=args.warmup,
        iterations=args.iterations,
        cache_config=cache_config,
        checkpoint=args.checkpoint,
        random_weights=bool(args.random_weights),
    )
    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
