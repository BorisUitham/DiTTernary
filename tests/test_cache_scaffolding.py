import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

# Ensure the project root (which contains `models.py`) is importable when the
# test file is executed directly, e.g. ``python tests/test_cache_scaffolding.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DiT


def _load_module(name: str, relative_path: Path):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(name, relative_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        parent_name = name.rsplit(".", 1)[0]
        if parent_name not in sys.modules:
            try:
                parent_module = importlib.import_module(parent_name)
            except ModuleNotFoundError:
                parent_module = ModuleType(parent_name)
                sys.modules[parent_name] = parent_module
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


def _root_path() -> Path:
    return PROJECT_ROOT


def load_cache_components():
    module_path = _root_path() / "src" / "diffusers" / "utils" / "dit_cache.py"
    module = _load_module("diffusers.utils.dit_cache", module_path)
    return (
        module,
        module.CacheConfig,
        module.CacheLevel,
        module.CachePolicy,
        module.FeatureCache,
    )


def load_pipeline():
    module_path = _root_path() / "src" / "diffusers" / "pipelines" / "dit" / "pipeline_dit.py"
    return _load_module("diffusers.pipelines.dit.pipeline_dit", module_path)


(
    CacheModule,
    CacheConfig,
    CacheLevel,
    CachePolicy,
    FeatureCache,
) = load_cache_components()
CFGShareMode = CacheModule.CFGShareMode
CFGBranch = CacheModule.CFGBranch


def make_model(cache_config=None):
    kwargs = dict(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        class_dropout_prob=0.0,
        num_classes=10,
        learn_sigma=False,
        cache_config=cache_config,
    )
    return DiT(**kwargs)


def test_model_initialization_equivalence():
    torch.manual_seed(0)
    model_plain = make_model()
    torch.manual_seed(0)
    model_cached = make_model(CacheConfig(enable=False, level=CacheLevel.NONE, policy=CachePolicy.DISABLED))
    plain_state = dict(model_plain.state_dict())
    cached_state = dict(model_cached.state_dict())
    assert plain_state.keys() == cached_state.keys()
    for key in plain_state:
        assert torch.equal(plain_state[key], cached_state[key])


def test_forward_equivalence_when_cache_disabled():
    torch.manual_seed(0)
    model_plain = make_model()
    torch.manual_seed(0)
    model_cached = make_model(CacheConfig(enable=False, level=CacheLevel.NONE, policy=CachePolicy.DISABLED))
    model_plain.eval()
    model_cached.eval()
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        out_plain = model_plain(x, t, y)
        out_cached = model_cached(x, t, y)
    assert torch.allclose(out_plain, out_cached)


def test_pipeline_threads_cache_config():
    module = load_pipeline()
    cfg = CacheConfig(enable=True, level=CacheLevel.BLOCK, policy=CachePolicy.DISABLED)

    class DummyTransformer:
        def __call__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return kwargs

    transformer = DummyTransformer()
    pipeline = module.DiTPipeline(transformer)
    result = pipeline(1, 2, cache_config=cfg)
    assert transformer.kwargs.get("cache_config") is cfg
    assert result is transformer.kwargs


def test_cache_config_cfg_share_flag():
    cfg = CacheConfig.from_flags(
        enable=True,
        level=CacheLevel.ATTN,
        policy=CachePolicy.DISABLED,
        cfg_share="kv",
    )
    assert cfg.cfg_share is CFGShareMode.KV


def _synchronize_models(cache_config):
    torch.manual_seed(0)
    base = make_model()
    torch.manual_seed(0)
    cached = make_model(cache_config)
    cached.load_state_dict(base.state_dict())
    return base, cached


def test_block_cache_alpha_one_preserves_outputs():
    cfg = CacheConfig(
        enable=True,
        level=CacheLevel.BLOCK,
        policy=CachePolicy.DISABLED,
        alpha=1.0,
        cosine_threshold=-1.0,
    )
    model_plain, model_cached = _synchronize_models(cfg)
    feature_cache = FeatureCache(cfg)
    model_plain.eval()
    model_cached.eval()
    x0 = torch.randn(2, 4, 8, 8)
    t0 = torch.full((2,), 5)
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        out_plain_0 = model_plain(x0, t0, y)
        out_cached_0 = model_cached(x0, t0, y, cache_config=cfg, feature_cache=feature_cache)
    assert torch.allclose(out_plain_0, out_cached_0)
    x1 = torch.randn(2, 4, 8, 8)
    t1 = torch.full((2,), 4)
    with torch.no_grad():
        out_plain_1 = model_plain(x1, t1, y)
        out_cached_1 = model_cached(x1, t1, y, cache_config=cfg, feature_cache=feature_cache)
    assert torch.allclose(out_plain_1, out_cached_1)
    assert feature_cache.metrics.cache_hits >= 1


def test_block_cache_threshold_bypass_matches_baseline():
    cfg = CacheConfig(
        enable=True,
        level=CacheLevel.BLOCK,
        policy=CachePolicy.DISABLED,
        alpha=0.5,
        cosine_threshold=2.0,
    )
    model_plain, model_cached = _synchronize_models(cfg)
    feature_cache = FeatureCache(cfg)
    model_plain.eval()
    model_cached.eval()
    x0 = torch.randn(2, 4, 8, 8)
    t0 = torch.full((2,), 3)
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        _ = model_plain(x0, t0, y)
        _ = model_cached(x0, t0, y, cache_config=cfg, feature_cache=feature_cache)
    x1 = torch.randn(2, 4, 8, 8)
    t1 = torch.full((2,), 2)
    with torch.no_grad():
        out_plain = model_plain(x1, t1, y)
        out_cached = model_cached(x1, t1, y, cache_config=cfg, feature_cache=feature_cache)
    assert torch.allclose(out_plain, out_cached)
    assert feature_cache.metrics.cache_hits == 0


def test_feature_cache_cfg_kv_roundtrip():
    cfg = CacheConfig(
        enable=True,
        level=CacheLevel.ATTN,
        policy=CachePolicy.DISABLED,
        cfg_share=CFGShareMode.KV,
    )
    cache = FeatureCache(cfg)
    cache.on_cfg_batch_start()
    cache.on_cfg_branch(CFGBranch.UNCOND)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn_like(key)
    _ = cache.prepare_cfg_attention(0, device=key.device, dtype=key.dtype)
    cache.store_cfg_attention(0, key, value)
    cache.on_cfg_branch(CFGBranch.COND)
    state = cache.prepare_cfg_attention(0, device=key.device, dtype=key.dtype)
    assert state is not None
    assert state.use_shared_kv()


def test_feature_cache_cfg_attn_context_reuse():
    cfg = CacheConfig(
        enable=True,
        level=CacheLevel.ATTN,
        policy=CachePolicy.DISABLED,
        cfg_share=CFGShareMode.ATTN,
    )
    cache = FeatureCache(cfg)
    cache.on_cfg_batch_start()
    cache.on_cfg_branch(CFGBranch.UNCOND)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn_like(key)
    context = torch.randn(1, 6, 8)
    _ = cache.prepare_cfg_attention(1, device=key.device, dtype=key.dtype)
    cache.store_cfg_attention(1, key, value, context=context)
    cache.on_cfg_branch(CFGBranch.COND)
    state = cache.prepare_cfg_attention(1, device=key.device, dtype=key.dtype)
    assert state is not None
    assert state.use_shared_context()
    assert torch.allclose(state.context, context)
