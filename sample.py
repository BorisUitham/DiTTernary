# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_cache_config():
    try:
        from diffusers.utils.dit_cache import CacheConfig
        return CacheConfig
    except ModuleNotFoundError:
        cache_path = Path(__file__).resolve().parent / "src" / "diffusers" / "utils" / "dit_cache.py"
        if not cache_path.exists():
            raise
        spec = importlib.util.spec_from_file_location("diffusers.utils.dit_cache", cache_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        try:
            parent = importlib.import_module("diffusers.utils")
        except ModuleNotFoundError:
            parent = ModuleType("diffusers.utils")
            sys.modules["diffusers.utils"] = parent
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.CacheConfig


CacheConfig = _load_cache_config()


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_config = CacheConfig.from_flags(
        enable=args.cache_enable,
        level=args.cache_level,
        policy=args.cache_policy,
        delta=args.cache_delta,
        alpha=args.cache_alpha,
        cosine_threshold=args.cache_cosine_threshold,
        warmup_steps=args.cache_warmup_steps,
        kv_blend=args.cache_kv_blend,
        reset_on_shape_change=args.cache_reset_on_shape_change,
        cfg_share=args.cache_cfg_share,
    )

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        cache_config=cache_config,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--cache.enable", dest="cache_enable", type=str, default="false")
    parser.add_argument(
        "--cache.level",
        dest="cache_level",
        type=str,
        choices=["none", "block", "attn"],
        default="none",
    )
    parser.add_argument(
        "--cache.policy",
        dest="cache_policy",
        type=str,
        choices=["disabled"],
        default="disabled",
    )
    parser.add_argument("--cache.delta", dest="cache_delta", type=int, default=1)
    parser.add_argument("--cache.alpha", dest="cache_alpha", type=float, default=1.0)
    parser.add_argument("--cache.kv-blend", dest="cache_kv_blend", type=float, default=0.0)
    parser.add_argument(
        "--cache.cosine-threshold",
        dest="cache_cosine_threshold",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cache.warmup-steps",
        dest="cache_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cache.reset-on-shape-change",
        dest="cache_reset_on_shape_change",
        type=str,
        default="true",
    )
    parser.add_argument(
        "--cache.cfg-share",
        dest="cache_cfg_share",
        type=str,
        choices=["off", "kv", "attn"],
        default="off",
    )
    args = parser.parse_args()
    main(args)
