"""Quality regression metrics for DiT image generations.

The script evaluates a directory of images against a fixed prompt pack and
(optionally) a set of reference renders.  It reports CLIPScore statistics,
per-image hashes, and a lightweight perceptual distance computed from VGG16
features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torchvision import models
from torchvision.models import VGG16_Weights

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:  # pragma: no cover - informative error for CI
    raise SystemExit(
        "The transformers package is required to compute CLIPScore. "
        "Install it with `pip install transformers` before running this script."
    ) from exc


DEFAULT_PROMPTS: Tuple[str, ...] = (
    "a photo of a corgi wearing a bowtie",
    "sunset over a futuristic city skyline",
    "macro photograph of a dew-covered leaf",
    "astronaut riding a horse on Mars",
    "a serene mountain lake at dawn",
    "oil painting of a ship in stormy seas",
    "a watercolor of cherry blossoms",
    "architectural rendering of a glass pavilion",
    "portrait of a cyberpunk samurai",
    "a cozy cabin in a snowy forest",
    "abstract geometric shapes in vibrant colors",
    "close-up of a watch mechanism",
    "neon-lit alley in the rain",
    "ultra-wide shot of a desert canyon",
    "top-down photograph of a ramen bowl",
    "children playing in a park in spring",
    "retro 80s synthwave landscape",
    "black and white photograph of a dancer",
    "high-speed capture of splashing water",
    "illustration of a dragon curled around a tower",
    "studio photo of sneakers on a pedestal",
    "concept art of a floating island",
    "modern living room interior at golden hour",
    "aerial photo of a winding river",
    "moody portrait lit by neon",
    "surreal collage of clocks and flowers",
    "stormy ocean waves crashing on rocks",
    "macro of a butterfly wing",
    "galaxy swirling with bright nebulae",
    "chef plating gourmet dessert",
    "vintage car parked under palm trees",
    "impressionist painting of a bustling market",
)


@dataclass
class ImageRecord:
    path: Path
    prompt: str


@dataclass
class CLIPStats:
    scores: List[float]

    @property
    def mean(self) -> float:
        return float(sum(self.scores) / len(self.scores)) if self.scores else 0.0

    @property
    def median(self) -> float:
        if not self.scores:
            return 0.0
        ordered = sorted(self.scores)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 0:
            return float((ordered[mid - 1] + ordered[mid]) / 2.0)
        return float(ordered[mid])


class PerceptualMetric:
    """Lightweight perceptual distance using VGG16 conv features."""

    def __init__(self, device: torch.device) -> None:
        weights = VGG16_Weights.IMAGENET1K_FEATURES
        self.model = models.vgg16(weights=weights).features[:16].to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.preprocess = weights.transforms()
        self.device = device

    def distance(self, reference: Image.Image, candidate: Image.Image) -> float:
        with torch.no_grad():
            ref_tensor = self.preprocess(reference).unsqueeze(0).to(self.device)
            cand_tensor = self.preprocess(candidate).unsqueeze(0).to(self.device)
            ref_features = self.model(ref_tensor)
            cand_features = self.model(cand_tensor)
            return torch.nn.functional.mse_loss(cand_features, ref_features).item()


def _resolve_device(text: str) -> torch.device:
    cleaned = text.strip().lower()
    if cleaned in {"auto", "default"}:
        cleaned = "cuda" if torch.cuda.is_available() else "cpu"
    if cleaned == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    return torch.device(cleaned)


def _load_prompts(path: Optional[Path]) -> Sequence[str]:
    if path is None:
        return DEFAULT_PROMPTS
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != len(DEFAULT_PROMPTS):
        raise ValueError(
            f"Expected {len(DEFAULT_PROMPTS)} prompts but found {len(lines)} in {path}"
        )
    return lines


def _enumerate_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory {directory} does not exist")
    candidates = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not candidates:
        raise ValueError(f"No supported images found in {directory}")
    return candidates


def _zip_records(images: List[Path], prompts: Sequence[str]) -> List[ImageRecord]:
    if len(images) != len(prompts):
        raise ValueError(
            f"Expected {len(prompts)} images to match the prompt pack but found {len(images)}"
        )
    return [ImageRecord(path=path, prompt=prompt) for path, prompt in zip(images, prompts)]


def _hash_image(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_records(records: Sequence[ImageRecord], batch_size: int) -> Iterable[Sequence[ImageRecord]]:
    for index in range(0, len(records), batch_size):
        yield records[index : index + batch_size]


def _load_images(batch: Sequence[ImageRecord]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for record in batch:
        with Image.open(record.path) as handle:
            images.append(handle.convert("RGB"))
    return images


def compute_clip_scores(
    *,
    records: Sequence[ImageRecord],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    batch_size: int,
) -> CLIPStats:
    scores: List[float] = []
    for batch in _batch_records(records, batch_size):
        images = _load_images(batch)
        prompts = [record.prompt for record in batch]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        batch_scores = (image_features * text_features).sum(dim=-1)
        scores.extend(batch_scores.cpu().tolist())
    return CLIPStats(scores=scores)


def compute_perceptual_distances(
    *,
    records: Sequence[ImageRecord],
    reference_directory: Path,
    perceptual: PerceptualMetric,
) -> List[float]:
    distances: List[float] = []
    for record in records:
        ref_path = reference_directory / record.path.name
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image {ref_path} is missing")
        with Image.open(record.path) as cand_handle:
            candidate_image = cand_handle.convert("RGB")
        with Image.open(ref_path) as ref_handle:
            reference_image = ref_handle.convert("RGB")
        distances.append(perceptual.distance(reference_image, candidate_image))
    return distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DiT generations for regression drift")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing generated images")
    parser.add_argument("--reference", type=Path, help="Optional directory with baseline renders")
    parser.add_argument("--prompts", type=Path, help="Optional custom prompt pack (32 lines)")
    parser.add_argument("--output", type=Path, help="Path to write JSON metrics")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    prompts = _load_prompts(args.prompts)
    image_paths = _enumerate_images(args.images)
    records = _zip_records(image_paths, prompts)

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()

    clip_stats = compute_clip_scores(
        records=records,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    result: Dict[str, object] = {
        "clip": {
            "mean": clip_stats.mean,
            "median": clip_stats.median,
            "per_image": dict(zip([record.path.name for record in records], clip_stats.scores)),
        },
        "hashes": {
            "candidate": {record.path.name: _hash_image(record.path) for record in records}
        },
    }

    if args.reference is not None:
        reference = args.reference
        reference_paths = _enumerate_images(reference)
        if len(reference_paths) != len(records):
            raise ValueError(
                f"Reference set must contain {len(records)} images but found {len(reference_paths)}"
            )
        perceptual = PerceptualMetric(device=device)
        distances = compute_perceptual_distances(
            records=records,
            reference_directory=reference,
            perceptual=perceptual,
        )
        result["hashes"]["reference"] = {path.name: _hash_image(path) for path in reference_paths}
        result["perceptual"] = {
            "mean": float(sum(distances) / len(distances)) if distances else 0.0,
            "per_image": dict(zip([record.path.name for record in records], distances)),
        }

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
