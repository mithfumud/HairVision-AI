"""
Visual segmentation diagnostics for HairSegmenter.

Not a unit test — run manually to inspect mask quality:

    python tests/test_segmentation.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running as ``python tests/test_segmentation.py`` from HairVision-AI/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.segmentation import HairSegmenter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEST_DIRS = (
    ROOT / "test_images" / "front",
    ROOT / "test_images" / "crown",
)
OUTPUT_ROOT = ROOT / "outputs" / "segmentation"

HAIR_OVERLAY_COLOR = (40, 180, 60)
HAIR_OVERLAY_ALPHA = 0.45
SKIN_OVERLAY_COLOR = (40, 100, 255)
SKIN_OVERLAY_ALPHA = 0.40
HEAD_OUTLINE_COLOR = (255, 0, 0)


def load_images() -> list[Path]:
    """Collect front/crown test images with supported extensions."""
    images: list[Path] = []
    for folder in TEST_DIRS:
        if not folder.is_dir():
            print(f"Warning: missing directory {folder}")
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
    return images


def save_binary_mask(mask: np.ndarray, destination: Path) -> None:
    """Save a boolean/0-1 mask as an 8-bit PNG (0 or 255)."""
    binary = np.asarray(mask).astype(bool)
    pixels = (binary.astype(np.uint8) * 255)
    Image.fromarray(pixels, mode="L").save(destination)


def _blend_mask(
    base: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Alpha-blend ``color`` onto ``base`` where ``mask`` is True."""
    out = base.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    selected = np.asarray(mask).astype(bool)
    if np.any(selected):
        out[selected] = (1.0 - alpha) * out[selected] + alpha * color_arr
    return out


def _head_outline(head_mask: np.ndarray) -> np.ndarray:
    """Return a thin outline of the head mask using morphological gradient."""
    try:
        import cv2
    except ImportError:
        # Fallback: edge via neighbor difference if OpenCV is unavailable.
        head = head_mask.astype(np.uint8)
        padded = np.pad(head, 1, mode="edge")
        eroded = (
            padded[0:-2, 0:-2]
            & padded[0:-2, 1:-1]
            & padded[0:-2, 2:]
            & padded[1:-1, 0:-2]
            & padded[1:-1, 2:]
            & padded[2:, 0:-2]
            & padded[2:, 1:-1]
            & padded[2:, 2:]
        )
        return (head.astype(bool) & ~eroded.astype(bool))

    head_u8 = (np.asarray(head_mask).astype(bool).astype(np.uint8)) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(head_u8, kernel, iterations=1)
    outline = cv2.subtract(head_u8, eroded)
    return outline.astype(bool)


def create_overlay(
    image: Image.Image,
    hair_mask: np.ndarray,
    skin_mask: np.ndarray,
    head_mask: np.ndarray,
) -> Image.Image:
    """
    Build an RGB overlay: green hair, blue skin, red head outline.

    The original image remains the background.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = _blend_mask(base, hair_mask, HAIR_OVERLAY_COLOR, HAIR_OVERLAY_ALPHA)
    blended = _blend_mask(blended, skin_mask, SKIN_OVERLAY_COLOR, SKIN_OVERLAY_ALPHA)

    outline = _head_outline(head_mask)
    if np.any(outline):
        blended[outline] = np.array(HEAD_OUTLINE_COLOR, dtype=np.float32)

    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def print_summary(
    image_path: Path,
    image_size: tuple[int, int],
    hair_pixels: int,
    skin_pixels: int,
    head_pixels: int,
    hair_confidence: float | None,
) -> None:
    """Print a readable per-image segmentation summary."""
    print("=" * 50)
    print(f"Image: {image_path.name}")
    print()
    print(f"Image Size: {image_size}")
    print()
    print(f"Hair Pixels: {hair_pixels}")
    print(f"Skin Pixels: {skin_pixels}")
    print(f"Head Pixels: {head_pixels}")
    print()
    if hair_confidence is not None:
        print(f"Hair Confidence: {hair_confidence:.2f}")
    else:
        print("Hair Confidence: n/a")
    print("=" * 50)


def _hair_confidence(
    hair_mask: np.ndarray,
    hair_probability: np.ndarray | None,
) -> float | None:
    """Mean hair probability on predicted hair pixels, if available."""
    if hair_probability is None:
        return None
    probs = np.asarray(hair_probability, dtype=np.float64)
    mask = np.asarray(hair_mask).astype(bool)
    if np.any(mask):
        return float(probs[mask].mean())
    return float(probs.mean()) if probs.size else None


def process_image(image_path: Path, segmenter: HairSegmenter) -> None:
    """Segment one image, print diagnostics, and write mask/overlay outputs."""
    print(f"Processing {image_path.name}...")

    image = Image.open(image_path).convert("RGB")
    result = segmenter.segment(image)

    hair_mask = result["hair_mask"]
    skin_mask = result["skin_mask"]
    head_mask = result["head_mask"]
    image_size = result.get("image_size", image.size)
    hair_probability = result.get("hair_probability")

    hair_pixels = int(np.count_nonzero(hair_mask))
    skin_pixels = int(np.count_nonzero(skin_mask))
    head_pixels = int(np.count_nonzero(head_mask))
    confidence = _hair_confidence(hair_mask, hair_probability)

    print("✓ Segmentation complete")
    print_summary(
        image_path,
        image_size if isinstance(image_size, tuple) else tuple(image_size),
        hair_pixels,
        skin_pixels,
        head_pixels,
        confidence,
    )

    out_dir = OUTPUT_ROOT / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image.save(out_dir / "original.png")
    save_binary_mask(hair_mask, out_dir / "hair_mask.png")
    save_binary_mask(skin_mask, out_dir / "skin_mask.png")
    save_binary_mask(head_mask, out_dir / "head_mask.png")
    print("✓ Saved masks")

    overlay = create_overlay(image, hair_mask, skin_mask, head_mask)
    overlay.save(out_dir / "overlay.png")
    print("✓ Saved overlay")
    print("-" * 50)


def main() -> None:
    """Run segmentation diagnostics on all images under test_images/."""
    image_paths = load_images()
    if not image_paths:
        print("No images found under test_images/front or test_images/crown.")
        print(f"Expected extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        return

    print(f"Found {len(image_paths)} image(s). Loading HairSegmenter...")
    segmenter = HairSegmenter()
    print("Segmenter ready.\n")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    failures = 0

    for path in image_paths:
        try:
            process_image(path, segmenter)
        except Exception as exc:  # noqa: BLE001 — continue after per-image failures
            failures += 1
            print(f"✗ Failed on {path.name}: {exc}")
            traceback.print_exc()
            print("-" * 50)

    print(
        f"Done. Processed {len(image_paths)} image(s) "
        f"({len(image_paths) - failures} ok, {failures} failed)."
    )
    print(f"Outputs written to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
