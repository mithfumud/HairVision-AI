"""Explainable numerical features from segmentation masks."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_bool_mask(mask: Any, name: str) -> np.ndarray:
    """Coerce a segmentation mask to a boolean ndarray."""
    if mask is None:
        raise KeyError(f"segmentation is missing required key: {name}")
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return arr.astype(bool)


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, or 0.0 when the denominator is zero."""
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


class FeatureExtractor:
    """Derive measurable hair/scalp ratios from HairSegmenter output."""

    def extract(self, segmentation: dict, image_type: str) -> dict[str, Any]:
        """
        Compute density and coverage features from segmentation masks.

        Scalp measurements use the model's skin_mask (not head minus hair).
        """
        if image_type not in ("front", "crown"):
            raise ValueError("image_type must be 'front' or 'crown'")
        if not isinstance(segmentation, dict):
            raise TypeError("segmentation must be a dict")

        hair_mask = _as_bool_mask(segmentation.get("hair_mask"), "hair_mask")
        skin_mask = _as_bool_mask(segmentation.get("skin_mask"), "skin_mask")
        head_mask = _as_bool_mask(segmentation.get("head_mask"), "head_mask")

        if hair_mask.shape != skin_mask.shape or hair_mask.shape != head_mask.shape:
            raise ValueError(
                "hair_mask, skin_mask, and head_mask must share the same shape"
            )

        # Scalp evidence comes from the skin class inside the head region.
        scalp_mask = skin_mask & head_mask

        hair_pixels = int(np.count_nonzero(hair_mask))
        scalp_pixels = int(np.count_nonzero(scalp_mask))
        head_pixels = int(np.count_nonzero(head_mask))

        hair_and_scalp = hair_pixels + scalp_pixels

        return {
            "image_type": image_type,
            "hair_pixels": hair_pixels,
            "scalp_pixels": scalp_pixels,
            "head_pixels": head_pixels,
            "hair_density": _safe_ratio(hair_pixels, hair_and_scalp),
            "scalp_exposure": _safe_ratio(scalp_pixels, head_pixels),
            "hair_percentage": _safe_ratio(hair_pixels, head_pixels),
            "scalp_percentage": _safe_ratio(scalp_pixels, head_pixels),
            "segmentation_coverage": _safe_ratio(hair_pixels + scalp_pixels, head_pixels),
        }
