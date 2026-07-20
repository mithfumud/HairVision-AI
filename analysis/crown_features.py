"""Crown-view measurable features from segmentation masks."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

_DEFAULT_MIN_REGION_AREA_RATIO = 0.002
_DEFAULT_CROWN_TOP_FRACTION = 0.55
_DEFAULT_CROWN_WIDTH_FRACTION = 0.60


def _as_bool_mask(mask: Any, name: str) -> np.ndarray:
    """Coerce a segmentation mask to a boolean ndarray."""
    if mask is None:
        raise KeyError(f"segmentation is missing required key: {name}")
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return arr.astype(bool)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when the denominator is zero."""
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


class CrownFeatureExtractor:
    """Extract measurable crown-view features (no classification)."""

    def __init__(
        self,
        min_region_area_ratio: float = _DEFAULT_MIN_REGION_AREA_RATIO,
        crown_top_fraction: float = _DEFAULT_CROWN_TOP_FRACTION,
        crown_width_fraction: float = _DEFAULT_CROWN_WIDTH_FRACTION,
    ) -> None:
        if not 0.0 < min_region_area_ratio < 1.0:
            raise ValueError("min_region_area_ratio must be in the interval (0, 1)")
        if not 0.0 < crown_top_fraction <= 1.0:
            raise ValueError("crown_top_fraction must be in the interval (0, 1]")
        if not 0.0 < crown_width_fraction <= 1.0:
            raise ValueError("crown_width_fraction must be in the interval (0, 1]")

        self.min_region_area_ratio = min_region_area_ratio
        self.crown_top_fraction = crown_top_fraction
        self.crown_width_fraction = crown_width_fraction

    def extract(
        self,
        image: Image.Image,
        segmentation: dict,
    ) -> dict[str, float | int]:
        """
        Compute crown-view measurements from hair/skin/head masks.

        Returns pixel area for the largest bald region plus resolution-aware
        percentages and a normalized crown-center offset.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if not isinstance(segmentation, dict):
            raise TypeError("segmentation must be a dict")

        width, height = image.size
        hair_mask = _as_bool_mask(segmentation.get("hair_mask"), "hair_mask")
        head_mask = _as_bool_mask(segmentation.get("head_mask"), "head_mask")
        skin_mask = _as_bool_mask(segmentation.get("skin_mask"), "skin_mask")

        expected = (height, width)
        for name, mask in (
            ("hair_mask", hair_mask),
            ("head_mask", head_mask),
            ("skin_mask", skin_mask),
        ):
            if mask.shape != expected:
                raise ValueError(
                    f"{name} must match the image dimensions (expected {expected})"
                )

        # Scalp = visible non-hair area inside the head.
        # Prefer head & ~hair over skin ∩ head: face-parsing "skin" often misses
        # crown/scalp tones, while any head pixel that is not hair is exposed scalp
        # (or skin) for crown analysis.
        scalp_mask = head_mask & ~hair_mask
        head_pixels = int(np.count_nonzero(head_mask))
        min_area = max(1, int(self.min_region_area_ratio * max(head_pixels, 1)))

        components = self._scalp_components(scalp_mask, min_area)
        largest = self._largest_component(components)

        largest_area = int(largest["area"]) if largest is not None else 0
        largest_pct = 100.0 * _safe_ratio(float(largest_area), float(head_pixels))

        crown_roi = self._crown_roi(head_mask)
        crown_scalp_exposure = self._roi_scalp_exposure(scalp_mask, crown_roi)

        return {
            "largest_bald_region_area": largest_area,
            "largest_bald_region_percentage": largest_pct,
            "crown_scalp_exposure": crown_scalp_exposure,
            "thinning_region_count": len(components),
            "crown_center_offset": self._crown_center_offset(
                head_mask, largest, head_pixels
            ),
        }

    def _scalp_components(
        self,
        scalp_mask: np.ndarray,
        min_area: int,
    ) -> list[dict[str, Any]]:
        """Connected scalp components larger than ``min_area``."""
        if not np.any(scalp_mask):
            return []

        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            scalp_mask.astype(np.uint8),
            connectivity=8,
        )

        regions: list[dict[str, Any]] = []
        # Label 0 is background.
        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            regions.append(
                {
                    "label": label,
                    "area": area,
                    "centroid": (
                        float(centroids[label][0]),
                        float(centroids[label][1]),
                    ),
                    "mask": labels == label,
                }
            )
        return regions

    def _largest_component(
        self,
        components: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the component with the greatest area, if any."""
        if not components:
            return None
        return max(components, key=lambda item: item["area"])

    def _crown_roi(self, head_mask: np.ndarray) -> np.ndarray:
        """
        Upper-central ROI within the head mask.

        Uses the head bounding box: keep the top ``crown_top_fraction`` of its
        height and the central ``crown_width_fraction`` of its width.
        """
        ys, xs = np.where(head_mask)
        roi = np.zeros_like(head_mask, dtype=bool)
        if len(xs) == 0:
            return roi

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        box_w = max(1, x_max - x_min + 1)
        box_h = max(1, y_max - y_min + 1)

        y_cut = y_min + int(self.crown_top_fraction * box_h)
        x_center = 0.5 * (x_min + x_max)
        half_w = 0.5 * self.crown_width_fraction * box_w
        x0 = int(np.floor(x_center - half_w))
        x1 = int(np.ceil(x_center + half_w))

        band = np.zeros_like(head_mask, dtype=bool)
        band[y_min : y_cut + 1, max(0, x0) : x1 + 1] = True
        return head_mask & band

    def _roi_scalp_exposure(
        self,
        scalp_mask: np.ndarray,
        crown_roi: np.ndarray,
    ) -> float:
        """Scalp exposure inside the crown ROI as a percentage (0–100)."""
        roi_pixels = int(np.count_nonzero(crown_roi))
        if roi_pixels == 0:
            return 0.0
        scalp_in_roi = int(np.count_nonzero(scalp_mask & crown_roi))
        return 100.0 * _safe_ratio(float(scalp_in_roi), float(roi_pixels))

    def _crown_center_offset(
        self,
        head_mask: np.ndarray,
        largest: dict[str, Any] | None,
        head_pixels: int,
    ) -> float:
        """
        Normalized distance between head centroid and largest bald centroid.

        Normalization uses the equivalent head radius ``sqrt(area / pi)``.
        """
        if largest is None or head_pixels <= 0:
            return 0.0

        head_ys, head_xs = np.where(head_mask)
        if len(head_xs) == 0:
            return 0.0

        head_cx = float(head_xs.mean())
        head_cy = float(head_ys.mean())
        bald_cx, bald_cy = largest["centroid"]

        distance = float(np.hypot(bald_cx - head_cx, bald_cy - head_cy))
        head_radius = float(np.sqrt(head_pixels / np.pi))
        return _safe_ratio(distance, head_radius)
