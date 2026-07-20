"""Metric helpers for hair deficit analysis."""

from __future__ import annotations

import cv2
import numpy as np

from analysis.deficit.types import AnatomicalZone, DeficitStatistics
from analysis.normative_region.types import RegionMasks

# Default minimum connected-component area as a fraction of head mask pixels.
# min_component_area = min_component_fraction × head_mask_area
DEFAULT_MIN_COMPONENT_FRACTION = 0.002

_ZONE_FIELDS: tuple[tuple[str, AnatomicalZone], ...] = (
    ("frontal", "frontal"),
    ("left_temple", "left_temple"),
    ("right_temple", "right_temple"),
    ("crown", "crown"),
)


def compute_raw_deficit_mask(
    normative_region_mask: np.ndarray,
    hair_mask: np.ndarray,
) -> np.ndarray:
    """Pixels inside normative where hair is absent."""
    normative = np.asarray(normative_region_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)
    return normative & ~hair


def min_component_area(head_mask: np.ndarray, *, min_component_fraction: float) -> int:
    """Scale the speckle-removal threshold from head mask area."""
    head_pixels = int(np.count_nonzero(np.asarray(head_mask).astype(bool)))
    return max(1, int(min_component_fraction * head_pixels))


def clean_deficit_mask(
    raw_deficit_mask: np.ndarray,
    head_mask: np.ndarray,
    *,
    min_component_fraction: float = DEFAULT_MIN_COMPONENT_FRACTION,
) -> np.ndarray:
    """Remove tiny connected components using head-scaled minimum area."""
    raw = np.asarray(raw_deficit_mask).astype(bool)
    head = np.asarray(head_mask).astype(bool)
    if not np.any(raw):
        return np.zeros_like(raw, dtype=bool)

    threshold = min_component_area(head, min_component_fraction=min_component_fraction)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(raw, dtype=bool)
    for label in range(1, label_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= threshold:
            cleaned[labels == label] = True
    return cleaned & head


def zone_pixel_counts(region_masks: RegionMasks) -> dict[AnatomicalZone, int]:
    """Return pixel counts for each anatomical subregion."""
    counts: dict[AnatomicalZone, int] = {}
    for field_name, zone_name in _ZONE_FIELDS:
        mask = getattr(region_masks, field_name)
        counts[zone_name] = int(np.count_nonzero(np.asarray(mask).astype(bool)))
    return counts


def assign_component_zone(
    component_mask: np.ndarray,
    region_masks: RegionMasks,
) -> tuple[AnatomicalZone, float]:
    """
    Assign the anatomical zone with the largest overlap.

    Returns ``(zone, zone_overlap_ratio)`` where ``zone_overlap_ratio`` is
    the fraction of component pixels inside the assigned zone.
    """
    component = np.asarray(component_mask).astype(bool)
    area_px = int(np.count_nonzero(component))
    if area_px == 0:
        return "unknown", 0.0

    best_zone: AnatomicalZone = "unknown"
    best_overlap = 0

    for field_name, zone_name in _ZONE_FIELDS:
        zone_mask = np.asarray(getattr(region_masks, field_name)).astype(bool)
        overlap = int(np.count_nonzero(component & zone_mask))
        if overlap > best_overlap:
            best_overlap = overlap
            best_zone = zone_name

    zone_overlap_ratio = float(best_overlap / area_px) if area_px > 0 else 0.0
    return best_zone, zone_overlap_ratio


def compute_statistics(
    *,
    normative_region_mask: np.ndarray,
    raw_deficit_mask: np.ndarray,
    deficit_mask: np.ndarray,
    component_areas: list[int],
) -> DeficitStatistics:
    """Aggregate deficit statistics from cleaned and raw masks."""
    normative_area = int(np.count_nonzero(np.asarray(normative_region_mask).astype(bool)))
    total_deficit = int(np.count_nonzero(np.asarray(deficit_mask).astype(bool)))
    raw_deficit = int(np.count_nonzero(np.asarray(raw_deficit_mask).astype(bool)))
    largest_area = max(component_areas) if component_areas else 0

    deficit_percentage = (
        float(total_deficit / normative_area) if normative_area > 0 else 0.0
    )

    return DeficitStatistics(
        total_deficit_area_px=total_deficit,
        deficit_percentage=deficit_percentage,
        normative_area_px=normative_area,
        component_count=len(component_areas),
        largest_component_area_px=largest_area,
        raw_deficit_area_px=raw_deficit,
    )


def component_centroid(component_mask: np.ndarray) -> tuple[float, float]:
    """Return (x, y) centroid for one binary component mask."""
    ys, xs = np.where(np.asarray(component_mask).astype(bool))
    if len(xs) == 0:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def component_contour(component_mask: np.ndarray) -> np.ndarray:
    """Return the external contour of one component as an (N, 1, 2) array."""
    mask_u8 = np.asarray(component_mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 1, 2), dtype=np.int32)
    return contours[0]


def bounding_box_from_stats(
    stats_row: np.ndarray,
) -> tuple[int, int, int, int]:
    """Convert OpenCV CC stats row to (x, y, width, height)."""
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    width = int(stats_row[cv2.CC_STAT_WIDTH])
    height = int(stats_row[cv2.CC_STAT_HEIGHT])
    return x, y, width, height
