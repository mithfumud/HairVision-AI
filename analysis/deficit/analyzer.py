"""Hair deficit analysis from normative regions and segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np

from analysis.deficit.metrics import (
    DEFAULT_MIN_COMPONENT_FRACTION,
    assign_component_zone,
    bounding_box_from_stats,
    clean_deficit_mask,
    component_centroid,
    component_contour,
    compute_raw_deficit_mask,
    compute_statistics,
    zone_pixel_counts,
)
from analysis.deficit.types import DeficitComponent, HairDeficitResult
from analysis.normative_region.types import RegionMasks


class HairDeficitAnalyzer:
    """
    Compute structured hair deficit from normative and segmentation masks.

    Analysis only — no rendering or visualization.
    """

    def __init__(
        self,
        *,
        min_component_fraction: float = DEFAULT_MIN_COMPONENT_FRACTION,
    ) -> None:
        self.min_component_fraction = min_component_fraction

    def analyze(
        self,
        normative_region_mask: np.ndarray,
        hair_mask: np.ndarray,
        head_mask: np.ndarray,
        region_masks: RegionMasks,
    ) -> HairDeficitResult:
        """
        Compute hair deficit inside the normative envelope.

        Pipeline:
            1. raw_deficit = normative & ~hair
            2. remove tiny components (threshold scales with head area)
            3. extract connected components with geometry and zone labels
            4. aggregate statistics
        """
        normative = np.asarray(normative_region_mask).astype(bool)
        hair = np.asarray(hair_mask).astype(bool)
        head = np.asarray(head_mask).astype(bool)

        if normative.shape != hair.shape or normative.shape != head.shape:
            raise ValueError("normative, hair, and head masks must share the same shape")

        raw_deficit = compute_raw_deficit_mask(normative, hair)
        deficit_mask = clean_deficit_mask(
            raw_deficit,
            head,
            min_component_fraction=self.min_component_fraction,
        )

        components = self._extract_components(
            deficit_mask,
            region_masks,
        )
        component_areas = [component.area_px for component in components]
        largest_component = (
            max(components, key=lambda item: item.area_px) if components else None
        )

        statistics = compute_statistics(
            normative_region_mask=normative,
            raw_deficit_mask=raw_deficit,
            deficit_mask=deficit_mask,
            component_areas=component_areas,
        )

        return HairDeficitResult(
            deficit_mask=deficit_mask,
            components=tuple(components),
            largest_component=largest_component,
            statistics=statistics,
        )

    def _extract_components(
        self,
        deficit_mask: np.ndarray,
        region_masks: RegionMasks,
    ) -> list[DeficitComponent]:
        """Build structured component records from a cleaned deficit mask."""
        deficit = np.asarray(deficit_mask).astype(bool)
        if not np.any(deficit):
            return []

        zone_counts = zone_pixel_counts(region_masks)
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            deficit.astype(np.uint8),
            connectivity=8,
        )

        components: list[DeficitComponent] = []
        for label in range(1, label_count):
            component_mask = labels == label
            area_px = int(stats[label, cv2.CC_STAT_AREA])
            zone, zone_overlap_ratio = assign_component_zone(component_mask, region_masks)
            zone_area = zone_counts.get(zone, 0)
            zone_area_ratio = float(area_px / zone_area) if zone_area > 0 else 0.0

            components.append(
                DeficitComponent(
                    label_id=label,
                    zone=zone,
                    area_px=area_px,
                    zone_area_ratio=zone_area_ratio,
                    zone_overlap_ratio=zone_overlap_ratio,
                    centroid=component_centroid(component_mask),
                    bounding_box=bounding_box_from_stats(stats[label]),
                    contour=component_contour(component_mask),
                )
            )

        components.sort(key=lambda item: item.area_px, reverse=True)
        return components
