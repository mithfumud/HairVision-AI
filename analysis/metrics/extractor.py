"""Convert hair deficit analysis results into clinical metrics."""

from __future__ import annotations

import numpy as np

from analysis.deficit.types import HairDeficitResult
from analysis.metrics.types import AnalysisMode, HairLossMetrics
from analysis.normative_region.types import RegionMasks

_REGION_FIELDS: tuple[tuple[str, str], ...] = (
    ("frontal", "front_loss_percentage"),
    ("left_temple", "left_temple_loss_percentage"),
    ("right_temple", "right_temple_loss_percentage"),
    ("crown", "crown_loss_percentage"),
)


def _clamp_percentage(value: float) -> float:
    """Keep a percentage within [0, 100]."""
    return float(min(100.0, max(0.0, value)))


def _region_normative_pixels(region_mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(region_mask).astype(bool)))


def _region_deficit_pixels(
    deficit_mask: np.ndarray,
    region_mask: np.ndarray,
) -> int:
    deficit = np.asarray(deficit_mask).astype(bool)
    region = np.asarray(region_mask).astype(bool)
    return int(np.count_nonzero(deficit & region))


def _region_loss_percentage(
    deficit_mask: np.ndarray,
    region_mask: np.ndarray,
) -> float | None:
    """
    Compute regional loss percentage.

    Formula:
        (deficit pixels in region / normative pixels in region) × 100

    Returns ``None`` when the normative subregion is empty (unavailable).
    """
    normative_pixels = _region_normative_pixels(region_mask)
    if normative_pixels == 0:
        return None

    deficit_pixels = _region_deficit_pixels(deficit_mask, region_mask)
    return _clamp_percentage(100.0 * deficit_pixels / normative_pixels)


def _front_available(region_masks: RegionMasks) -> bool:
    """True when any front-view normative subregion is populated."""
    frontal = _region_normative_pixels(region_masks.frontal)
    left = _region_normative_pixels(region_masks.left_temple)
    right = _region_normative_pixels(region_masks.right_temple)
    return (frontal + left + right) > 0


def _crown_available(region_masks: RegionMasks) -> bool:
    """True when the crown normative subregion is populated."""
    return _region_normative_pixels(region_masks.crown) > 0


def _analysis_mode(front_available: bool, crown_available: bool) -> AnalysisMode:
    """
    Derive analysis mode from populated normative subregions.

    Rules:
        front + no crown  → FRONT_ONLY
        crown + no front  → CROWN_ONLY
        both              → COMBINED
    """
    if front_available and crown_available:
        return AnalysisMode.COMBINED
    if front_available:
        return AnalysisMode.FRONT_ONLY
    if crown_available:
        return AnalysisMode.CROWN_ONLY
    raise ValueError("RegionMasks contain no populated front or crown subregions")


class HairLossMetricsExtractor:
    """
    Extract clinical metrics from ``HairDeficitResult`` and ``RegionMasks``.

    This layer performs no image processing — it only aggregates counts and
    percentages already implied by the deficit mask and normative subregions.
    """

    def extract(
        self,
        deficit_result: HairDeficitResult,
        region_masks: RegionMasks,
    ) -> HairLossMetrics:
        """
        Build ``HairLossMetrics`` from deficit analysis output.

        Parameters
        ----------
        deficit_result
            Output of ``HairDeficitAnalyzer.analyze()``.
        region_masks
            Anatomical subregion masks from the normative region estimator.

        Returns
        -------
        HairLossMetrics
            Region-wise and overall loss percentages plus component summaries.
        """
        deficit_mask = np.asarray(deficit_result.deficit_mask).astype(bool)
        stats = deficit_result.statistics

        regional: dict[str, float | None] = {}
        for field_name, metric_name in _REGION_FIELDS:
            region_mask = getattr(region_masks, field_name)
            regional[metric_name] = _region_loss_percentage(deficit_mask, region_mask)

        overall_loss: float | None
        overall_coverage: float | None
        if stats.normative_area_px > 0:
            overall_loss = _clamp_percentage(stats.deficit_percentage * 100.0)
            overall_coverage = _clamp_percentage(100.0 - overall_loss)
        else:
            overall_loss = None
            overall_coverage = None

        largest = deficit_result.largest_component
        if largest is None:
            largest_zone = None
            largest_percentage = None
        else:
            largest_zone = largest.zone
            largest_percentage = _clamp_percentage(largest.zone_area_ratio * 100.0)

        front_available = _front_available(region_masks)
        crown_available = _crown_available(region_masks)
        analysis_mode = _analysis_mode(front_available, crown_available)

        return HairLossMetrics(
            front_loss_percentage=regional["front_loss_percentage"],
            left_temple_loss_percentage=regional["left_temple_loss_percentage"],
            right_temple_loss_percentage=regional["right_temple_loss_percentage"],
            crown_loss_percentage=regional["crown_loss_percentage"],
            overall_hair_loss_percentage=overall_loss,
            overall_hair_coverage_percentage=overall_coverage,
            component_count=stats.component_count,
            largest_deficit_zone=largest_zone,
            largest_deficit_percentage=largest_percentage,
            analysis_mode=analysis_mode,
            front_available=front_available,
            crown_available=crown_available,
        )
