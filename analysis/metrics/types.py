"""Structured clinical metrics derived from hair deficit analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from analysis.deficit.types import AnatomicalZone


class AnalysisMode(str, Enum):
    """Which anatomical views contributed normative subregions."""

    FRONT_ONLY = "front_only"
    CROWN_ONLY = "crown_only"
    COMBINED = "combined"


@dataclass(frozen=True)
class HairLossMetrics:
    """
    Quantitative hair-loss features for classification and reporting.

    All percentage fields are in [0, 100] when present. Regional fields are
    ``None`` when the corresponding normative subregion is empty (unavailable
    for that view).

    Analysis context (Norwood prep)
    -------------------------------
    ``analysis_mode``, ``front_available``, ``crown_available``:

        Describe which anatomical views were analyzed so the Norwood
        classifier can select rules without re-inspecting ``RegionMasks``.
        Computed once by ``HairLossMetricsExtractor``.

    Regional metrics (Norwood prep)
    -------------------------------
    ``front_loss_percentage``, ``left_temple_loss_percentage``,
    ``right_temple_loss_percentage``, ``crown_loss_percentage``:

        (deficit pixels in zone / normative pixels in zone) × 100

    Represent how much of each anatomical hair-bearing zone is missing hair.
    Later used by the Norwood classifier to distinguish frontal recession,
    temple thinning, and vertex loss patterns.

    Global metrics (Norwood prep)
    -----------------------------
    ``overall_hair_loss_percentage``:

        (total deficit pixels / total normative pixels) × 100

    Single summary of hair loss severity across the active normative envelope.

    ``overall_hair_coverage_percentage``:

        100 − overall_hair_loss_percentage

    Complementary coverage score; useful for reporting and threshold rules.

    Component metrics (Norwood prep)
    --------------------------------
    ``component_count``:

        Number of distinct connected deficit regions from analysis.

    ``largest_deficit_zone`` / ``largest_deficit_percentage``:

        Zone and severity of the largest single deficit patch, where severity
        is (largest patch area / normative area of that zone) × 100. Helps
        identify the dominant loss pattern when multiple patches exist.
    """

    front_loss_percentage: float | None
    left_temple_loss_percentage: float | None
    right_temple_loss_percentage: float | None
    crown_loss_percentage: float | None

    overall_hair_loss_percentage: float | None
    overall_hair_coverage_percentage: float | None

    component_count: int
    largest_deficit_zone: AnatomicalZone | None
    largest_deficit_percentage: float | None

    analysis_mode: AnalysisMode
    front_available: bool
    crown_available: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "front_loss_percentage": self.front_loss_percentage,
            "left_temple_loss_percentage": self.left_temple_loss_percentage,
            "right_temple_loss_percentage": self.right_temple_loss_percentage,
            "crown_loss_percentage": self.crown_loss_percentage,
            "overall_hair_loss_percentage": self.overall_hair_loss_percentage,
            "overall_hair_coverage_percentage": self.overall_hair_coverage_percentage,
            "component_count": self.component_count,
            "largest_deficit_zone": self.largest_deficit_zone,
            "largest_deficit_percentage": self.largest_deficit_percentage,
            "analysis_mode": self.analysis_mode.value,
            "front_available": self.front_available,
            "crown_available": self.crown_available,
        }
