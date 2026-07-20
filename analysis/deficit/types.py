"""Structured types for hair deficit analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

AnatomicalZone = Literal[
    "frontal",
    "left_temple",
    "right_temple",
    "crown",
    "unknown",
]


@dataclass(frozen=True)
class DeficitComponent:
    """
    One connected hair-deficit region inside the normative envelope.

    ``zone_area_ratio`` is the component area divided by the total pixel
    count of the assigned anatomical subregion (how much of that zone is
    affected).

    ``zone_overlap_ratio`` is the fraction of this component's pixels that
    fall inside the assigned zone:

        (component pixels inside assigned zone) / (total component pixels)

    Example: a component with 91% of its pixels in ``left_temple`` and 9%
    in ``frontal`` is assigned ``left_temple`` with
    ``zone_overlap_ratio = 0.91``.

    A high ratio (near 1.0) means the zone label is confident. A lower ratio
    indicates the deficit spans zone boundaries — useful later for Norwood
    staging, explainability, and flagging ambiguous regional hair loss.
    """

    label_id: int
    zone: AnatomicalZone
    area_px: int
    zone_area_ratio: float
    zone_overlap_ratio: float
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    contour: np.ndarray


@dataclass(frozen=True)
class DeficitStatistics:
    """Aggregate hair-deficit measurements."""

    total_deficit_area_px: int
    deficit_percentage: float
    normative_area_px: int
    component_count: int
    largest_component_area_px: int
    raw_deficit_area_px: int


@dataclass(frozen=True)
class HairDeficitResult:
    """Complete output of hair deficit analysis."""

    deficit_mask: np.ndarray
    components: tuple[DeficitComponent, ...]
    largest_component: DeficitComponent | None
    statistics: DeficitStatistics
