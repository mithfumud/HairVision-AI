"""Shared types for normative hair-bearing region estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ViewType = Literal["front", "crown"]


@dataclass(frozen=True)
class RegionMasks:
    """
    Anatomical subregions of the normative hair-bearing envelope.

    Each field is a boolean mask with shape (height, width).
    """

    frontal: np.ndarray
    left_temple: np.ndarray
    right_temple: np.ndarray
    crown: np.ndarray


@dataclass(frozen=True)
class NormativeRegionResult:
    """
    Result of normative hair-bearing region estimation.

    ``normative_region_mask`` is the union of active subregions for the
    given view. ``metadata`` holds estimator-specific details (e.g. face
    scale, constants used) and is populated by later milestones.
    """

    normative_region_mask: np.ndarray
    region_masks: RegionMasks
    view: ViewType
    metadata: dict[str, Any]
