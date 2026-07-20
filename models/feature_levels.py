"""
Map numeric hair-analysis measurements to semantic feature levels.

All bin boundaries live here so Norwood rules stay free of raw thresholds.
Bins follow each feature's natural scale from the extraction pipeline:
- ratios from FeatureExtractor / FrontFeatureExtractor are in [0, 1]
- crown area/exposure metrics from CrownFeatureExtractor are in [0, 100]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable


class FeatureLevel(Enum):
    """Strongly typed semantic levels used across all Norwood features."""

    VERY_LOW = auto()
    LOW = auto()
    MODERATE = auto()
    MEDIUM = auto()
    HIGH = auto()
    VERY_HIGH = auto()
    NORMAL = auto()
    ELEVATED = auto()
    SMALL = auto()
    LARGE = auto()


@dataclass(frozen=True)
class LevelBin:
    """Half-open interval [low, high) mapped to a semantic level."""

    level: FeatureLevel
    low: float
    high: float


# ---------------------------------------------------------------------------
# Centralized numeric → semantic mappings
# ---------------------------------------------------------------------------

HAIR_DENSITY_LEVELS: tuple[LevelBin, ...] = (
    LevelBin(FeatureLevel.VERY_LOW, 0.0, 0.30),
    LevelBin(FeatureLevel.LOW, 0.30, 0.50),
    LevelBin(FeatureLevel.MEDIUM, 0.50, 0.70),
    LevelBin(FeatureLevel.HIGH, 0.70, 1.01),
)

SCALP_EXPOSURE_LEVELS: tuple[LevelBin, ...] = (
    LevelBin(FeatureLevel.LOW, 0.0, 0.30),
    LevelBin(FeatureLevel.MEDIUM, 0.30, 0.55),
    LevelBin(FeatureLevel.HIGH, 0.55, 1.01),
)

TEMPLE_RECESSION_LEVELS: tuple[LevelBin, ...] = (
    LevelBin(FeatureLevel.LOW, 0.0, 0.20),
    LevelBin(FeatureLevel.MODERATE, 0.20, 0.45),
    LevelBin(FeatureLevel.HIGH, 0.45, 10.0),
)

HAIRLINE_HEIGHT_LEVELS: tuple[LevelBin, ...] = (
    LevelBin(FeatureLevel.NORMAL, 0.0, 0.12),
    LevelBin(FeatureLevel.ELEVATED, 0.12, 0.25),
    LevelBin(FeatureLevel.HIGH, 0.25, 10.0),
)

LARGEST_BALD_REGION_LEVELS: tuple[LevelBin, ...] = (
    LevelBin(FeatureLevel.SMALL, 0.0, 12.0),
    LevelBin(FeatureLevel.MEDIUM, 12.0, 30.0),
    LevelBin(FeatureLevel.LARGE, 30.0, 100.01),
)

CROWN_SCALP_EXPOSURE_LEVELS: tuple[LevelBin, ...] = (
    LevelBin(FeatureLevel.LOW, 0.0, 30.0),
    LevelBin(FeatureLevel.MEDIUM, 30.0, 55.0),
    LevelBin(FeatureLevel.HIGH, 55.0, 100.01),
)

_FEATURE_BINS: dict[str, tuple[LevelBin, ...]] = {
    "hair_density": HAIR_DENSITY_LEVELS,
    "scalp_exposure": SCALP_EXPOSURE_LEVELS,
    "left_temple_recession": TEMPLE_RECESSION_LEVELS,
    "right_temple_recession": TEMPLE_RECESSION_LEVELS,
    "temple_recession": TEMPLE_RECESSION_LEVELS,
    "hairline_height": HAIRLINE_HEIGHT_LEVELS,
    "largest_bald_region_percentage": LARGEST_BALD_REGION_LEVELS,
    "crown_scalp_exposure": CROWN_SCALP_EXPOSURE_LEVELS,
}

_FEATURE_LABELS: dict[str, str] = {
    "hair_density": "Hair Density",
    "scalp_exposure": "Scalp Exposure",
    "left_temple_recession": "Left Temple Recession",
    "right_temple_recession": "Right Temple Recession",
    "temple_recession": "Temple Recession",
    "hairline_height": "Hairline Height",
    "largest_bald_region_percentage": "Largest Bald Region",
    "crown_scalp_exposure": "Crown Scalp Exposure",
}

_PERCENT_FEATURES = frozenset(
    {
        "largest_bald_region_percentage",
        "crown_scalp_exposure",
    }
)


def level_for_value(bins: Iterable[LevelBin], value: float) -> FeatureLevel:
    """Return the semantic level for ``value`` given ordered bins."""
    ordered = tuple(bins)
    if not ordered:
        raise ValueError("bins must not be empty")
    if value < ordered[0].low:
        return ordered[0].level
    for bin_ in ordered:
        if bin_.low <= value < bin_.high:
            return bin_.level
    return ordered[-1].level


def format_measurement(feature: str, value: float) -> str:
    """Format a raw measurement for concise reasoning text."""
    if feature in _PERCENT_FEATURES:
        return f"{value:.0f}%"
    return f"{value:.2f}"


class FeatureLevelEvaluator:
    """Convert numeric measurement dicts into semantic feature levels."""

    def __init__(
        self,
        feature_bins: dict[str, tuple[LevelBin, ...]] | None = None,
    ) -> None:
        self._bins = feature_bins if feature_bins is not None else dict(_FEATURE_BINS)

    def label_for(self, feature: str) -> str:
        """Pretty name used in reasoning lines."""
        return _FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

    def level(self, feature: str, value: float) -> FeatureLevel:
        """Map one numeric measurement to a semantic level."""
        bins = self._bins.get(feature)
        if bins is None:
            raise KeyError(f"no level mapping defined for feature: {feature}")
        return level_for_value(bins, value)

    def evaluate(self, features: dict[str, float]) -> dict[str, FeatureLevel]:
        """
        Build a level map for all known features present in ``features``.

        Also derives ``temple_recession`` as max(left, right) when either side
        is available, so rules can reason about overall temple severity.
        """
        levels: dict[str, FeatureLevel] = {}
        for feature, bins in self._bins.items():
            if feature == "temple_recession":
                continue
            if feature not in features:
                continue
            levels[feature] = level_for_value(bins, features[feature])

        left = features.get("left_temple_recession")
        right = features.get("right_temple_recession")
        if left is not None or right is not None:
            severity = max(v for v in (left, right) if v is not None)
            levels["temple_recession"] = level_for_value(
                self._bins["temple_recession"], severity
            )

        return levels
