"""Confidence computation for Norwood classification.

Starts at 100% and applies penalties. Never changes the predicted stage.
"""

from __future__ import annotations

from analysis.metrics.types import AnalysisMode, HairLossMetrics
from analysis.norwood.types import ConfidenceBand, ConfidenceResult, NorwoodStage, RuleMatch

_START_CONFIDENCE = 100.0

# Penalty magnitudes (percentage points).
_PENALTY_FRONT_ONLY = 12.0
_PENALTY_CROWN_ONLY = 15.0
_PENALTY_NEAR_BOUNDARY = 10.0
_PENALTY_OVERALL_CONFLICT = 12.0
_PENALTY_MISSING_SUPPORT = 8.0
_PENALTY_IV_PLUS_UNDERSPECIFIED = 10.0

# Near-boundary windows around rule thresholds (percent points).
_TEMPLE_MILD = 12.0
_TEMPLE_MODERATE = 25.0
_FRONT_MILD = 10.0
_FRONT_MODERATE = 22.0
_CROWN_MODERATE = 20.0
_CROWN_ADVANCED = 40.0
_BOUNDARY_WINDOW = 3.0


def _v(value: float | None, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _near(value: float, threshold: float, window: float = _BOUNDARY_WINDOW) -> bool:
    return abs(value - threshold) <= window


def _confidence_band(confidence: float) -> ConfidenceBand:
    if confidence >= 75.0:
        return ConfidenceBand.HIGH
    if confidence >= 50.0:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


def _analysis_mode_penalties(metrics: HairLossMetrics) -> list[tuple[str, float]]:
    if metrics.analysis_mode == AnalysisMode.FRONT_ONLY:
        return [("front_only_analysis", _PENALTY_FRONT_ONLY)]
    if metrics.analysis_mode == AnalysisMode.CROWN_ONLY:
        return [("crown_only_analysis", _PENALTY_CROWN_ONLY)]
    return []


def _near_boundary_penalties(
    metrics: HairLossMetrics,
    match: RuleMatch,
) -> list[tuple[str, float]]:
    front = _v(metrics.front_loss_percentage)
    temple = max(
        _v(metrics.left_temple_loss_percentage),
        _v(metrics.right_temple_loss_percentage),
    )
    crown = _v(metrics.crown_loss_percentage)

    near = False
    if match.stage in (NorwoodStage.I, NorwoodStage.II):
        near = _near(temple, _TEMPLE_MILD) or _near(front, _FRONT_MILD)
    elif match.stage == NorwoodStage.III:
        near = _near(temple, _TEMPLE_MODERATE) or _near(front, _FRONT_MODERATE)
    elif match.stage == NorwoodStage.III_VERTEX:
        near = _near(crown, _CROWN_MODERATE)
    elif match.stage == NorwoodStage.IV_PLUS:
        near = _near(crown, _CROWN_ADVANCED) or _near(
            _v(metrics.overall_hair_loss_percentage),
            35.0,
        )

    if near:
        return [("near_decision_boundary", _PENALTY_NEAR_BOUNDARY)]
    return []


def _overall_conflict_penalties(
    metrics: HairLossMetrics,
    match: RuleMatch,
) -> list[tuple[str, float]]:
    overall = metrics.overall_hair_loss_percentage
    if overall is None:
        return []

    # Low overall loss but advanced stage (or vice versa).
    if match.stage == NorwoodStage.IV_PLUS and overall < 25.0:
        return [("overall_metric_conflict", _PENALTY_OVERALL_CONFLICT)]
    if match.stage == NorwoodStage.I and overall > 20.0:
        return [("overall_metric_conflict", _PENALTY_OVERALL_CONFLICT)]
    if match.stage == NorwoodStage.III_VERTEX and overall > 45.0:
        return [("overall_metric_conflict", _PENALTY_OVERALL_CONFLICT)]
    return []


def _missing_support_penalties(
    metrics: HairLossMetrics,
    match: RuleMatch,
) -> list[tuple[str, float]]:
    penalties: list[tuple[str, float]] = []

    if match.stage in (NorwoodStage.II, NorwoodStage.III) and not metrics.front_available:
        penalties.append(("missing_supporting_evidence", _PENALTY_MISSING_SUPPORT))

    if match.stage == NorwoodStage.III_VERTEX and not metrics.crown_available:
        penalties.append(("missing_supporting_evidence", _PENALTY_MISSING_SUPPORT))

    if match.stage == NorwoodStage.IV_PLUS and metrics.analysis_mode != AnalysisMode.COMBINED:
        penalties.append(("iv_plus_underspecified", _PENALTY_IV_PLUS_UNDERSPECIFIED))

    return penalties


class NorwoodConfidenceEngine:
    """
    Compute confidence from analysis context and metric consistency.

    Never modifies ``RuleMatch.stage``.
    """

    def evaluate(
        self,
        metrics: HairLossMetrics,
        match: RuleMatch,
    ) -> ConfidenceResult:
        """Start at 100 and apply documented penalties; clamp to [0, 100]."""
        score = _START_CONFIDENCE
        applied: list[str] = []

        for label, amount in (
            *_analysis_mode_penalties(metrics),
            *_near_boundary_penalties(metrics, match),
            *_overall_conflict_penalties(metrics, match),
            *_missing_support_penalties(metrics, match),
        ):
            score -= amount
            applied.append(label)

        confidence = float(min(100.0, max(0.0, score)))
        return ConfidenceResult(
            confidence=confidence,
            confidence_band=_confidence_band(confidence),
            penalties=tuple(applied),
        )
