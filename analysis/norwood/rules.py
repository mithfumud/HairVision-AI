"""Deterministic ordered Norwood rule engine.

One rule fires. No scoring, weights, probabilities, or ML.
Confidence and explanations are handled in separate modules.
"""

from __future__ import annotations

from collections.abc import Callable

from analysis.metrics.types import AnalysisMode, HairLossMetrics
from analysis.norwood.types import NorwoodStage, RuleMatch

# Regional / overall thresholds (percentages in [0, 100]).
_TEMPLE_MILD = 12.0
_TEMPLE_MODERATE = 25.0
_TEMPLE_ADVANCED = 45.0
_FRONT_MILD = 10.0
_FRONT_MODERATE = 22.0
_FRONT_ADVANCED = 40.0
_CROWN_MODERATE = 20.0
_CROWN_ADVANCED = 40.0
_OVERALL_ADVANCED = 35.0
_OVERALL_MILD = 8.0


def _v(value: float | None, default: float = 0.0) -> float:
    """Treat unavailable regional metrics as 0 for rule predicates."""
    return default if value is None else float(value)


def _max_temple(metrics: HairLossMetrics) -> float:
    return max(
        _v(metrics.left_temple_loss_percentage),
        _v(metrics.right_temple_loss_percentage),
    )


def _mean_temple(metrics: HairLossMetrics) -> float:
    left = metrics.left_temple_loss_percentage
    right = metrics.right_temple_loss_percentage
    if left is None and right is None:
        return 0.0
    if left is None:
        return float(right)
    if right is None:
        return float(left)
    return 0.5 * (float(left) + float(right))


def _match_iv_plus(metrics: HairLossMetrics) -> bool:
    """
    Advanced combined loss consistent with Norwood IV–VII.

    Fires when overall severity is high and at least two major zones show
    substantial deficit, or when front+temple+crown are all advanced.
    """
    overall = _v(metrics.overall_hair_loss_percentage)
    front = _v(metrics.front_loss_percentage)
    temple = _max_temple(metrics)
    crown = _v(metrics.crown_loss_percentage)

    if metrics.analysis_mode == AnalysisMode.CROWN_ONLY:
        return crown >= _CROWN_ADVANCED and overall >= _OVERALL_ADVANCED

    if metrics.analysis_mode == AnalysisMode.FRONT_ONLY:
        return (
            overall >= _OVERALL_ADVANCED
            and front >= _FRONT_ADVANCED
            and temple >= _TEMPLE_ADVANCED
        )

    # COMBINED
    advanced_zones = sum(
        (
            front >= _FRONT_ADVANCED,
            temple >= _TEMPLE_ADVANCED,
            crown >= _CROWN_ADVANCED,
        )
    )
    if advanced_zones >= 3:
        return True
    if overall >= _OVERALL_ADVANCED and advanced_zones >= 2:
        return True
    # Front + crown both advanced is advanced patterned loss even when
    # overall % is diluted by remaining hair in other zones.
    if front >= _FRONT_ADVANCED and crown >= _CROWN_ADVANCED:
        return True
    return False


def _match_iii_vertex(metrics: HairLossMetrics) -> bool:
    """Crown-dominant pattern with limited or moderate frontal/temple loss."""
    crown = _v(metrics.crown_loss_percentage)
    front = _v(metrics.front_loss_percentage)
    temple = _max_temple(metrics)

    if metrics.analysis_mode == AnalysisMode.FRONT_ONLY:
        return False

    if metrics.analysis_mode == AnalysisMode.CROWN_ONLY:
        return crown >= _CROWN_MODERATE

    return (
        crown >= _CROWN_MODERATE
        and crown >= front
        and crown >= temple
        and front < _FRONT_ADVANCED
        and temple < _TEMPLE_ADVANCED
    )


def _match_iii(metrics: HairLossMetrics) -> bool:
    """Deeper frontal / temporal recession (classic Norwood III)."""
    front = _v(metrics.front_loss_percentage)
    temple = _max_temple(metrics)
    crown = _v(metrics.crown_loss_percentage)

    if metrics.analysis_mode == AnalysisMode.CROWN_ONLY:
        return False

    deep_temples = temple >= _TEMPLE_MODERATE
    deep_front = front >= _FRONT_MODERATE
    crown_not_dominant = (
        metrics.analysis_mode != AnalysisMode.COMBINED
        or crown < max(front, temple)
        or crown < _CROWN_MODERATE
    )
    return (deep_temples or deep_front) and crown_not_dominant


def _match_ii(metrics: HairLossMetrics) -> bool:
    """Mild temporal and/or frontal recession."""
    front = _v(metrics.front_loss_percentage)
    temple = _max_temple(metrics)

    if metrics.analysis_mode == AnalysisMode.CROWN_ONLY:
        return False

    mild_temples = _TEMPLE_MILD <= temple < _TEMPLE_MODERATE
    mild_front = _FRONT_MILD <= front < _FRONT_MODERATE
    return mild_temples or mild_front


def _match_i(metrics: HairLossMetrics) -> bool:
    """
    Minimal loss.

    Requires measured zones to sit below mild thresholds so severe but
    previously unmatched patterns cannot silently fall through to Stage I.
    """
    front = _v(metrics.front_loss_percentage)
    temple = _max_temple(metrics)
    crown = _v(metrics.crown_loss_percentage)

    if metrics.front_available and (
        front >= _FRONT_MILD or temple >= _TEMPLE_MILD
    ):
        return False
    if metrics.crown_available and crown >= _CROWN_MODERATE:
        return False
    return True


# Ordered from most severe to mildest; first match wins.
_RULES: tuple[tuple[str, NorwoodStage, Callable[[HairLossMetrics], bool]], ...] = (
    ("NW-IVPLUS-01", NorwoodStage.IV_PLUS, _match_iv_plus),
    ("NW-IIIV-01", NorwoodStage.III_VERTEX, _match_iii_vertex),
    ("NW-III-01", NorwoodStage.III, _match_iii),
    ("NW-II-01", NorwoodStage.II, _match_ii),
    ("NW-I-01", NorwoodStage.I, _match_i),
)


class NorwoodRuleEngine:
    """
    Deterministic rule list over ``HairLossMetrics``.

    Exactly one rule fires. Does not compute confidence or explanations.
    """

    def __init__(
        self,
        rules: tuple[
            tuple[str, NorwoodStage, Callable[[HairLossMetrics], bool]],
            ...,
        ]
        | None = None,
    ) -> None:
        self._rules = rules if rules is not None else _RULES

    def evaluate(self, metrics: HairLossMetrics) -> RuleMatch:
        """Return the first matching rule (ordered severe → mild)."""
        for rule_id, stage, predicate in self._rules:
            if predicate(metrics):
                return RuleMatch(stage=stage, rule_id=rule_id)
        # Safety net for rare unmatched severe patterns after Stage I
        # stopped being an unconditional catch-all.
        return RuleMatch(stage=NorwoodStage.IV_PLUS, rule_id="NW-IVPLUS-01")
