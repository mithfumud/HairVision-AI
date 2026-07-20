"""Explainable rule-based Norwood stage classification."""

from __future__ import annotations

from typing import Any, Literal

from .feature_levels import (
    FeatureLevel,
    FeatureLevelEvaluator,
    format_measurement,
)
from .norwood_rules import NorwoodRule, norwood_rules

_REQUIRED_FEATURES = ("hair_density", "scalp_exposure")
_DEFAULT_MIN_CLASSIFICATION_SCORE = 0.60

ClassificationStatus = Literal["matched", "inconclusive"]


class NorwoodClassifier:
    """
    Expert-system Norwood classifier.

    Pipeline: measurements → feature levels → rules → stage scores → prediction.
    """

    def __init__(
        self,
        rules: tuple[NorwoodRule, ...] | None = None,
        level_evaluator: FeatureLevelEvaluator | None = None,
        min_classification_score: float = _DEFAULT_MIN_CLASSIFICATION_SCORE,
    ) -> None:
        if not 0.0 <= min_classification_score <= 1.0:
            raise ValueError("min_classification_score must be in [0, 1]")
        self._rules = rules if rules is not None else norwood_rules()
        self._levels = (
            level_evaluator
            if level_evaluator is not None
            else FeatureLevelEvaluator()
        )
        self.min_classification_score = min_classification_score

    def classify(self, features: dict) -> dict[str, Any]:
        """
        Score every Norwood stage and return a structured prediction.

        ``score`` is the weighted fraction of applicable semantic conditions
        matched for the best candidate (not an ML probability).
        """
        if not isinstance(features, dict):
            raise TypeError("features must be a dict")

        numeric = self._normalize_features(features)
        self._validate_required(numeric)
        levels = self._levels.evaluate(numeric)

        evaluations = [
            self._evaluate_rule(rule, levels, numeric) for rule in self._rules
        ]
        ranked = sorted(
            evaluations,
            key=lambda item: (item["score"], item["matched_count"]),
            reverse=True,
        )
        best = ranked[0]
        all_stage_scores = {
            item["stage"]: item["score"] for item in ranked
        }

        status: ClassificationStatus
        if best["score"] >= self.min_classification_score:
            status = "matched"
            stage = best["stage"]
            reason = None
        else:
            status = "inconclusive"
            stage = "Inconclusive"
            reason = "No Norwood stage matched strongly enough."

        return {
            "stage": stage,
            "matched_rule": best["matched_rule"],
            "score": best["score"],
            "best_candidate": best["stage"],
            "classification_status": status,
            "all_stage_scores": all_stage_scores,
            "reasoning": {
                "matched": best["matched"],
                "unmatched": best["unmatched"],
            },
            "feature_levels": {
                key: level.name for key, level in levels.items()
            },
            "measurements": dict(features),
            **({"reason": reason} if reason is not None else {}),
        }

    def _normalize_features(self, features: dict) -> dict[str, float]:
        """Keep numeric measurements; derive temple_recession when possible."""
        normalized: dict[str, float] = {}
        for key, value in features.items():
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                normalized[key] = float(value)

        if "coverage_ratio" not in normalized and "segmentation_coverage" in normalized:
            normalized["coverage_ratio"] = normalized["segmentation_coverage"]

        left = normalized.get("left_temple_recession")
        right = normalized.get("right_temple_recession")
        if left is not None or right is not None:
            normalized["temple_recession"] = max(
                v for v in (left, right) if v is not None
            )
        return normalized

    def _validate_required(self, features: dict[str, float]) -> None:
        """Ensure core common measurements are present."""
        missing = [key for key in _REQUIRED_FEATURES if key not in features]
        if missing:
            raise KeyError(f"missing required feature(s): {', '.join(missing)}")

    def _evaluate_rule(
        self,
        rule: NorwoodRule,
        levels: dict[str, FeatureLevel],
        numeric: dict[str, float],
    ) -> dict[str, Any]:
        """Score one Norwood rule; collect matched and unmatched explanations."""
        matched_weight = 0.0
        total_weight = 0.0
        matched_count = 0
        matched: list[str] = []
        unmatched: list[str] = []

        for condition in rule.conditions:
            label = self._levels.label_for(condition.feature)

            if condition.feature not in levels:
                if condition.optional:
                    continue
                total_weight += condition.weight
                expected = self._format_expected(condition.accepted_levels)
                unmatched.append(
                    f"✗ {label} expected {expected} but feature was unavailable"
                )
                continue

            level = levels[condition.feature]
            total_weight += condition.weight
            value = numeric.get(condition.feature)
            value_text = (
                format_measurement(condition.feature, value)
                if value is not None
                else "n/a"
            )

            if level in condition.accepted_levels:
                matched_weight += condition.weight
                matched_count += 1
                matched.append(f"✓ {label} is {level.name} ({value_text})")
            else:
                expected = self._format_expected(condition.accepted_levels)
                unmatched.append(
                    f"✗ {label} expected {expected} but observed {level.name}"
                )

        score = matched_weight / total_weight if total_weight > 0 else 0.0
        return {
            "stage": rule.stage,
            "matched_rule": rule.name,
            "score": round(float(score), 4),
            "matched_count": matched_count,
            "matched": matched,
            "unmatched": unmatched,
        }

    def _format_expected(self, accepted: tuple[FeatureLevel, ...]) -> str:
        """Join accepted levels for unmatched reasoning (e.g. HIGH or MEDIUM)."""
        names = [level.name for level in accepted]
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} or {names[1]}"
        return ", ".join(names[:-1]) + f", or {names[-1]}"
