"""Orchestrate Norwood classification from HairLossMetrics."""

from __future__ import annotations

from analysis.metrics.types import HairLossMetrics
from analysis.norwood.confidence import NorwoodConfidenceEngine
from analysis.norwood.explanation import NorwoodExplanationBuilder
from analysis.norwood.rules import NorwoodRuleEngine
from analysis.norwood.types import NorwoodClassificationResult


class NorwoodClassifier:
    """
    Pipeline: metrics → rules → confidence → explanation → result.

    Pure consumer of ``HairLossMetrics``. Does not modify metrics or run
    image processing.
    """

    def __init__(
        self,
        *,
        rule_engine: NorwoodRuleEngine | None = None,
        confidence_engine: NorwoodConfidenceEngine | None = None,
        explanation_builder: NorwoodExplanationBuilder | None = None,
    ) -> None:
        self._rules = rule_engine or NorwoodRuleEngine()
        self._confidence = confidence_engine or NorwoodConfidenceEngine()
        self._explanation = explanation_builder or NorwoodExplanationBuilder()

    def classify(self, metrics: HairLossMetrics) -> NorwoodClassificationResult:
        """
        Classify patterned hair loss from extracted clinical metrics.

        Flow:
            1. Rule engine selects exactly one stage / rule_id
            2. Confidence engine scores the match (never changes stage)
            3. Explanation builder produces evidence and narrative fields
        """
        if not isinstance(metrics, HairLossMetrics):
            raise TypeError("metrics must be a HairLossMetrics instance")

        match = self._rules.evaluate(metrics)
        confidence = self._confidence.evaluate(metrics, match)
        bundle = self._explanation.build(metrics, match, confidence)

        flags = confidence.penalties

        return NorwoodClassificationResult(
            stage=match.stage,
            confidence=confidence.confidence,
            confidence_band=confidence.confidence_band,
            analysis_mode=metrics.analysis_mode,
            rule_id=match.rule_id,
            evidence=bundle.evidence,
            explanation=bundle.explanation,
            limitations=bundle.limitations,
            recommendation=bundle.recommendation,
            flags=flags,
        )
