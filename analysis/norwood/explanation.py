"""Build evidence, explanation, limitations, and recommendations."""

from __future__ import annotations

from analysis.metrics.types import AnalysisMode, HairLossMetrics
from analysis.norwood.types import (
    ConfidenceResult,
    ExplanationBundle,
    NorwoodStage,
    RuleMatch,
)

_STAGE_RECOMMENDATIONS: dict[NorwoodStage, str] = {
    NorwoodStage.I: (
        "Hairline pattern appears within the early / minimal range. "
        "Periodic monitoring and lifestyle factors that support scalp health "
        "may be useful."
    ),
    NorwoodStage.II: (
        "Mild temporal or frontal recession is suggested. Early consultation "
        "with a clinician can clarify options for monitoring or intervention."
    ),
    NorwoodStage.III: (
        "Moderate frontal or temporal recession is suggested. A specialist "
        "assessment can help review medical and procedural options."
    ),
    NorwoodStage.III_VERTEX: (
        "Crown / vertex thinning appears more prominent than frontal loss. "
        "A clinician can advise on crown-focused evaluation and care plans."
    ),
    NorwoodStage.IV_PLUS: (
        "Findings are consistent with advanced combined hair loss (Norwood "
        "IV–VII range). A professional clinical assessment is recommended "
        "to refine staging and discuss management options."
    ),
}

_STAGE_EXPLANATIONS: dict[NorwoodStage, str] = {
    NorwoodStage.I: (
        "Rule {rule_id} matched because available regional loss metrics "
        "remain below mild-recession thresholds, indicating minimal "
        "patterned hair loss."
    ),
    NorwoodStage.II: (
        "Rule {rule_id} matched because mild frontal and/or temple loss was "
        "detected without reaching moderate Norwood III thresholds."
    ),
    NorwoodStage.III: (
        "Rule {rule_id} matched because frontal and/or temple loss reached "
        "moderate levels typical of classic Norwood III recession."
    ),
    NorwoodStage.III_VERTEX: (
        "Rule {rule_id} matched because crown / vertex loss is the dominant "
        "available finding, consistent with Norwood III Vertex."
    ),
    NorwoodStage.IV_PLUS: (
        "Rule {rule_id} matched because overall severity and/or multi-zone "
        "loss indicate advanced patterned baldness spanning Norwood IV–VII. "
        "Finer stages within that range are not differentiated in this MVP."
    ),
}


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}%"


def build_evidence(metrics: HairLossMetrics) -> tuple[str, ...]:
    """Objective findings from metrics (independent of stage)."""
    lines = [
        f"Front loss: {_fmt(metrics.front_loss_percentage)}",
        f"Left temple loss: {_fmt(metrics.left_temple_loss_percentage)}",
        f"Right temple loss: {_fmt(metrics.right_temple_loss_percentage)}",
        f"Crown loss: {_fmt(metrics.crown_loss_percentage)}",
        f"Overall loss: {_fmt(metrics.overall_hair_loss_percentage)}",
        f"Overall coverage: {_fmt(metrics.overall_hair_coverage_percentage)}",
        f"Component count: {metrics.component_count}",
        (
            f"Largest deficit zone: "
            f"{metrics.largest_deficit_zone or 'n/a'} "
            f"({_fmt(metrics.largest_deficit_percentage)})"
        ),
        f"Analysis mode: {metrics.analysis_mode.value}",
    ]
    return tuple(lines)


def build_limitations(
    metrics: HairLossMetrics,
    match: RuleMatch,
    confidence: ConfidenceResult,
) -> tuple[str, ...]:
    """Contextual caveats for the classification."""
    items: list[str] = []

    if metrics.analysis_mode == AnalysisMode.FRONT_ONLY:
        items.append(
            "Front-only analysis: crown / vertex loss was not measured."
        )
    elif metrics.analysis_mode == AnalysisMode.CROWN_ONLY:
        items.append(
            "Crown-only analysis: frontal and temple loss were not measured."
        )

    if match.stage == NorwoodStage.IV_PLUS:
        items.append(
            "Advanced stages IV–VII cannot be fully differentiated with "
            "the current anatomical feature set; reported as IV+."
        )

    if "near_decision_boundary" in confidence.penalties:
        items.append(
            "Metrics are near a decision boundary; adjacent stages remain "
            "plausible."
        )

    if "overall_metric_conflict" in confidence.penalties:
        items.append(
            "Overall loss severity is partially inconsistent with the "
            "selected regional pattern."
        )

    if not items:
        items.append("No major analysis limitations identified for this result.")

    return tuple(items)


def build_explanation(match: RuleMatch) -> str:
    """Concise description of why the selected rule fired."""
    template = _STAGE_EXPLANATIONS[match.stage]
    return template.format(rule_id=match.rule_id)


def build_recommendation(match: RuleMatch) -> str:
    """Short informational recommendation for the matched stage."""
    return _STAGE_RECOMMENDATIONS[match.stage]


class NorwoodExplanationBuilder:
    """Assemble evidence, explanation, limitations, and recommendation."""

    def build(
        self,
        metrics: HairLossMetrics,
        match: RuleMatch,
        confidence: ConfidenceResult,
    ) -> ExplanationBundle:
        return ExplanationBundle(
            evidence=build_evidence(metrics),
            explanation=build_explanation(match),
            limitations=build_limitations(metrics, match, confidence),
            recommendation=build_recommendation(match),
        )
