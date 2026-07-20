"""Declarative Norwood rule definitions (semantic FeatureLevel only)."""

from __future__ import annotations

from dataclasses import dataclass

from .feature_levels import FeatureLevel


@dataclass(frozen=True)
class RuleCondition:
    """
    One semantic condition that contributes to a stage score.

    Matches when the evaluated feature level is in ``accepted_levels``.
    """

    feature: str
    accepted_levels: tuple[FeatureLevel, ...]
    weight: float = 1.0
    optional: bool = False


@dataclass(frozen=True)
class NorwoodRule:
    """Named Norwood stage with weighted semantic conditions."""

    stage: str
    name: str
    conditions: tuple[RuleCondition, ...]


def norwood_rules() -> tuple[NorwoodRule, ...]:
    """
    Norwood I–VI rules expressed over FeatureLevel values.

    Norwood VII is intentionally omitted: front/crown views alone cannot
    reliably separate VI from VII (needs side/posterior scalp evidence).
    """
    return (
        NorwoodRule(
            stage="Norwood I",
            name="minimal_or_no_recession",
            conditions=(
                RuleCondition("hair_density", (FeatureLevel.HIGH,)),
                RuleCondition("scalp_exposure", (FeatureLevel.LOW,)),
                RuleCondition(
                    "temple_recession", (FeatureLevel.LOW,), weight=0.8, optional=True
                ),
                RuleCondition(
                    "hairline_height",
                    (FeatureLevel.NORMAL,),
                    weight=0.7,
                    optional=True,
                ),
                RuleCondition(
                    "largest_bald_region_percentage",
                    (FeatureLevel.SMALL,),
                    weight=0.7,
                    optional=True,
                ),
                RuleCondition(
                    "crown_scalp_exposure",
                    (FeatureLevel.LOW,),
                    weight=0.7,
                    optional=True,
                ),
            ),
        ),
        NorwoodRule(
            stage="Norwood II",
            name="slight_temple_recession",
            conditions=(
                RuleCondition(
                    "hair_density", (FeatureLevel.HIGH, FeatureLevel.MEDIUM)
                ),
                RuleCondition(
                    "scalp_exposure", (FeatureLevel.LOW, FeatureLevel.MEDIUM)
                ),
                RuleCondition(
                    "temple_recession",
                    (FeatureLevel.MODERATE,),
                    weight=1.0,
                    optional=True,
                ),
                RuleCondition(
                    "hairline_height",
                    (FeatureLevel.NORMAL, FeatureLevel.ELEVATED),
                    weight=0.6,
                    optional=True,
                ),
                RuleCondition(
                    "largest_bald_region_percentage",
                    (FeatureLevel.SMALL,),
                    weight=0.6,
                    optional=True,
                ),
                RuleCondition(
                    "crown_scalp_exposure",
                    (FeatureLevel.LOW, FeatureLevel.MEDIUM),
                    weight=0.5,
                    optional=True,
                ),
            ),
        ),
        NorwoodRule(
            stage="Norwood III",
            name="deeper_frontal_temple_recession",
            conditions=(
                RuleCondition(
                    "hair_density", (FeatureLevel.MEDIUM, FeatureLevel.LOW)
                ),
                RuleCondition(
                    "scalp_exposure", (FeatureLevel.MEDIUM, FeatureLevel.HIGH)
                ),
                RuleCondition(
                    "temple_recession",
                    (FeatureLevel.MODERATE, FeatureLevel.HIGH),
                    weight=1.0,
                    optional=True,
                ),
                RuleCondition(
                    "hairline_height",
                    (FeatureLevel.ELEVATED, FeatureLevel.HIGH),
                    weight=0.8,
                    optional=True,
                ),
                RuleCondition(
                    "largest_bald_region_percentage",
                    (FeatureLevel.SMALL, FeatureLevel.MEDIUM),
                    weight=0.5,
                    optional=True,
                ),
            ),
        ),
        NorwoodRule(
            stage="Norwood III Vertex",
            name="crown_thinning_with_milder_front",
            conditions=(
                RuleCondition(
                    "crown_scalp_exposure",
                    (FeatureLevel.MEDIUM, FeatureLevel.HIGH),
                    optional=True,
                ),
                RuleCondition(
                    "largest_bald_region_percentage",
                    (FeatureLevel.MEDIUM, FeatureLevel.LARGE),
                    optional=True,
                ),
                RuleCondition(
                    "hair_density",
                    (FeatureLevel.MEDIUM, FeatureLevel.HIGH),
                    weight=0.7,
                ),
                RuleCondition(
                    "scalp_exposure",
                    (FeatureLevel.LOW, FeatureLevel.MEDIUM),
                    weight=0.6,
                ),
                RuleCondition(
                    "temple_recession",
                    (FeatureLevel.LOW, FeatureLevel.MODERATE),
                    weight=0.6,
                    optional=True,
                ),
            ),
        ),
        NorwoodRule(
            stage="Norwood IV",
            name="frontal_and_crown_separation",
            conditions=(
                RuleCondition(
                    "hair_density", (FeatureLevel.LOW, FeatureLevel.MEDIUM)
                ),
                RuleCondition(
                    "scalp_exposure", (FeatureLevel.MEDIUM, FeatureLevel.HIGH)
                ),
                RuleCondition(
                    "crown_scalp_exposure",
                    (FeatureLevel.MEDIUM, FeatureLevel.HIGH),
                    optional=True,
                ),
                RuleCondition(
                    "largest_bald_region_percentage",
                    (FeatureLevel.MEDIUM, FeatureLevel.LARGE),
                    optional=True,
                ),
                RuleCondition(
                    "temple_recession",
                    (FeatureLevel.MODERATE, FeatureLevel.HIGH),
                    weight=0.7,
                    optional=True,
                ),
            ),
        ),
        NorwoodRule(
            stage="Norwood V",
            name="bridging_frontal_and_crown_loss",
            conditions=(
                RuleCondition(
                    "hair_density", (FeatureLevel.LOW, FeatureLevel.VERY_LOW)
                ),
                RuleCondition(
                    "scalp_exposure", (FeatureLevel.HIGH, FeatureLevel.MEDIUM)
                ),
                RuleCondition(
                    "crown_scalp_exposure",
                    (FeatureLevel.HIGH, FeatureLevel.MEDIUM),
                    optional=True,
                ),
                RuleCondition(
                    "largest_bald_region_percentage",
                    (FeatureLevel.LARGE, FeatureLevel.MEDIUM),
                    optional=True,
                ),
                RuleCondition(
                    "temple_recession",
                    (FeatureLevel.HIGH, FeatureLevel.MODERATE),
                    weight=0.6,
                    optional=True,
                ),
            ),
        ),
        NorwoodRule(
            stage="Norwood VI",
            name="extensive_frontal_crown_confluence",
            conditions=(
                RuleCondition(
                    "hair_density", (FeatureLevel.VERY_LOW, FeatureLevel.LOW)
                ),
                RuleCondition("scalp_exposure", (FeatureLevel.HIGH,)),
                RuleCondition(
                    "crown_scalp_exposure", (FeatureLevel.HIGH,), optional=True
                ),
                RuleCondition(
                    "largest_bald_region_percentage",
                    (FeatureLevel.LARGE,),
                    optional=True,
                ),
                RuleCondition(
                    "temple_recession",
                    (FeatureLevel.HIGH,),
                    weight=0.5,
                    optional=True,
                ),
                RuleCondition(
                    "hairline_height",
                    (FeatureLevel.HIGH,),
                    weight=0.5,
                    optional=True,
                ),
            ),
        ),
    )
