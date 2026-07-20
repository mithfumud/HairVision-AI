"""Types for Norwood stage classification from HairLossMetrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from analysis.metrics.types import AnalysisMode


class NorwoodStage(str, Enum):
    """MVP Norwood stages supported by the rule engine."""

    I = "I"
    II = "II"
    III = "III"
    III_VERTEX = "III_Vertex"
    IV_PLUS = "IV+"


class ConfidenceBand(str, Enum):
    """Qualitative band derived from numeric confidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class RuleMatch:
    """Outcome of the deterministic rule engine (stage + rule id only)."""

    stage: NorwoodStage
    rule_id: str


@dataclass(frozen=True)
class ConfidenceResult:
    """Confidence score and applied penalties (never changes stage)."""

    confidence: float
    confidence_band: ConfidenceBand
    penalties: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationBundle:
    """Human-readable evidence, explanation, limitations, and recommendation."""

    evidence: tuple[str, ...]
    explanation: str
    limitations: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True)
class NorwoodClassificationResult:
    """
    Complete Norwood classification output.

    Produced by orchestrating rules → confidence → explanation. Purely
    derived from ``HairLossMetrics``; no image processing.
    """

    stage: NorwoodStage
    confidence: float
    confidence_band: ConfidenceBand
    analysis_mode: AnalysisMode
    rule_id: str
    evidence: tuple[str, ...]
    explanation: str
    limitations: tuple[str, ...]
    recommendation: str
    flags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "stage": self.stage.value,
            "confidence": round(self.confidence, 2),
            "confidence_band": self.confidence_band.value,
            "analysis_mode": self.analysis_mode.value,
            "rule_id": self.rule_id,
            "evidence": list(self.evidence),
            "explanation": self.explanation,
            "limitations": list(self.limitations),
            "recommendation": self.recommendation,
            "flags": list(self.flags),
        }
