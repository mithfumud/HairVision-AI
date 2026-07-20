"""Unit tests for the Norwood classification package."""

from __future__ import annotations

import unittest

from analysis.metrics.types import AnalysisMode, HairLossMetrics
from analysis.norwood.classifier import NorwoodClassifier
from analysis.norwood.confidence import NorwoodConfidenceEngine
from analysis.norwood.explanation import NorwoodExplanationBuilder, build_evidence
from analysis.norwood.rules import NorwoodRuleEngine
from analysis.norwood.types import (
    ConfidenceBand,
    NorwoodStage,
    RuleMatch,
)


def _metrics(
    *,
    front: float | None = None,
    left: float | None = None,
    right: float | None = None,
    crown: float | None = None,
    overall: float | None = 0.0,
    coverage: float | None = 100.0,
    component_count: int = 0,
    largest_zone: str | None = None,
    largest_pct: float | None = None,
    mode: AnalysisMode = AnalysisMode.FRONT_ONLY,
) -> HairLossMetrics:
    front_available = any(v is not None for v in (front, left, right))
    crown_available = crown is not None
    if mode == AnalysisMode.FRONT_ONLY:
        front_available = True
        crown_available = False
        crown = None
    elif mode == AnalysisMode.CROWN_ONLY:
        front_available = False
        crown_available = True
        front = left = right = None
    else:
        front_available = True
        crown_available = True

    return HairLossMetrics(
        front_loss_percentage=front,
        left_temple_loss_percentage=left,
        right_temple_loss_percentage=right,
        crown_loss_percentage=crown,
        overall_hair_loss_percentage=overall,
        overall_hair_coverage_percentage=coverage,
        component_count=component_count,
        largest_deficit_zone=largest_zone,  # type: ignore[arg-type]
        largest_deficit_percentage=largest_pct,
        analysis_mode=mode,
        front_available=front_available,
        crown_available=crown_available,
    )


class NorwoodRuleEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = NorwoodRuleEngine()

    def test_stage_i_minimal_loss(self) -> None:
        match = self.engine.evaluate(
            _metrics(front=2.0, left=3.0, right=2.0, overall=2.0, mode=AnalysisMode.FRONT_ONLY)
        )
        self.assertEqual(match.stage, NorwoodStage.I)
        self.assertEqual(match.rule_id, "NW-I-01")

    def test_stage_ii_mild_temple(self) -> None:
        match = self.engine.evaluate(
            _metrics(front=8.0, left=15.0, right=14.0, overall=10.0, mode=AnalysisMode.FRONT_ONLY)
        )
        self.assertEqual(match.stage, NorwoodStage.II)
        self.assertEqual(match.rule_id, "NW-II-01")

    def test_stage_iii_moderate_temple(self) -> None:
        match = self.engine.evaluate(
            _metrics(front=18.0, left=30.0, right=28.0, overall=20.0, mode=AnalysisMode.FRONT_ONLY)
        )
        self.assertEqual(match.stage, NorwoodStage.III)
        self.assertEqual(match.rule_id, "NW-III-01")

    def test_stage_iii_vertex_crown_dominant(self) -> None:
        match = self.engine.evaluate(
            _metrics(
                front=12.0,
                left=10.0,
                right=10.0,
                crown=35.0,
                overall=18.0,
                mode=AnalysisMode.COMBINED,
                largest_zone="crown",
                largest_pct=35.0,
            )
        )
        self.assertEqual(match.stage, NorwoodStage.III_VERTEX)
        self.assertEqual(match.rule_id, "NW-IIIV-01")

    def test_stage_iv_plus_combined_advanced(self) -> None:
        match = self.engine.evaluate(
            _metrics(
                front=45.0,
                left=50.0,
                right=48.0,
                crown=42.0,
                overall=40.0,
                mode=AnalysisMode.COMBINED,
            )
        )
        self.assertEqual(match.stage, NorwoodStage.IV_PLUS)
        self.assertEqual(match.rule_id, "NW-IVPLUS-01")

    def test_front_only_mode_cannot_be_iii_vertex(self) -> None:
        match = self.engine.evaluate(
            _metrics(front=15.0, left=20.0, right=18.0, overall=16.0, mode=AnalysisMode.FRONT_ONLY)
        )
        self.assertNotEqual(match.stage, NorwoodStage.III_VERTEX)

    def test_crown_only_moderate_is_iii_vertex(self) -> None:
        match = self.engine.evaluate(
            _metrics(crown=30.0, overall=30.0, mode=AnalysisMode.CROWN_ONLY, largest_zone="crown")
        )
        self.assertEqual(match.stage, NorwoodStage.III_VERTEX)
        self.assertEqual(match.rule_id, "NW-IIIV-01")

    def test_crown_only_advanced_is_iv_plus(self) -> None:
        match = self.engine.evaluate(
            _metrics(crown=55.0, overall=55.0, mode=AnalysisMode.CROWN_ONLY)
        )
        self.assertEqual(match.stage, NorwoodStage.IV_PLUS)
        self.assertEqual(match.rule_id, "NW-IVPLUS-01")

    def test_exactly_one_rule_fires(self) -> None:
        cases = [
            _metrics(front=1.0, left=1.0, right=1.0, overall=1.0),
            _metrics(front=8.0, left=15.0, right=12.0, overall=10.0),
            _metrics(front=25.0, left=30.0, right=28.0, overall=22.0),
            _metrics(
                front=10.0,
                left=8.0,
                right=8.0,
                crown=40.0,
                overall=15.0,
                mode=AnalysisMode.COMBINED,
            ),
            _metrics(
                front=50.0,
                left=50.0,
                right=50.0,
                crown=50.0,
                overall=50.0,
                mode=AnalysisMode.COMBINED,
            ),
        ]
        for metrics in cases:
            match = self.engine.evaluate(metrics)
            self.assertIsInstance(match, RuleMatch)
            self.assertTrue(match.rule_id.startswith("NW-"))


class NorwoodConfidenceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = NorwoodConfidenceEngine()

    def test_starts_near_100_for_clean_front(self) -> None:
        metrics = _metrics(front=2.0, left=2.0, right=2.0, overall=2.0)
        match = RuleMatch(stage=NorwoodStage.I, rule_id="NW-I-01")
        result = self.engine.evaluate(metrics, match)
        self.assertLessEqual(result.confidence, 100.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertIn("front_only_analysis", result.penalties)
        self.assertEqual(result.confidence, 88.0)  # 100 - 12

    def test_combined_mode_no_mode_penalty(self) -> None:
        metrics = _metrics(
            front=2.0,
            left=2.0,
            right=2.0,
            crown=1.0,
            overall=2.0,
            mode=AnalysisMode.COMBINED,
        )
        match = RuleMatch(stage=NorwoodStage.I, rule_id="NW-I-01")
        result = self.engine.evaluate(metrics, match)
        self.assertNotIn("front_only_analysis", result.penalties)
        self.assertNotIn("crown_only_analysis", result.penalties)
        self.assertEqual(result.confidence_band, ConfidenceBand.HIGH)

    def test_never_changes_stage(self) -> None:
        metrics = _metrics(front=50.0, left=50.0, right=50.0, overall=10.0)
        match = RuleMatch(stage=NorwoodStage.IV_PLUS, rule_id="NW-IVPLUS-01")
        before = match.stage
        self.engine.evaluate(metrics, match)
        self.assertEqual(match.stage, before)

    def test_confidence_clamped(self) -> None:
        metrics = _metrics(crown=55.0, overall=10.0, mode=AnalysisMode.CROWN_ONLY)
        match = RuleMatch(stage=NorwoodStage.IV_PLUS, rule_id="NW-IVPLUS-01")
        result = self.engine.evaluate(metrics, match)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 100.0)

    def test_near_boundary_penalty(self) -> None:
        # Temple just at mild threshold window for I/II boundary.
        metrics = _metrics(front=2.0, left=12.0, right=11.0, overall=5.0)
        match = RuleMatch(stage=NorwoodStage.II, rule_id="NW-II-01")
        result = self.engine.evaluate(metrics, match)
        self.assertIn("near_decision_boundary", result.penalties)


class NorwoodExplanationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = NorwoodExplanationBuilder()
        self.confidence = NorwoodConfidenceEngine()

    def test_evidence_contains_key_metrics(self) -> None:
        metrics = _metrics(front=10.0, left=12.0, right=11.0, overall=9.0)
        evidence = build_evidence(metrics)
        joined = "\n".join(evidence)
        self.assertIn("Front loss:", joined)
        self.assertIn("Left temple loss:", joined)
        self.assertIn("Right temple loss:", joined)
        self.assertIn("Crown loss:", joined)
        self.assertIn("Overall loss:", joined)
        self.assertIn("Largest deficit zone:", joined)

    def test_explanation_mentions_rule_id(self) -> None:
        metrics = _metrics(front=2.0, left=2.0, right=2.0, overall=2.0)
        match = RuleMatch(stage=NorwoodStage.I, rule_id="NW-I-01")
        conf = self.confidence.evaluate(metrics, match)
        bundle = self.builder.build(metrics, match, conf)
        self.assertIn("NW-I-01", bundle.explanation)

    def test_recommendation_is_non_empty_for_all_stages(self) -> None:
        for stage, rule_id in (
            (NorwoodStage.I, "NW-I-01"),
            (NorwoodStage.II, "NW-II-01"),
            (NorwoodStage.III, "NW-III-01"),
            (NorwoodStage.III_VERTEX, "NW-IIIV-01"),
            (NorwoodStage.IV_PLUS, "NW-IVPLUS-01"),
        ):
            metrics = _metrics(front=1.0, left=1.0, right=1.0, overall=1.0)
            match = RuleMatch(stage=stage, rule_id=rule_id)
            conf = self.confidence.evaluate(metrics, match)
            bundle = self.builder.build(metrics, match, conf)
            self.assertTrue(len(bundle.recommendation) > 20)

    def test_front_only_limitation(self) -> None:
        metrics = _metrics(front=2.0, left=2.0, right=2.0, overall=2.0)
        match = RuleMatch(stage=NorwoodStage.I, rule_id="NW-I-01")
        conf = self.confidence.evaluate(metrics, match)
        bundle = self.builder.build(metrics, match, conf)
        self.assertTrue(any("Front-only" in item for item in bundle.limitations))

    def test_iv_plus_limitation(self) -> None:
        metrics = _metrics(
            front=50.0,
            left=50.0,
            right=50.0,
            crown=50.0,
            overall=50.0,
            mode=AnalysisMode.COMBINED,
        )
        match = RuleMatch(stage=NorwoodStage.IV_PLUS, rule_id="NW-IVPLUS-01")
        conf = self.confidence.evaluate(metrics, match)
        bundle = self.builder.build(metrics, match, conf)
        self.assertTrue(any("IV–VII" in item or "IV-VII" in item for item in bundle.limitations))


class NorwoodClassifierIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = NorwoodClassifier()

    def test_end_to_end_stage_i(self) -> None:
        result = self.classifier.classify(
            _metrics(front=1.0, left=2.0, right=1.0, overall=1.5)
        )
        self.assertEqual(result.stage, NorwoodStage.I)
        self.assertEqual(result.rule_id, "NW-I-01")
        self.assertEqual(result.analysis_mode, AnalysisMode.FRONT_ONLY)
        self.assertTrue(result.evidence)
        self.assertTrue(result.explanation)
        self.assertTrue(result.recommendation)
        self.assertTrue(result.limitations)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 100.0)

    def test_end_to_end_stage_ii(self) -> None:
        result = self.classifier.classify(
            _metrics(front=8.0, left=16.0, right=15.0, overall=11.0)
        )
        self.assertEqual(result.stage, NorwoodStage.II)
        self.assertEqual(result.rule_id, "NW-II-01")

    def test_end_to_end_stage_iii(self) -> None:
        result = self.classifier.classify(
            _metrics(front=20.0, left=32.0, right=30.0, overall=22.0)
        )
        self.assertEqual(result.stage, NorwoodStage.III)
        self.assertEqual(result.rule_id, "NW-III-01")

    def test_end_to_end_iii_vertex_combined(self) -> None:
        result = self.classifier.classify(
            _metrics(
                front=10.0,
                left=8.0,
                right=8.0,
                crown=38.0,
                overall=16.0,
                mode=AnalysisMode.COMBINED,
                largest_zone="crown",
                largest_pct=38.0,
            )
        )
        self.assertEqual(result.stage, NorwoodStage.III_VERTEX)
        self.assertEqual(result.rule_id, "NW-IIIV-01")
        self.assertEqual(result.analysis_mode, AnalysisMode.COMBINED)

    def test_end_to_end_iv_plus(self) -> None:
        result = self.classifier.classify(
            _metrics(
                front=48.0,
                left=52.0,
                right=50.0,
                crown=45.0,
                overall=42.0,
                mode=AnalysisMode.COMBINED,
            )
        )
        self.assertEqual(result.stage, NorwoodStage.IV_PLUS)
        self.assertEqual(result.rule_id, "NW-IVPLUS-01")

    def test_to_dict_keys(self) -> None:
        result = self.classifier.classify(
            _metrics(front=1.0, left=1.0, right=1.0, overall=1.0)
        )
        payload = result.to_dict()
        for key in (
            "stage",
            "confidence",
            "confidence_band",
            "analysis_mode",
            "rule_id",
            "evidence",
            "explanation",
            "limitations",
            "recommendation",
            "flags",
        ):
            self.assertIn(key, payload)

    def test_rejects_non_metrics(self) -> None:
        with self.assertRaises(TypeError):
            self.classifier.classify({"front_loss_percentage": 1.0})  # type: ignore[arg-type]


class NorwoodValidationAuditTests(unittest.TestCase):
    """Regression tests from the pre-integration validation audit."""

    def setUp(self) -> None:
        self.classifier = NorwoodClassifier()
        self.engine = NorwoodRuleEngine()

    def test_rule_ids_are_unique(self) -> None:
        from analysis.norwood.rules import _RULES

        ids = [rule_id for rule_id, _, _ in _RULES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_severe_front_and_crown_never_maps_to_stage_i(self) -> None:
        """
        Audit bug: front=45 + crown=50 + overall=30 previously fell through
        every rule and landed on Stage I with a contradictory explanation.
        """
        result = self.classifier.classify(
            _metrics(
                front=45.0,
                left=10.0,
                right=10.0,
                crown=50.0,
                overall=30.0,
                mode=AnalysisMode.COMBINED,
            )
        )
        self.assertNotEqual(result.stage, NorwoodStage.I)
        self.assertEqual(result.stage, NorwoodStage.IV_PLUS)
        self.assertEqual(result.rule_id, "NW-IVPLUS-01")
        self.assertNotIn("below mild-recession thresholds", result.explanation)

    def test_stage_i_explanation_does_not_claim_unmeasured_crown(self) -> None:
        result = self.classifier.classify(
            _metrics(front=1.0, left=1.0, right=1.0, overall=1.0, mode=AnalysisMode.FRONT_ONLY)
        )
        self.assertEqual(result.stage, NorwoodStage.I)
        self.assertNotIn("measured frontal, temple, and crown", result.explanation)

    def test_crown_only_low_loss_stage_i_explanation_is_mode_safe(self) -> None:
        result = self.classifier.classify(
            _metrics(crown=5.0, overall=5.0, mode=AnalysisMode.CROWN_ONLY)
        )
        self.assertEqual(result.stage, NorwoodStage.I)
        self.assertNotIn("measured frontal, temple, and crown", result.explanation)

    def test_asymmetric_left_temple_only(self) -> None:
        result = self.classifier.classify(
            _metrics(front=0.0, left=100.0, right=0.0, overall=20.0)
        )
        self.assertEqual(result.stage, NorwoodStage.III)

    def test_asymmetric_right_temple_only(self) -> None:
        result = self.classifier.classify(
            _metrics(front=0.0, left=0.0, right=100.0, overall=20.0)
        )
        self.assertEqual(result.stage, NorwoodStage.III)

    def test_all_metrics_zero(self) -> None:
        result = self.classifier.classify(
            _metrics(
                front=0.0,
                left=0.0,
                right=0.0,
                crown=0.0,
                overall=0.0,
                mode=AnalysisMode.COMBINED,
            )
        )
        self.assertEqual(result.stage, NorwoodStage.I)
        self.assertEqual(result.confidence_band, ConfidenceBand.HIGH)

    def test_all_metrics_100_combined(self) -> None:
        result = self.classifier.classify(
            _metrics(
                front=100.0,
                left=100.0,
                right=100.0,
                crown=100.0,
                overall=100.0,
                mode=AnalysisMode.COMBINED,
            )
        )
        self.assertEqual(result.stage, NorwoodStage.IV_PLUS)

    def test_determinism(self) -> None:
        metrics = _metrics(front=15.0, left=20.0, right=18.0, overall=16.0)
        first = self.classifier.classify(metrics).to_dict()
        for _ in range(10):
            self.assertEqual(self.classifier.classify(metrics).to_dict(), first)

    def test_confidence_never_alters_stage_across_modes(self) -> None:
        cases = [
            _metrics(front=1.0, left=1.0, right=1.0, overall=1.0),
            _metrics(crown=30.0, overall=30.0, mode=AnalysisMode.CROWN_ONLY),
            _metrics(
                front=50.0,
                left=50.0,
                right=50.0,
                crown=50.0,
                overall=50.0,
                mode=AnalysisMode.COMBINED,
            ),
        ]
        for metrics in cases:
            match = self.engine.evaluate(metrics)
            result = self.classifier.classify(metrics)
            self.assertEqual(result.stage, match.stage)
            self.assertEqual(result.rule_id, match.rule_id)

    def test_output_fields_always_populated(self) -> None:
        result = self.classifier.classify(
            _metrics(front=1.0, left=1.0, right=1.0, overall=1.0)
        )
        self.assertIsNotNone(result.stage)
        self.assertIsNotNone(result.confidence_band)
        self.assertIsNotNone(result.analysis_mode)
        self.assertTrue(result.rule_id)
        self.assertTrue(result.evidence)
        self.assertTrue(result.explanation)
        self.assertTrue(result.limitations)
        self.assertTrue(result.recommendation)
        self.assertIsInstance(result.flags, tuple)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 100.0)

    def test_overall_none_does_not_crash(self) -> None:
        result = self.classifier.classify(
            _metrics(
                front=50.0,
                left=50.0,
                right=50.0,
                crown=50.0,
                overall=None,
                coverage=None,
                mode=AnalysisMode.COMBINED,
            )
        )
        self.assertEqual(result.stage, NorwoodStage.IV_PLUS)

    def test_temple_crown_severe_without_overall_does_not_map_to_i(self) -> None:
        """Safety-net path: unmatched severe pattern must not become Stage I."""
        result = self.classifier.classify(
            _metrics(
                front=30.0,
                left=50.0,
                right=50.0,
                crown=50.0,
                overall=20.0,
                mode=AnalysisMode.COMBINED,
            )
        )
        self.assertNotEqual(result.stage, NorwoodStage.I)


if __name__ == "__main__":
    unittest.main()
