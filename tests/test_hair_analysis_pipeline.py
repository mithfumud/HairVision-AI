"""Unit tests for HairAnalysisPipeline orchestration (mocked algorithms)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from analysis.deficit.types import DeficitStatistics, HairDeficitResult
from analysis.metrics.types import AnalysisMode, HairLossMetrics
from analysis.normative_region.types import NormativeRegionResult, RegionMasks
from analysis.norwood.types import (
    ConfidenceBand,
    NorwoodClassificationResult,
    NorwoodStage,
)
from analysis.pipeline.pipeline import HairAnalysisPipeline, _merge_combined_metrics
from analysis.pipeline.report import HairAnalysisReport
from analysis.validation.view_validator import ViewValidationResult


def _blank(size: tuple[int, int] = (32, 32)) -> Image.Image:
    return Image.new("RGB", size, color=(120, 100, 80))


def _regions(
    shape: tuple[int, int] = (32, 32),
    *,
    front: bool = False,
    crown: bool = False,
) -> RegionMasks:
    empty = np.zeros(shape, dtype=bool)
    frontal = empty.copy()
    left = empty.copy()
    right = empty.copy()
    crown_mask = empty.copy()
    if front:
        frontal[:, shape[1] // 3 : 2 * shape[1] // 3] = True
        left[:, : shape[1] // 3] = True
        right[:, 2 * shape[1] // 3 :] = True
    if crown:
        crown_mask[8:24, 8:24] = True
    return RegionMasks(
        frontal=frontal,
        left_temple=left,
        right_temple=right,
        crown=crown_mask,
    )


def _normative(
    view: str,
    regions: RegionMasks,
) -> NormativeRegionResult:
    if view == "front":
        mask = regions.frontal | regions.left_temple | regions.right_temple
    else:
        mask = regions.crown.copy()
    return NormativeRegionResult(
        normative_region_mask=mask,
        region_masks=regions,
        view=view,  # type: ignore[arg-type]
        metadata={"source": "mock"},
    )


def _deficit(mask: np.ndarray) -> HairDeficitResult:
    area = int(np.count_nonzero(mask))
    normative = max(area, 100)
    return HairDeficitResult(
        deficit_mask=mask.astype(bool),
        components=(),
        largest_component=None,
        statistics=DeficitStatistics(
            total_deficit_area_px=area,
            deficit_percentage=float(area / normative),
            normative_area_px=normative,
            component_count=0,
            largest_component_area_px=0,
            raw_deficit_area_px=area,
        ),
    )


def _metrics(
    mode: AnalysisMode,
    *,
    front: float | None = 5.0,
    left: float | None = 5.0,
    right: float | None = 5.0,
    crown: float | None = None,
    overall: float = 5.0,
) -> HairLossMetrics:
    return HairLossMetrics(
        front_loss_percentage=front,
        left_temple_loss_percentage=left,
        right_temple_loss_percentage=right,
        crown_loss_percentage=crown,
        overall_hair_loss_percentage=overall,
        overall_hair_coverage_percentage=100.0 - overall,
        component_count=0,
        largest_deficit_zone=None,
        largest_deficit_percentage=None,
        analysis_mode=mode,
        front_available=mode in (AnalysisMode.FRONT_ONLY, AnalysisMode.COMBINED),
        crown_available=mode in (AnalysisMode.CROWN_ONLY, AnalysisMode.COMBINED),
    )


def _norwood(mode: AnalysisMode) -> NorwoodClassificationResult:
    return NorwoodClassificationResult(
        stage=NorwoodStage.I,
        confidence=88.0,
        confidence_band=ConfidenceBand.HIGH,
        analysis_mode=mode,
        rule_id="NW-I-01",
        evidence=("Front loss: 5.0%",),
        explanation="mock explanation",
        limitations=("mock limitation",),
        recommendation="mock recommendation",
        flags=(),
    )


def _view_ok(view: str) -> ViewValidationResult:
    return ViewValidationResult(
        is_valid=True,
        expected_view=view,  # type: ignore[arg-type]
        detected_view=view,  # type: ignore[arg-type]
        confidence=0.9,
        message="",
        reason="ok",
        scores={},
    )


def _view_fail(view: str) -> ViewValidationResult:
    return ViewValidationResult(
        is_valid=False,
        expected_view=view,  # type: ignore[arg-type]
        detected_view="unknown",
        confidence=0.2,
        message="invalid view",
        reason="mismatch",
        scores={},
    )


def _seg(shape: tuple[int, int] = (32, 32)) -> dict:
    head = np.ones(shape, dtype=bool)
    hair = np.ones(shape, dtype=bool)
    hair[10:20, 10:20] = False
    return {
        "image_size": (shape[1], shape[0]),
        "hair_mask": hair,
        "skin_mask": ~hair & head,
        "head_mask": head,
        "segmentation_mask": np.zeros(shape, dtype=np.int32),
        "hair_probability": np.ones(shape, dtype=np.float32),
    }


class HairAnalysisPipelineTests(unittest.TestCase):
    def _build_pipeline(self, **overrides) -> HairAnalysisPipeline:
        quality = MagicMock()
        quality.validate.return_value = {
            "valid": True,
            "image_type": "front",
            "errors": [],
            "warnings": [],
            "prepared_image": _blank(),
            "upscaled": False,
            "metrics": {},
        }
        quality.detect_landmarks.return_value = np.zeros((478, 3), dtype=np.float64)

        segmenter = MagicMock()
        segmenter.segment.return_value = _seg()

        validator = MagicMock()
        validator.validate.side_effect = lambda image, expected_view, **kw: _view_ok(
            expected_view
        )

        front_est = MagicMock()
        front_regions = _regions(front=True)
        front_est.estimate.return_value = _normative("front", front_regions)

        crown_est = MagicMock()
        crown_regions = _regions(crown=True)
        crown_est.estimate.return_value = _normative("crown", crown_regions)

        deficit = MagicMock()
        deficit.analyze.side_effect = lambda normative, hair, head, regions: _deficit(
            np.asarray(normative).astype(bool) & ~np.asarray(hair).astype(bool)
        )

        metrics_ext = MagicMock()

        def _extract(deficit_result, region_masks):
            has_front = int(np.count_nonzero(region_masks.frontal)) > 0
            has_crown = int(np.count_nonzero(region_masks.crown)) > 0
            if has_front and not has_crown:
                return _metrics(AnalysisMode.FRONT_ONLY, crown=None)
            if has_crown and not has_front:
                return _metrics(
                    AnalysisMode.CROWN_ONLY,
                    front=None,
                    left=None,
                    right=None,
                    crown=20.0,
                    overall=20.0,
                )
            return _metrics(AnalysisMode.COMBINED, crown=20.0)

        metrics_ext.extract.side_effect = _extract

        norwood = MagicMock()
        norwood.classify.side_effect = lambda metrics: _norwood(metrics.analysis_mode)

        defaults = dict(
            quality_checker=quality,
            segmenter=segmenter,
            view_validator=validator,
            front_estimator=front_est,
            crown_estimator=crown_est,
            deficit_analyzer=deficit,
            metrics_extractor=metrics_ext,
            norwood_classifier=norwood,
            stop_on_quality_failure=True,
        )
        defaults.update(overrides)
        return HairAnalysisPipeline(**defaults)

    def test_no_input_fails(self) -> None:
        pipeline = self._build_pipeline()
        report = pipeline.analyze()
        self.assertFalse(report.success)
        self.assertTrue(any("no_input_image" in e for e in report.errors))

    def test_front_only_orchestration(self) -> None:
        pipeline = self._build_pipeline()
        report = pipeline.analyze(front_image=_blank())
        self.assertTrue(report.success)
        self.assertEqual(report.analysis_mode, AnalysisMode.FRONT_ONLY)
        self.assertIsNotNone(report.front_estimation_result)
        self.assertIsNone(report.crown_estimation_result)
        self.assertIsNotNone(report.hair_loss_metrics)
        self.assertIsNotNone(report.norwood_result)
        self.assertEqual(report.norwood_result.stage, NorwoodStage.I)
        self.assertEqual(report.errors, [])

    def test_crown_only_orchestration(self) -> None:
        pipeline = self._build_pipeline()
        report = pipeline.analyze(crown_image=_blank())
        self.assertTrue(report.success)
        self.assertEqual(report.analysis_mode, AnalysisMode.CROWN_ONLY)
        self.assertIsNone(report.front_estimation_result)
        self.assertIsNotNone(report.crown_estimation_result)
        self.assertEqual(report.hair_loss_metrics.analysis_mode, AnalysisMode.CROWN_ONLY)

    def test_combined_orchestration(self) -> None:
        pipeline = self._build_pipeline()
        report = pipeline.analyze(front_image=_blank(), crown_image=_blank())
        self.assertTrue(report.success)
        self.assertEqual(report.analysis_mode, AnalysisMode.COMBINED)
        self.assertIsNotNone(report.front_estimation_result)
        self.assertIsNotNone(report.crown_estimation_result)
        self.assertIsNotNone(report.front_deficit_result)
        self.assertIsNotNone(report.crown_deficit_result)
        self.assertEqual(report.hair_loss_metrics.analysis_mode, AnalysisMode.COMBINED)
        self.assertTrue(report.hair_loss_metrics.front_available)
        self.assertTrue(report.hair_loss_metrics.crown_available)

    def test_view_validation_failure(self) -> None:
        validator = MagicMock()
        validator.validate.return_value = _view_fail("front")
        pipeline = self._build_pipeline(view_validator=validator)
        report = pipeline.analyze(front_image=_blank())
        self.assertFalse(report.success)
        self.assertTrue(any("view_invalid" in e for e in report.errors))
        self.assertIsNone(report.norwood_result)

    def test_quality_failure_stops_view(self) -> None:
        quality = MagicMock()
        quality.validate.return_value = {
            "valid": False,
            "image_type": "front",
            "errors": ["no_face_detected (required for front images)"],
            "warnings": [],
            "prepared_image": _blank(),
            "upscaled": False,
            "metrics": {},
        }
        pipeline = self._build_pipeline(quality_checker=quality)
        report = pipeline.analyze(front_image=_blank())
        self.assertFalse(report.success)
        self.assertTrue(any("quality_invalid" in e for e in report.errors))

    def test_quality_warning_continues_when_valid(self) -> None:
        quality = MagicMock()
        quality.validate.return_value = {
            "valid": True,
            "image_type": "front",
            "errors": [],
            "warnings": ["image_may_be_blurry"],
            "prepared_image": _blank(),
            "upscaled": False,
            "metrics": {},
        }
        quality.detect_landmarks.return_value = np.zeros((478, 3))
        pipeline = self._build_pipeline(quality_checker=quality)
        report = pipeline.analyze(front_image=_blank())
        self.assertTrue(report.success)
        self.assertTrue(any("blurry" in w for w in report.warnings))

    def test_low_resolution_warning_continues_with_upscale(self) -> None:
        """Sub-recommended resolution is soft: warn, upscale, continue."""
        quality = MagicMock()
        upscaled = Image.new("RGB", (512, 550), color=(120, 120, 120))
        quality.validate.return_value = {
            "valid": True,
            "image_type": "front",
            "errors": [],
            "warnings": [
                "Image resolution is lower than recommended. Results may be less accurate."
            ],
            "prepared_image": upscaled,
            "upscaled": True,
            "metrics": {
                "resolution": [401, 431],
                "prepared_resolution": [512, 550],
            },
        }
        quality.detect_landmarks.return_value = np.zeros((478, 3))
        pipeline = self._build_pipeline(quality_checker=quality)
        small = Image.new("RGB", (401, 431), color=(120, 120, 120))
        report = pipeline.analyze(front_image=small)
        self.assertTrue(report.success)
        self.assertTrue(any("resolution" in w.lower() for w in report.warnings))
        self.assertFalse(any("quality_invalid" in e for e in report.errors))

    def test_partial_combined_falls_back_to_single_view(self) -> None:
        """If crown fails view validation, front-only result still succeeds."""
        validator = MagicMock()

        def _validate(image, expected_view, **kw):
            if expected_view == "crown":
                return _view_fail("crown")
            return _view_ok("front")

        validator.validate.side_effect = _validate
        pipeline = self._build_pipeline(view_validator=validator)
        report = pipeline.analyze(front_image=_blank(), crown_image=_blank())
        self.assertTrue(report.success)
        self.assertEqual(report.analysis_mode, AnalysisMode.FRONT_ONLY)
        self.assertTrue(any("crown_view_invalid" in e for e in report.errors))

    def test_report_serialization(self) -> None:
        pipeline = self._build_pipeline()
        report = pipeline.analyze(front_image=_blank())
        payload = report.to_dict()
        for key in (
            "success",
            "analysis_mode",
            "quality_result",
            "view_validation",
            "segmentation_result",
            "front_estimation_result",
            "crown_estimation_result",
            "hair_deficit_result",
            "hair_loss_metrics",
            "norwood_result",
            "warnings",
            "errors",
            "processing_time",
            "metadata",
        ):
            self.assertIn(key, payload)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["analysis_mode"], "front_only")
        self.assertIsInstance(payload["processing_time"], float)

    def test_merge_combined_metrics_helper(self) -> None:
        front_m = _metrics(AnalysisMode.FRONT_ONLY, front=10.0, left=12.0, right=8.0)
        crown_m = _metrics(
            AnalysisMode.CROWN_ONLY,
            front=None,
            left=None,
            right=None,
            crown=40.0,
            overall=40.0,
        )
        front_d = _deficit(np.zeros((32, 32), dtype=bool))
        crown_d = _deficit(np.ones((32, 32), dtype=bool))
        merged = _merge_combined_metrics(front_m, crown_m, front_d, crown_d)
        self.assertEqual(merged.analysis_mode, AnalysisMode.COMBINED)
        self.assertEqual(merged.front_loss_percentage, 10.0)
        self.assertEqual(merged.crown_loss_percentage, 40.0)
        self.assertTrue(merged.front_available)
        self.assertTrue(merged.crown_available)

    def test_unexpected_exception_is_captured(self) -> None:
        segmenter = MagicMock()
        segmenter.segment.side_effect = RuntimeError("boom")
        pipeline = self._build_pipeline(segmenter=segmenter)
        report = pipeline.analyze(front_image=_blank())
        self.assertFalse(report.success)
        self.assertTrue(any("segmentation_error" in e or "boom" in e for e in report.errors))


class HairAnalysisReportTests(unittest.TestCase):
    def test_failed_report_defaults(self) -> None:
        report = HairAnalysisReport(success=False, analysis_mode=None, errors=["x"])
        payload = report.to_dict()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["errors"], ["x"])
        self.assertIsNone(payload["norwood_result"])


if __name__ == "__main__":
    unittest.main()
