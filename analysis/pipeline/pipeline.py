"""End-to-end orchestration of hair-analysis modules."""

from __future__ import annotations

import time
import traceback
from typing import Any, Literal

import numpy as np
from PIL import Image

from analysis.coverage.refiner import HairCoverageRefiner
from analysis.deficit.analyzer import HairDeficitAnalyzer
from analysis.deficit.types import HairDeficitResult
from analysis.metrics.extractor import HairLossMetricsExtractor
from analysis.metrics.types import AnalysisMode, HairLossMetrics
from analysis.normative_region.crown_estimator import CrownNormativeRegionEstimator
from analysis.normative_region.front_estimator import FrontNormativeRegionEstimator
from analysis.normative_region.types import NormativeRegionResult
from analysis.norwood.classifier import NorwoodClassifier
from analysis.pipeline.report import HairAnalysisReport
from analysis.quality import ImageQualityChecker
from analysis.validation.view_validator import ViewValidator
from models.segmentation import HairSegmenter

PIPELINE_VERSION = "1.1"


def _merge_combined_metrics(
    front_metrics: HairLossMetrics | None,
    crown_metrics: HairLossMetrics | None,
    front_deficit: HairDeficitResult | None,
    crown_deficit: HairDeficitResult | None,
) -> HairLossMetrics:
    """
    Aggregate front and crown extractor outputs into one COMBINED metrics object.

    Does not re-run deficit analysis — only combines existing percentages/stats.
    """
    if front_metrics is None or crown_metrics is None:
        raise ValueError("combined metrics require both front and crown metric results")
    if front_deficit is None or crown_deficit is None:
        raise ValueError("combined metrics require both front and crown deficit results")

    front_stats = front_deficit.statistics
    crown_stats = crown_deficit.statistics
    normative_total = front_stats.normative_area_px + crown_stats.normative_area_px
    deficit_total = (
        front_stats.total_deficit_area_px + crown_stats.total_deficit_area_px
    )
    if normative_total > 0:
        overall_loss = 100.0 * float(deficit_total) / float(normative_total)
        overall_coverage = 100.0 - overall_loss
    else:
        overall_loss = None
        overall_coverage = None

    candidates = [
        c
        for c in (front_deficit.largest_component, crown_deficit.largest_component)
        if c is not None
    ]
    if candidates:
        largest = max(candidates, key=lambda item: item.area_px)
        largest_zone = largest.zone
        largest_pct = float(largest.zone_area_ratio * 100.0)
    else:
        largest_zone = None
        largest_pct = None

    return HairLossMetrics(
        front_loss_percentage=front_metrics.front_loss_percentage,
        left_temple_loss_percentage=front_metrics.left_temple_loss_percentage,
        right_temple_loss_percentage=front_metrics.right_temple_loss_percentage,
        crown_loss_percentage=crown_metrics.crown_loss_percentage,
        overall_hair_loss_percentage=overall_loss,
        overall_hair_coverage_percentage=overall_coverage,
        component_count=(
            front_stats.component_count + crown_stats.component_count
        ),
        largest_deficit_zone=largest_zone,
        largest_deficit_percentage=largest_pct,
        analysis_mode=AnalysisMode.COMBINED,
        front_available=True,
        crown_available=True,
    )


class HairAnalysisPipeline:
    """
    Orchestrate quality → segmentation → estimation → deficit → metrics → Norwood.

    Business logic lives in the injected analytical modules. This class only
    coordinates calls and packages a ``HairAnalysisReport``.
    """

    def __init__(
        self,
        *,
        quality_checker: ImageQualityChecker | None = None,
        segmenter: HairSegmenter | None = None,
        view_validator: ViewValidator | None = None,
        front_estimator: FrontNormativeRegionEstimator | None = None,
        crown_estimator: CrownNormativeRegionEstimator | None = None,
        deficit_analyzer: HairDeficitAnalyzer | None = None,
        metrics_extractor: HairLossMetricsExtractor | None = None,
        norwood_classifier: NorwoodClassifier | None = None,
        coverage_refiner: HairCoverageRefiner | None = None,
        stop_on_quality_failure: bool = True,
        refine_hair_coverage: bool = True,
    ) -> None:
        self._quality_checker = quality_checker
        self._segmenter = segmenter
        self._view_validator = view_validator
        self._front_estimator = front_estimator or FrontNormativeRegionEstimator()
        self._crown_estimator = crown_estimator or CrownNormativeRegionEstimator()
        self._deficit_analyzer = deficit_analyzer or HairDeficitAnalyzer()
        self._metrics_extractor = metrics_extractor or HairLossMetricsExtractor()
        self._norwood_classifier = norwood_classifier or NorwoodClassifier()
        self._coverage_refiner = coverage_refiner or HairCoverageRefiner()
        self.refine_hair_coverage = refine_hair_coverage
        self.stop_on_quality_failure = stop_on_quality_failure
        self._owns_quality = quality_checker is None
        self._owns_validator = view_validator is None

    def _get_quality_checker(self) -> ImageQualityChecker:
        if self._quality_checker is None:
            self._quality_checker = ImageQualityChecker()
        return self._quality_checker

    def _get_segmenter(self) -> HairSegmenter:
        if self._segmenter is None:
            self._segmenter = HairSegmenter()
        return self._segmenter

    def _get_view_validator(self) -> ViewValidator:
        if self._view_validator is None:
            self._view_validator = ViewValidator(
                quality_checker=self._get_quality_checker(),
                segmenter=self._get_segmenter(),
            )
            self._owns_validator = True
        return self._view_validator

    def close(self) -> None:
        """Release owned resources."""
        if self._owns_validator and self._view_validator is not None:
            self._view_validator.close()
        elif self._owns_quality and self._quality_checker is not None:
            self._quality_checker.close()

    def __enter__(self) -> HairAnalysisPipeline:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def analyze(
        self,
        *,
        front_image: Image.Image | None = None,
        crown_image: Image.Image | None = None,
    ) -> HairAnalysisReport:
        """
        Run the full analysis pipeline for the provided view image(s).

        Provide ``front_image``, ``crown_image``, or both (combined mode).
        """
        started = time.perf_counter()
        warnings: list[str] = []
        errors: list[str] = []

        if front_image is None and crown_image is None:
            return HairAnalysisReport(
                success=False,
                analysis_mode=None,
                errors=["no_input_image: provide front_image and/or crown_image"],
                processing_time=time.perf_counter() - started,
                metadata={"pipeline_version": PIPELINE_VERSION},
            )

        quality_result: dict[str, Any] = {}
        view_validation: dict[str, Any] = {}
        segmentation_result: dict[str, Any] = {}
        front_estimation: NormativeRegionResult | None = None
        crown_estimation: NormativeRegionResult | None = None
        front_deficit: HairDeficitResult | None = None
        crown_deficit: HairDeficitResult | None = None
        front_metrics: HairLossMetrics | None = None
        crown_metrics: HairLossMetrics | None = None

        try:
            if front_image is not None:
                front_out = self._analyze_front_view(front_image)
                quality_result["front"] = front_out["quality"]
                view_validation["front"] = front_out["view_validation"]
                warnings.extend(front_out["warnings"])
                errors.extend(front_out["errors"])
                if front_out["segmentation"] is not None:
                    segmentation_result["front"] = front_out["segmentation"]
                front_estimation = front_out["estimation"]
                front_deficit = front_out["deficit"]
                front_metrics = front_out["metrics"]

            if crown_image is not None:
                crown_out = self._analyze_crown_view(crown_image)
                quality_result["crown"] = crown_out["quality"]
                view_validation["crown"] = crown_out["view_validation"]
                warnings.extend(crown_out["warnings"])
                errors.extend(crown_out["errors"])
                if crown_out["segmentation"] is not None:
                    segmentation_result["crown"] = crown_out["segmentation"]
                crown_estimation = crown_out["estimation"]
                crown_deficit = crown_out["deficit"]
                crown_metrics = crown_out["metrics"]

            hair_loss_metrics: HairLossMetrics | None = None
            hair_deficit_result: HairDeficitResult | None = None
            analysis_mode: AnalysisMode | None = None

            if front_metrics is not None and crown_metrics is not None:
                analysis_mode = AnalysisMode.COMBINED
                hair_loss_metrics = _merge_combined_metrics(
                    front_metrics,
                    crown_metrics,
                    front_deficit,
                    crown_deficit,
                )
                hair_deficit_result = front_deficit
            elif front_metrics is not None:
                analysis_mode = AnalysisMode.FRONT_ONLY
                hair_loss_metrics = front_metrics
                hair_deficit_result = front_deficit
            elif crown_metrics is not None:
                analysis_mode = AnalysisMode.CROWN_ONLY
                hair_loss_metrics = crown_metrics
                hair_deficit_result = crown_deficit
            else:
                return HairAnalysisReport(
                    success=False,
                    analysis_mode=None,
                    quality_result=quality_result or None,
                    view_validation=view_validation or None,
                    segmentation_result=segmentation_result or None,
                    front_estimation_result=front_estimation,
                    crown_estimation_result=crown_estimation,
                    warnings=warnings,
                    errors=errors
                    or ["analysis_failed: no view completed successfully"],
                    processing_time=time.perf_counter() - started,
                    metadata={"pipeline_version": PIPELINE_VERSION},
                )

            norwood_result = self._norwood_classifier.classify(hair_loss_metrics)

            return HairAnalysisReport(
                success=True,
                analysis_mode=analysis_mode,
                quality_result=quality_result or None,
                view_validation=view_validation or None,
                segmentation_result=segmentation_result or None,
                front_estimation_result=front_estimation,
                crown_estimation_result=crown_estimation,
                hair_deficit_result=hair_deficit_result,
                front_deficit_result=front_deficit,
                crown_deficit_result=crown_deficit,
                hair_loss_metrics=hair_loss_metrics,
                norwood_result=norwood_result,
                warnings=warnings,
                errors=errors,
                processing_time=time.perf_counter() - started,
                metadata={
                    "pipeline_version": PIPELINE_VERSION,
                    "views_requested": [
                        name
                        for name, img in (
                            ("front", front_image),
                            ("crown", crown_image),
                        )
                        if img is not None
                    ],
                },
            )
        except Exception as exc:  # noqa: BLE001 — UI-facing orchestration boundary
            errors.append(f"pipeline_error: {exc}")
            return HairAnalysisReport(
                success=False,
                analysis_mode=None,
                quality_result=quality_result or None,
                view_validation=view_validation or None,
                segmentation_result=segmentation_result or None,
                front_estimation_result=front_estimation,
                crown_estimation_result=crown_estimation,
                hair_deficit_result=front_deficit or crown_deficit,
                front_deficit_result=front_deficit,
                crown_deficit_result=crown_deficit,
                hair_loss_metrics=front_metrics or crown_metrics,
                warnings=warnings,
                errors=errors,
                processing_time=time.perf_counter() - started,
                metadata={
                    "pipeline_version": PIPELINE_VERSION,
                    "traceback": traceback.format_exc(),
                },
            )

    def _analyze_front_view(self, image: Image.Image) -> dict[str, Any]:
        return self._analyze_single_view(image, expected_view="front")

    def _analyze_crown_view(self, image: Image.Image) -> dict[str, Any]:
        return self._analyze_single_view(image, expected_view="crown")

    def _analyze_single_view(
        self,
        image: Image.Image,
        *,
        expected_view: Literal["front", "crown"],
    ) -> dict[str, Any]:
        """
        Run view validation → quality → segmentation → estimation → deficit → metrics.

        Returns a dict with results and per-view warnings/errors. Does not raise
        for recoverable validation failures.
        """
        warnings: list[str] = []
        errors: list[str] = []
        out: dict[str, Any] = {
            "quality": None,
            "view_validation": None,
            "segmentation": None,
            "estimation": None,
            "deficit": None,
            "metrics": None,
            "coverage_refinement": None,
            "warnings": warnings,
            "errors": errors,
        }

        if not isinstance(image, Image.Image):
            errors.append(f"{expected_view}: image must be a PIL.Image.Image")
            return out

        image = image.convert("RGB")

        try:
            view_result = self._get_view_validator().validate(image, expected_view)
            out["view_validation"] = view_result.to_dict()
            if not view_result.is_valid:
                errors.append(f"{expected_view}_view_invalid: {view_result.message}")
                return out
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{expected_view}_view_validation_error: {exc}")
            return out

        try:
            quality = self._get_quality_checker().validate(image, expected_view)
            # Persist JSON-safe quality fields only (omit prepared PIL image).
            out["quality"] = {
                key: value
                for key, value in quality.items()
                if key != "prepared_image"
            }
            for warning in quality.get("warnings", []):
                warnings.append(f"{expected_view}: {warning}")
            hard_errors = list(quality.get("errors") or [])
            if not quality.get("valid", False) and self.stop_on_quality_failure:
                detail = "; ".join(
                    hard_errors
                    or quality.get("warnings", [])
                    or ["quality check failed"]
                )
                errors.append(f"{expected_view}_quality_invalid: {detail}")
                return out
            # Soft path: continue on warnings; use upscaled image when provided.
            prepared = quality.get("prepared_image")
            if isinstance(prepared, Image.Image):
                image = prepared
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{expected_view}_quality_error: {exc}")
            return out

        try:
            segmentation = self._get_segmenter().segment(image)
            out["segmentation"] = segmentation
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{expected_view}_segmentation_error: {exc}")
            return out

        head_mask = segmentation["head_mask"]
        hair_mask = segmentation["hair_mask"]

        try:
            if expected_view == "front":
                landmarks = self._get_quality_checker().detect_landmarks(image)
                estimation = self._front_estimator.estimate(
                    image,
                    landmarks,
                    head_mask,
                    segmentation_mask=segmentation.get("segmentation_mask"),
                )
            else:
                estimation = self._crown_estimator.estimate(
                    image,
                    head_mask,
                    hair_mask,
                )
            out["estimation"] = estimation
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{expected_view}_estimation_error: {exc}")
            return out

        try:
            hair_for_deficit = hair_mask
            # Coverage refinement targets sparse crown thinning where semantic
            # hair_mask over-labels visible scalp. Front keeps raw hair_mask.
            if self.refine_hair_coverage and expected_view == "crown":
                refine = self._coverage_refiner.refine(
                    image,
                    hair_mask,
                    head_mask,
                    hair_probability=segmentation.get("hair_probability"),
                    skin_mask=segmentation.get("skin_mask"),
                    gate_mask=estimation.normative_region_mask,
                )
                hair_for_deficit = refine.hair_coverage_mask
                out["coverage_refinement"] = {
                    "density_threshold": refine.density_threshold,
                    "window_radius_px": refine.window_radius_px,
                    "pixels_demoted": refine.pixels_demoted,
                    "sparse_exposed_pixels": int(
                        np.count_nonzero(refine.sparse_exposed_mask)
                    ),
                }

            deficit = self._deficit_analyzer.analyze(
                estimation.normative_region_mask,
                hair_for_deficit,
                head_mask,
                estimation.region_masks,
            )
            out["deficit"] = deficit
            metrics = self._metrics_extractor.extract(deficit, estimation.region_masks)
            out["metrics"] = metrics
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{expected_view}_deficit_metrics_error: {exc}")
            return out

        return out
