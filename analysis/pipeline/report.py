"""Structured end-to-end hair analysis report for UI and API consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analysis.deficit.types import HairDeficitResult
from analysis.metrics.types import AnalysisMode, HairLossMetrics
from analysis.normative_region.types import NormativeRegionResult, RegionMasks
from analysis.norwood.types import NorwoodClassificationResult
from analysis.validation.view_validator import ViewValidationResult


def _mask_summary(mask: np.ndarray | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    return {
        "shape": list(arr.shape),
        "true_pixels": int(np.count_nonzero(arr)),
    }


def _region_masks_summary(region_masks: RegionMasks | None) -> dict[str, Any] | None:
    if region_masks is None:
        return None
    return {
        "frontal": _mask_summary(region_masks.frontal),
        "left_temple": _mask_summary(region_masks.left_temple),
        "right_temple": _mask_summary(region_masks.right_temple),
        "crown": _mask_summary(region_masks.crown),
    }


def _normative_summary(result: NormativeRegionResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "view": result.view,
        "normative_region_mask": _mask_summary(result.normative_region_mask),
        "region_masks": _region_masks_summary(result.region_masks),
        "metadata": dict(result.metadata),
    }


def _deficit_summary(result: HairDeficitResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    stats = result.statistics
    largest = result.largest_component
    return {
        "deficit_mask": _mask_summary(result.deficit_mask),
        "component_count": stats.component_count,
        "total_deficit_area_px": stats.total_deficit_area_px,
        "deficit_percentage": stats.deficit_percentage,
        "normative_area_px": stats.normative_area_px,
        "largest_component": (
            None
            if largest is None
            else {
                "label_id": largest.label_id,
                "zone": largest.zone,
                "area_px": largest.area_px,
                "zone_area_ratio": largest.zone_area_ratio,
                "zone_overlap_ratio": largest.zone_overlap_ratio,
            }
        ),
        "statistics": {
            "total_deficit_area_px": stats.total_deficit_area_px,
            "deficit_percentage": stats.deficit_percentage,
            "normative_area_px": stats.normative_area_px,
            "component_count": stats.component_count,
            "largest_component_area_px": stats.largest_component_area_px,
            "raw_deficit_area_px": stats.raw_deficit_area_px,
        },
    }


def _segmentation_summary(segmentation: dict[str, Any] | None) -> dict[str, Any] | None:
    if segmentation is None:
        return None
    return {
        "image_size": segmentation.get("image_size"),
        "hair_mask": _mask_summary(segmentation.get("hair_mask")),
        "skin_mask": _mask_summary(segmentation.get("skin_mask")),
        "head_mask": _mask_summary(segmentation.get("head_mask")),
    }


def _view_validation_summary(
    result: ViewValidationResult | None,
) -> dict[str, Any] | None:
    if result is None:
        return None
    return result.to_dict()


@dataclass
class HairAnalysisReport:
    """
    Aggregate output of ``HairAnalysisPipeline``.

    Holds references to existing analytical results. Does not recompute them.
    """

    success: bool
    analysis_mode: AnalysisMode | None
    quality_result: dict[str, Any] | None = None
    view_validation: dict[str, Any] | None = None
    segmentation_result: dict[str, Any] | None = None
    front_estimation_result: NormativeRegionResult | None = None
    crown_estimation_result: NormativeRegionResult | None = None
    hair_deficit_result: HairDeficitResult | None = None
    # When both views are analyzed, optional per-view deficit results.
    front_deficit_result: HairDeficitResult | None = None
    crown_deficit_result: HairDeficitResult | None = None
    hair_loss_metrics: HairLossMetrics | None = None
    norwood_result: NorwoodClassificationResult | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary (no full mask arrays)."""
        segmentation_payload: dict[str, Any] | None = None
        if isinstance(self.segmentation_result, dict):
            segmentation_payload = {
                key: _segmentation_summary(value)
                for key, value in self.segmentation_result.items()
            }

        return {
            "success": self.success,
            "analysis_mode": (
                None if self.analysis_mode is None else self.analysis_mode.value
            ),
            "quality_result": self.quality_result,
            "view_validation": self.view_validation,
            "segmentation_result": segmentation_payload,
            "front_estimation_result": _normative_summary(self.front_estimation_result),
            "crown_estimation_result": _normative_summary(self.crown_estimation_result),
            "hair_deficit_result": _deficit_summary(self.hair_deficit_result),
            "front_deficit_result": _deficit_summary(self.front_deficit_result),
            "crown_deficit_result": _deficit_summary(self.crown_deficit_result),
            "hair_loss_metrics": (
                None if self.hair_loss_metrics is None else self.hair_loss_metrics.to_dict()
            ),
            "norwood_result": (
                None if self.norwood_result is None else self.norwood_result.to_dict()
            ),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "processing_time": round(self.processing_time, 4),
            "metadata": dict(self.metadata),
        }
