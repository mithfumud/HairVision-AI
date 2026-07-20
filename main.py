"""HairVision-AI pipeline orchestration (no CV algorithms here)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from analysis.crown_features import CrownFeatureExtractor
from analysis.features import FeatureExtractor
from analysis.front_features import FrontFeatureExtractor
from analysis.quality import ImageQualityChecker
from analysis.validation.view_validator import ViewValidator
from models.classifier import NorwoodClassifier
from models.segmentation import HairSegmenter
from utils.visualization import Visualizer

PIPELINE_VERSION = "1.0"


class HairVisionPipeline:
    """
    Coordinate quality → segmentation → features → Norwood → visualization.

    Heavy models are constructed once in ``__init__`` and reused.
    """

    def __init__(self) -> None:
        self.quality_checker = ImageQualityChecker()
        self.segmenter = HairSegmenter()
        self.common_feature_extractor = FeatureExtractor()
        self.front_feature_extractor = FrontFeatureExtractor()
        self.crown_feature_extractor = CrownFeatureExtractor()
        self.classifier = NorwoodClassifier()
        self.visualizer = Visualizer()

    def analyze(self, image: Image.Image, expected_image_type: str) -> dict[str, Any]:
        """
        Run the full HairVision pipeline on one image.

        ``expected_image_type`` is the caller's declared view (``\"front\"`` or
        ``\"crown\"``). After common feature extraction, the resolved type is
        read from those measurements so downstream steps stay aligned with the
        extractor output.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if expected_image_type not in ("front", "crown"):
            raise ValueError("expected_image_type must be 'front' or 'crown'")

        view_result = ViewValidator(quality_checker=self.quality_checker).validate(
            image,
            expected_image_type,
        )
        if not view_result.is_valid:
            raise ValueError(view_result.message)

        quality = self.quality_checker.validate(image, expected_image_type)
        if not quality["valid"]:
            detail = "; ".join(
                quality.get("errors")
                or quality.get("warnings", [])
                or ["quality check failed"]
            )
            raise ValueError(f"image failed quality checks: {detail}")

        analysis_image = quality.get("prepared_image") or image
        quality_payload = {
            key: value for key, value in quality.items() if key != "prepared_image"
        }

        segmentation = self.segmenter.segment(analysis_image)
        common = self.common_feature_extractor.extract(
            segmentation, expected_image_type
        )
        resolved_image_type = common.get("image_type", expected_image_type)
        if resolved_image_type not in ("front", "crown"):
            raise ValueError(
                f"unsupported image_type from features: {resolved_image_type}"
            )

        if resolved_image_type == "front":
            landmarks = self.quality_checker.detect_landmarks(analysis_image)
            specific = self.front_feature_extractor.extract(
                analysis_image, segmentation, landmarks
            )
        else:
            specific = self.crown_feature_extractor.extract(
                analysis_image, segmentation
            )

        measurements = self._merge_features(common, specific)
        classification = self.classifier.classify(measurements)
        visualization = self.visualizer.visualize(
            analysis_image, segmentation, measurements
        )

        return {
            "quality": quality_payload,
            "view_validation": view_result.to_dict(),
            "segmentation": self._segmentation_summary(segmentation),
            "measurements": measurements,
            "classification": classification,
            "visualization": visualization,
            "metadata": {
                "image_type": resolved_image_type,
                "pipeline_version": PIPELINE_VERSION,
                "upscaled": bool(quality.get("upscaled", False)),
            },
        }

    def _merge_features(
        self,
        common: dict[str, Any],
        specific: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge common and view-specific feature dicts without key collisions."""
        if not isinstance(common, dict) or not isinstance(specific, dict):
            raise TypeError("common and specific features must be dicts")

        overlap = set(common) & set(specific)
        if overlap:
            raise ValueError(
                "cannot merge features; colliding keys: "
                + ", ".join(sorted(overlap))
            )

        merged = dict(common)
        merged.update(specific)
        return merged

    def _segmentation_summary(self, segmentation: dict[str, Any]) -> dict[str, Any]:
        """Return JSON-friendly segmentation metadata (no full mask arrays)."""
        hair = segmentation.get("hair_mask")
        skin = segmentation.get("skin_mask")
        head = segmentation.get("head_mask")
        return {
            "image_size": segmentation.get("image_size"),
            "hair_pixels": int(np.count_nonzero(hair)) if hair is not None else 0,
            "skin_pixels": int(np.count_nonzero(skin)) if skin is not None else 0,
            "head_pixels": int(np.count_nonzero(head)) if head is not None else 0,
        }
