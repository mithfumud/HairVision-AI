"""Front-view normative hair-bearing region from landmarks and head geometry."""

from __future__ import annotations

import numpy as np
from PIL import Image

from analysis.front_features import (
    _FOREHEAD,
    _LEFT_TEMPLE_IDS,
    _RIGHT_TEMPLE_IDS,
    _as_bool_mask,
    _face_scale,
    _landmarks_to_pixels,
)
from analysis.normative_region.front_hairline import (
    _FOREHEAD_RIM_IDS,
    _LEFT_BROW_IDS,
    _RIGHT_BROW_IDS,
    build_scalp_column_mask,
    derive_normative_hairline,
    split_frontal_and_temples,
)
from analysis.normative_region.types import NormativeRegionResult, RegionMasks

# CelebAMask-HQ classes that are not hair-bearing (jonathandinu/face-parsing).
_DEFAULT_FACIAL_FEATURE_IDS = (2, 3, 4, 5, 6, 7, 10, 11, 12)


def _facial_feature_mask(segmentation_mask: np.ndarray) -> np.ndarray:
    """True where the segmentation label is a non-hair-bearing facial feature."""
    labels = np.asarray(segmentation_mask)
    out = np.zeros(labels.shape, dtype=bool)
    for class_id in _DEFAULT_FACIAL_FEATURE_IDS:
        out |= labels == class_id
    return out


def _head_contour_points(head_mask: np.ndarray, step: int = 4) -> list[list[float]]:
    """Sample outer head-contour points for debug visualization."""
    head = np.asarray(head_mask).astype(bool)
    h, w = head.shape
    pts: list[list[float]] = []
    for x in range(0, w, step):
        col = head[:, x]
        if not np.any(col):
            continue
        ys = np.where(col)[0]
        pts.append([float(x), float(ys.min())])
    for x in range(w - 1, -1, -step):
        col = head[:, x]
        if not np.any(col):
            continue
        ys = np.where(col)[0]
        pts.append([float(x), float(ys.max())])
    return pts


class FrontNormativeRegionEstimator:
    """
    Build a normative hair-bearing region for front-view images.

    The envelope follows the head contour superiorly, mirrored temporal
    boundaries laterally, and a multi-station cubic-spline frontal hairline
    whose control points are proportions of facial geometry (broad mid-frontal
    contour, gradual temporal descent).
    """

    def __init__(self) -> None:
        """Geometry is derived entirely from landmarks and head silhouette."""

    def estimate(
        self,
        image: Image.Image,
        landmarks: np.ndarray,
        head_mask: np.ndarray,
        segmentation_mask: np.ndarray | None = None,
    ) -> NormativeRegionResult:
        """
        Estimate normative hair-bearing subregions for a front-view image.

        Parameters
        ----------
        image
            Source image (used for width and height).
        landmarks
            MediaPipe face landmarks, shape (N, 2) or (N, 3).
        head_mask
            Boolean head region from segmentation.
        segmentation_mask
            Optional per-pixel class ids for facial-feature exclusion.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        width, height = image.size
        head = _as_bool_mask(head_mask, "head_mask")
        if head.shape != (height, width):
            raise ValueError(
                f"head_mask shape {head.shape} does not match image {(height, width)}"
            )
        if not np.any(head):
            raise ValueError("head_mask is empty")

        pixels = _landmarks_to_pixels(np.asarray(landmarks), width, height)
        face_height, face_width = _face_scale(pixels)

        hairline = derive_normative_hairline(
            pixels,
            face_height,
            face_width,
            head_mask=head,
        )

        scalp = build_scalp_column_mask(head, hairline)
        frontal, left_temple, right_temple = split_frontal_and_temples(scalp, hairline)
        crown = np.zeros((height, width), dtype=bool)

        if segmentation_mask is not None:
            seg = np.asarray(segmentation_mask)
            if seg.shape != head.shape:
                raise ValueError("segmentation_mask must match head_mask shape")
            features = _facial_feature_mask(seg)
            frontal &= ~features
            left_temple &= ~features
            right_temple &= ~features
            scalp &= ~features

        frontal &= head
        left_temple &= head
        right_temple &= head
        normative = (frontal | left_temple | right_temple) & head

        brow_ids = list(_LEFT_BROW_IDS) + list(_RIGHT_BROW_IDS)
        debug_landmarks = {
            "forehead": [float(pixels[_FOREHEAD, 0]), float(pixels[_FOREHEAD, 1])],
            "brow_points": pixels[brow_ids, :2].tolist(),
            "forehead_rim": pixels[list(_FOREHEAD_RIM_IDS), :2].tolist(),
            "left_temple_cluster": pixels[_LEFT_TEMPLE_IDS, :2].tolist(),
            "right_temple_cluster": pixels[_RIGHT_TEMPLE_IDS, :2].tolist(),
            "facial_centerline": [
                [hairline.center_x, float(np.min(pixels[list(_FOREHEAD_RIM_IDS), 1]))],
                [hairline.center_x, float(pixels[_FOREHEAD, 1] + face_height)],
            ],
            "head_contour": _head_contour_points(head),
        }

        return NormativeRegionResult(
            normative_region_mask=normative,
            region_masks=RegionMasks(
                frontal=frontal,
                left_temple=left_temple,
                right_temple=right_temple,
                crown=crown,
            ),
            view="front",
            metadata={
                "face_height": face_height,
                "face_width": face_width,
                "hairline_anchors": hairline.to_metadata(),
                "hairline_curve": hairline.sample_curve(),
                "hairline_control_points": list(hairline.control_points),
                "debug_landmarks": debug_landmarks,
                "parameters": {
                    "model": "mirrored_anatomical_spline",
                    "forehead_corrected": hairline.forehead_corrected,
                    "ideal_trichion_y": hairline.ideal_trichion_y,
                    "applied_offset": hairline.applied_offset,
                    "normalized_forehead_height": hairline.normalized_forehead_height,
                    "hairline_height_ratio": hairline.hairline_height_ratio,
                    "trichion_rim_blend": hairline.trichion_rim_blend,
                    "temple_depth_fraction": hairline.temple_depth_fraction,
                },
                "facial_feature_exclusion_applied": segmentation_mask is not None,
            },
        )
