"""Validate that an uploaded image matches the requested analysis view."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

from analysis.front_features import _FOREHEAD, _CHIN, _LEFT_CHEEK, _RIGHT_CHEEK
from analysis.quality import ImageQualityChecker
from models.segmentation import HairSegmenter

ViewType = Literal["front", "crown"]
DetectedView = Literal["front", "crown", "unknown"]

# User-facing validation messages.
MSG_NOT_FRONT = (
    "This image does not appear to be a frontal scalp image. "
    "Please upload a clear front-facing image."
)
MSG_NOT_CROWN = (
    "This appears to be a frontal image. Please upload a top/back view of "
    "the head for crown analysis."
)

# Landmark scoring.
_MIN_LANDMARK_COUNT = 468
_KEY_LANDMARK_IDS = (
    _FOREHEAD,
    _CHIN,
    _LEFT_CHEEK,
    _RIGHT_CHEEK,
    33,   # left eye outer
    263,  # right eye outer
    1,    # nose tip
    13,   # upper lip
)

# Segmentation labels treated as visible facial features (CelebAMask-HQ).
_FACIAL_FEATURE_LABELS = (
    "l_eye",
    "r_eye",
    "eye_g",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "l_brow",
    "r_brow",
)

# Combined score weights.
_FRONT_WEIGHTS = {
    "face_landmark": 0.45,
    "facial_feature": 0.30,
    "face_geometry": 0.15,
    "feature_visibility": 0.10,
}
_CROWN_WEIGHTS = {
    "no_face": 0.35,
    "head_dominance": 0.30,
    "low_facial_features": 0.20,
    "crown_geometry": 0.15,
}

# Decision thresholds (see design doc).
_HARD_CROWN_FACE_BLOCK = 0.70
_SOFT_CROWN_FACE_BLOCK = 0.50
_HARD_CROWN_FACIAL_BLOCK = 0.03
_HARD_FRONT_FACE_BLOCK = 0.25
_MIN_FRONT_LANDMARK = 0.40
_MAX_CROWN_LANDMARK = 0.50
_PASS_SCORE = 0.55
_SCORE_MARGIN = 0.10

# Normalization targets for segmentation / geometry cues.
_FACIAL_FEATURE_TARGET_FRAC = 0.03
_HEAD_FILL_TARGET_FRAC = 0.35
_MIN_FACE_HEIGHT_FRAC = 0.25


@dataclass(frozen=True)
class ViewValidationResult:
    """Outcome of validating an image against an expected analysis view."""

    is_valid: bool
    expected_view: ViewType
    detected_view: DetectedView
    confidence: float
    message: str
    reason: str
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""
        return {
            "is_valid": self.is_valid,
            "expected_view": self.expected_view,
            "detected_view": self.detected_view,
            "confidence": round(self.confidence, 4),
            "message": self.message,
            "reason": self.reason,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _landmarks_to_pixels(
    landmarks: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    pts = np.asarray(landmarks, dtype=np.float64)
    xy = pts[:, :2].copy()
    if np.nanmax(xy) <= 1.5:
        xy[:, 0] *= width
        xy[:, 1] *= height
    return xy


def _compute_face_landmark_score(
    landmarks: np.ndarray | None,
    width: int,
    height: int,
) -> tuple[float, float]:
    """
    Return (face_landmark_score, feature_visibility_score).

    Both values are in [0, 1].
    """
    if landmarks is None:
        return 0.0, 0.0

    count = int(landmarks.shape[0])
    if count == 0:
        return 0.0, 0.0

    pixels = _landmarks_to_pixels(landmarks, width, height)
    visible = 0
    for idx in _KEY_LANDMARK_IDS:
        if idx >= count:
            continue
        x, y = pixels[idx]
        if 0.0 <= x <= width and 0.0 <= y <= height:
            visible += 1

    visibility_ratio = visible / len(_KEY_LANDMARK_IDS)
    feature_visibility = _clamp(visibility_ratio)

    if count < _MIN_LANDMARK_COUNT:
        return _clamp(0.20 + 0.50 * visibility_ratio), feature_visibility

    score = 0.45 + 0.40 * visibility_ratio
    if visible >= 6:
        score += 0.10
    return _clamp(score), feature_visibility


def _compute_face_geometry_score(
    landmarks: np.ndarray | None,
    width: int,
    height: int,
) -> float:
    """Front portraits usually show a tall, centered face box."""
    if landmarks is None or landmarks.shape[0] <= _RIGHT_CHEEK:
        return 0.0

    pixels = _landmarks_to_pixels(landmarks, width, height)
    face_height = float(np.linalg.norm(pixels[_CHIN] - pixels[_FOREHEAD]))
    face_width = float(np.linalg.norm(pixels[_RIGHT_CHEEK] - pixels[_LEFT_CHEEK]))
    if face_height <= 1.0 or face_width <= 1.0:
        return 0.0

    height_ratio = face_height / float(height)
    aspect = face_height / face_width
    height_score = _clamp(height_ratio / _MIN_FACE_HEIGHT_FRAC)
    aspect_score = _clamp((aspect - 0.75) / 0.75)
    return _clamp(0.55 * height_score + 0.45 * aspect_score)


def _facial_feature_class_ids(id2label: dict[int, str]) -> list[int]:
    wanted = set(_FACIAL_FEATURE_LABELS)
    return [int(class_id) for class_id, name in id2label.items() if name in wanted]


def _compute_facial_feature_score(
    segmentation_mask: np.ndarray | None,
    id2label: dict[int, str] | None,
    image_area: int,
) -> float:
    if segmentation_mask is None or id2label is None or image_area <= 0:
        return 0.0

    class_ids = _facial_feature_class_ids(id2label)
    if not class_ids:
        return 0.0

    feature_pixels = int(np.isin(segmentation_mask, class_ids).sum())
    fraction = feature_pixels / float(image_area)
    return _clamp(fraction / _FACIAL_FEATURE_TARGET_FRAC)


def _compute_head_dominance_score(head_mask: np.ndarray | None, image_area: int) -> float:
    if head_mask is None or image_area <= 0:
        return 0.0
    head_pixels = int(np.count_nonzero(head_mask))
    return _clamp((head_pixels / float(image_area)) / _HEAD_FILL_TARGET_FRAC)


def _compute_crown_geometry_score(
    head_mask: np.ndarray | None,
    hair_mask: np.ndarray | None,
    skin_mask: np.ndarray | None,
) -> float:
    """Top-down crown shots often show a compact head with central scalp/hair."""
    if head_mask is None:
        return 0.0

    head = np.asarray(head_mask).astype(bool)
    if not np.any(head):
        return 0.0

    ys, xs = np.where(head)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    box_h = float(y1 - y0 + 1)
    box_w = float(x1 - x0 + 1)
    if box_h <= 0.0 or box_w <= 0.0:
        return 0.0

    compactness = min(box_w, box_h) / max(box_w, box_h)
    compactness_score = _clamp((compactness - 0.55) / 0.35)

    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    band_y = 0.20 * box_h
    band_x = 0.25 * box_w
    central = np.zeros_like(head, dtype=bool)
    central[
        int(round(cy - band_y)) : int(round(cy + band_y)) + 1,
        int(round(cx - band_x)) : int(round(cx + band_x)) + 1,
    ] = True
    central &= head

    central_area = int(np.count_nonzero(central))
    if central_area == 0:
        return 0.55 * compactness_score

    hair = np.asarray(hair_mask).astype(bool) if hair_mask is not None else np.zeros_like(head)
    skin = np.asarray(skin_mask).astype(bool) if skin_mask is not None else np.zeros_like(head)
    central_coverage = int(np.count_nonzero(central & (hair | skin))) / float(central_area)
    central_score = _clamp((central_coverage - 0.35) / 0.45)

    return _clamp(0.45 * compactness_score + 0.55 * central_score)


def _combine_front_score(
    face_landmark_score: float,
    facial_feature_score: float,
    face_geometry_score: float,
    feature_visibility: float,
) -> float:
    parts = {
        "face_landmark": face_landmark_score,
        "facial_feature": facial_feature_score,
        "face_geometry": face_geometry_score,
        "feature_visibility": feature_visibility,
    }
    return sum(_FRONT_WEIGHTS[key] * parts[key] for key in _FRONT_WEIGHTS)


def _combine_crown_score(
    face_landmark_score: float,
    facial_feature_score: float,
    head_dominance_score: float,
    crown_geometry_score: float,
) -> float:
    parts = {
        "no_face": 1.0 - face_landmark_score,
        "head_dominance": head_dominance_score,
        "low_facial_features": 1.0 - facial_feature_score,
        "crown_geometry": crown_geometry_score,
    }
    return sum(_CROWN_WEIGHTS[key] * parts[key] for key in _CROWN_WEIGHTS)


def _detected_view_from_scores(front_score: float, crown_score: float) -> DetectedView:
    if abs(front_score - crown_score) < 0.05:
        return "unknown"
    return "front" if front_score > crown_score else "crown"


class ViewValidator:
    """
    Determine whether an image is compatible with front or crown analysis.

    Uses MediaPipe landmarks and SegFormer face-parsing cues. This is an input
    validation gate — not a fallback mechanism for downstream estimators.
    """

    def __init__(
        self,
        *,
        quality_checker: ImageQualityChecker | None = None,
        segmenter: HairSegmenter | None = None,
    ) -> None:
        self._quality_checker = quality_checker or ImageQualityChecker()
        self._segmenter = segmenter
        self._owns_quality_checker = quality_checker is None

    def close(self) -> None:
        """Release the internally created Face Landmarker, if any."""
        if self._owns_quality_checker:
            self._quality_checker.close()

    def __enter__(self) -> ViewValidator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _try_detect_landmarks(self, image: Image.Image) -> np.ndarray | None:
        rgb = np.asarray(image.convert("RGB"))
        if not self._quality_checker._face_detected(rgb):
            return None
        try:
            return self._quality_checker.detect_landmarks(image)
        except ValueError:
            return None

    def _get_segmentation(self, image: Image.Image) -> dict[str, Any]:
        if self._segmenter is None:
            self._segmenter = HairSegmenter()
        return self._segmenter.segment(image)

    def validate(
        self,
        image: Image.Image,
        expected_view: ViewType,
        *,
        segmentation: dict[str, Any] | None = None,
        landmarks: np.ndarray | None = None,
    ) -> ViewValidationResult:
        """
        Validate ``expected_view`` against image content.

        Pass precomputed ``segmentation`` / ``landmarks`` to avoid duplicate
        model runs when the caller already computed them.
        """
        if expected_view not in ("front", "crown"):
            raise ValueError("expected_view must be 'front' or 'crown'")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        width, height = image.size
        image_area = width * height

        if landmarks is None:
            landmarks = self._try_detect_landmarks(image)

        face_landmark_score, feature_visibility = _compute_face_landmark_score(
            landmarks,
            width,
            height,
        )
        face_geometry_score = _compute_face_geometry_score(landmarks, width, height)

        need_segmentation = (
            segmentation is None
            and (
                expected_view == "front"
                or face_landmark_score >= _SOFT_CROWN_FACE_BLOCK
                or face_landmark_score < _MIN_FRONT_LANDMARK
            )
        )

        facial_feature_score = 0.0
        head_dominance_score = 0.0
        crown_geometry_score = 0.0

        if segmentation is not None or need_segmentation:
            seg = segmentation if segmentation is not None else self._get_segmentation(image)
            facial_feature_score = _compute_facial_feature_score(
                seg.get("segmentation_mask"),
                seg.get("id2label"),
                image_area,
            )
            head_dominance_score = _compute_head_dominance_score(
                seg.get("head_mask"),
                image_area,
            )
            crown_geometry_score = _compute_crown_geometry_score(
                seg.get("head_mask"),
                seg.get("hair_mask"),
                seg.get("skin_mask"),
            )

        front_score = _combine_front_score(
            face_landmark_score,
            facial_feature_score,
            face_geometry_score,
            feature_visibility,
        )
        crown_score = _combine_crown_score(
            face_landmark_score,
            facial_feature_score,
            head_dominance_score,
            crown_geometry_score,
        )
        detected_view = _detected_view_from_scores(front_score, crown_score)
        confidence = max(front_score, crown_score)

        scores = {
            "front_score": front_score,
            "crown_score": crown_score,
            "face_landmark_score": face_landmark_score,
            "facial_feature_score": facial_feature_score,
            "face_geometry_score": face_geometry_score,
            "feature_visibility": feature_visibility,
            "head_dominance_score": head_dominance_score,
            "crown_geometry_score": crown_geometry_score,
        }

        if expected_view == "crown":
            if face_landmark_score >= _HARD_CROWN_FACE_BLOCK:
                return ViewValidationResult(
                    is_valid=False,
                    expected_view=expected_view,
                    detected_view="front",
                    confidence=face_landmark_score,
                    message=MSG_NOT_CROWN,
                    reason="high_confidence_face_detected",
                    scores=scores,
                )
            if (
                face_landmark_score >= _SOFT_CROWN_FACE_BLOCK
                and facial_feature_score >= _HARD_CROWN_FACIAL_BLOCK
            ):
                return ViewValidationResult(
                    is_valid=False,
                    expected_view=expected_view,
                    detected_view="front",
                    confidence=front_score,
                    message=MSG_NOT_CROWN,
                    reason="facial_features_detected",
                    scores=scores,
                )

        if expected_view == "front" and face_landmark_score < _HARD_FRONT_FACE_BLOCK:
            return ViewValidationResult(
                is_valid=False,
                expected_view=expected_view,
                detected_view="crown" if crown_score >= front_score else "unknown",
                confidence=crown_score,
                message=MSG_NOT_FRONT,
                reason="no_face_detected",
                scores=scores,
            )

        is_valid = False
        reason = "view_mismatch"

        if expected_view == "front":
            is_valid = (
                face_landmark_score >= _MIN_FRONT_LANDMARK
                and front_score >= _PASS_SCORE
                and front_score >= crown_score + _SCORE_MARGIN
            )
            if is_valid:
                reason = "front_view_confirmed"
            elif detected_view == "crown":
                reason = "detected_crown_view"
        else:
            is_valid = (
                face_landmark_score < _MAX_CROWN_LANDMARK
                and crown_score >= _PASS_SCORE
                and crown_score >= front_score + _SCORE_MARGIN
            )
            if is_valid:
                reason = "crown_view_confirmed"
            elif detected_view == "front":
                reason = "detected_front_view"

        message = ""
        if not is_valid:
            message = MSG_NOT_FRONT if expected_view == "front" else MSG_NOT_CROWN

        return ViewValidationResult(
            is_valid=is_valid,
            expected_view=expected_view,
            detected_view=detected_view,
            confidence=confidence,
            message=message,
            reason=reason,
            scores=scores,
        )


def print_view_validation_result(result: ViewValidationResult) -> None:
    """Print a human-readable summary for manual QA scripts."""
    status = "PASS" if result.is_valid else "FAIL"
    print(f"View validation: {status}")
    print(f"  Expected view:   {result.expected_view}")
    print(f"  Detected view:   {result.detected_view}")
    print(f"  Confidence:      {result.confidence:.2f}")
    print(f"  Reason:          {result.reason}")
    if result.scores:
        print(
            "  Scores:          "
            f"front={result.scores.get('front_score', 0.0):.2f}, "
            f"crown={result.scores.get('crown_score', 0.0):.2f}, "
            f"face={result.scores.get('face_landmark_score', 0.0):.2f}"
        )
    if not result.is_valid:
        print(f"  Message:         {result.message}")
