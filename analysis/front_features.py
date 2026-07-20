"""Front-view measurable features from segmentation and MediaPipe landmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

# MediaPipe Face Mesh indices (subject-relative left/right).
_FOREHEAD = 10
_CHIN = 152
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454

# Temple cluster landmarks used to build local ROIs.
_LEFT_TEMPLE_IDS = [21, 54, 103, 67, 109, 127, 162, 234]
_RIGHT_TEMPLE_IDS = [251, 284, 332, 297, 338, 356, 389, 454]

_MIN_LANDMARKS = 455  # need index 454

_DEFAULT_HAIRLINE_COVERAGE_THRESHOLD = 0.30


def _as_bool_mask(mask: Any, name: str) -> np.ndarray:
    """Coerce a segmentation mask to a boolean ndarray."""
    if mask is None:
        raise KeyError(f"segmentation is missing required key: {name}")
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return arr.astype(bool)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when the denominator is zero."""
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _landmarks_to_pixels(
    landmarks: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert landmarks to pixel coordinates of shape (N, 2).

    Accepts (N, 2) or (N, 3). Values in [0, 1] are treated as normalized;
    larger values are treated as already-pixel coordinates.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("landmarks must have shape (N, 2) or (N, 3)")
    if pts.shape[0] < _MIN_LANDMARKS:
        raise ValueError(
            f"landmarks must contain at least {_MIN_LANDMARKS} points "
            f"(got {pts.shape[0]})"
        )

    xy = pts[:, :2].copy()
    if np.nanmax(xy) <= 1.5:
        xy[:, 0] *= width
        xy[:, 1] *= height
    return xy


def _face_scale(pixels: np.ndarray) -> tuple[float, float]:
    """Return (face_height, face_width) in pixels from key landmarks."""
    face_height = float(np.linalg.norm(pixels[_CHIN] - pixels[_FOREHEAD]))
    face_width = float(np.linalg.norm(pixels[_RIGHT_CHEEK] - pixels[_LEFT_CHEEK]))
    if face_height <= 1.0 or face_width <= 1.0:
        raise ValueError("face height/width from landmarks is degenerate")
    return face_height, face_width


def _clip_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Clip a float box to integer image bounds (x0, y0, x1, y1 exclusive)."""
    xa = int(np.clip(np.floor(min(x0, x1)), 0, width))
    xb = int(np.clip(np.ceil(max(x0, x1)), 0, width))
    ya = int(np.clip(np.floor(min(y0, y1)), 0, height))
    yb = int(np.clip(np.ceil(max(y0, y1)), 0, height))
    return xa, ya, xb, yb


def _rect_mask(
    bounds: tuple[int, int, int, int],
    height: int,
    width: int,
) -> np.ndarray:
    """Build a boolean mask that is True inside clipped bounds."""
    x0, y0, x1, y1 = bounds
    mask = np.zeros((height, width), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return mask
    mask[y0:y1, x0:x1] = True
    return mask


def _temple_roi_bounds(
    temple_points: np.ndarray,
    width: int,
    height: int,
    pad_x_fraction: float,
    pad_y_fraction: float,
) -> tuple[int, int, int, int]:
    """
    Clip bounds for a temple ROI from a landmark cluster and relative padding.

    Padding scales with the cluster span so the box adapts to face size.
    """
    xs = temple_points[:, 0]
    ys = temple_points[:, 1]
    temple_width = float(xs.max() - xs.min())
    temple_height = float(ys.max() - ys.min())
    pad_x = pad_x_fraction * max(temple_width, 1.0)
    pad_y = pad_y_fraction * max(temple_height, 1.0)
    return _clip_box(
        xs.min() - pad_x,
        ys.min() - pad_y,
        xs.max() + pad_x,
        ys.max() + pad_y,
        width,
        height,
    )


def _temple_roi_bounds_centered(
    temple_points: np.ndarray,
    face_width: float,
    face_height: float,
    width: int,
    height: int,
    width_fraction: float,
    height_fraction: float,
) -> tuple[int, int, int, int]:
    """Clip bounds for a temple ROI centered on the cluster with face-scaled size."""
    cx = float(temple_points[:, 0].mean())
    cy = float(temple_points[:, 1].mean())
    box_w = width_fraction * face_width
    box_h = height_fraction * face_height
    return _clip_box(
        cx - 0.5 * box_w,
        cy - 0.5 * box_h,
        cx + 0.5 * box_w,
        cy + 0.5 * box_h,
        width,
        height,
    )


def _normative_hairline_anchors(
    pixels: np.ndarray,
    face_height: float,
    hairline_alpha: float,
    hairline_beta: float,
) -> dict[str, float]:
    """
    Normative hairline anchor points in pixel coordinates.

    Anchors sit above the forehead and temple landmarks by fractions of
    face height (smaller y is higher on the image). The lower boundary of
    the frontal normative band is a parabola through these three points.
    """
    forehead = pixels[_FOREHEAD]
    left = pixels[_LEFT_TEMPLE_IDS].mean(axis=0)
    right = pixels[_RIGHT_TEMPLE_IDS].mean(axis=0)

    return {
        "center_x": float(forehead[0]),
        "center_y": float(forehead[1] - hairline_alpha * face_height),
        "left_x": float(left[0]),
        "left_y": float(left[1] - hairline_beta * face_height),
        "right_x": float(right[0]),
        "right_y": float(right[1] - hairline_beta * face_height),
        "forehead_y": float(forehead[1]),
    }


def _normative_hairline_y_at_x(
    x: float,
    left_x: float,
    left_y: float,
    center_x: float,
    center_y: float,
    right_x: float,
    right_y: float,
) -> float:
    """
    Normative hairline y at column x via quadratic Lagrange interpolation.

    The curve is a parabola passing through the left temple, forehead center,
    and right temple anchors. This is smoother than piecewise-linear segments
    (no derivative break at the forehead) while remaining fully determined by
    the same three geometry-derived points.

    Falls back to piecewise-linear interpolation when anchors are degenerate.
    """
    if left_x >= right_x:
        return center_y

    x_eval = float(np.clip(x, left_x, right_x))
    x0, y0 = left_x, left_y
    x1, y1 = center_x, center_y
    x2, y2 = right_x, right_y

    d01 = (x0 - x1) * (x0 - x2)
    d12 = (x1 - x0) * (x1 - x2)
    d20 = (x2 - x0) * (x2 - x1)
    if min(abs(d01), abs(d12), abs(d20)) < 1e-6:
        return _normative_hairline_y_at_x_linear(
            x_eval, left_x, left_y, center_x, center_y, right_x, right_y
        )

    l0 = ((x_eval - x1) * (x_eval - x2)) / d01
    l1 = ((x_eval - x0) * (x_eval - x2)) / d12
    l2 = ((x_eval - x0) * (x_eval - x1)) / d20
    return y0 * l0 + y1 * l1 + y2 * l2


def _normative_hairline_y_at_x_linear(
    x: float,
    left_x: float,
    left_y: float,
    center_x: float,
    center_y: float,
    right_x: float,
    right_y: float,
) -> float:
    """Piecewise-linear fallback through the three hairline anchors."""
    if x <= center_x:
        if center_x == left_x:
            return center_y
        t = (x - left_x) / (center_x - left_x)
        return left_y + t * (center_y - left_y)
    if center_x == right_x:
        return center_y
    t = (x - center_x) / (right_x - center_x)
    return center_y + t * (right_y - center_y)


class FrontFeatureExtractor:
    """Extract measurable front-view features (no classification)."""

    def __init__(
        self,
        hairline_coverage_threshold: float = _DEFAULT_HAIRLINE_COVERAGE_THRESHOLD,
    ) -> None:
        if not 0.0 < hairline_coverage_threshold <= 1.0:
            raise ValueError(
                "hairline_coverage_threshold must be in the interval (0, 1]"
            )
        self.hairline_coverage_threshold = hairline_coverage_threshold

    def extract(
        self,
        image: Image.Image,
        segmentation: dict,
        landmarks: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute front-view measurements from masks and facial landmarks.

        Returns
        -------
        hairline_height
            Vertical distance from forehead landmark to the frontal hairline,
            normalized by face height.
        left_temple_recession / right_temple_recession
            Missing-hair mass in each temple ROI, normalized by temple width
            and face height.
        temple_asymmetry
            Absolute difference between left and right temple recession.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if not isinstance(segmentation, dict):
            raise TypeError("segmentation must be a dict")

        width, height = image.size
        hair_mask = _as_bool_mask(segmentation.get("hair_mask"), "hair_mask")
        head_mask = _as_bool_mask(segmentation.get("head_mask"), "head_mask")

        if hair_mask.shape != (height, width) or head_mask.shape != (height, width):
            raise ValueError(
                "hair_mask and head_mask must match the image dimensions "
                f"(expected {(height, width)})"
            )
        if hair_mask.shape != head_mask.shape:
            raise ValueError("hair_mask and head_mask must share the same shape")

        pixels = _landmarks_to_pixels(np.asarray(landmarks), width, height)
        face_height, face_width = _face_scale(pixels)

        hairline_height = self._hairline_height(
            hair_mask, pixels, face_height, face_width, width, height
        )
        left_recession = self._temple_recession(
            hair_mask,
            head_mask,
            pixels[_LEFT_TEMPLE_IDS, :],
            face_height,
            width,
            height,
        )
        right_recession = self._temple_recession(
            hair_mask,
            head_mask,
            pixels[_RIGHT_TEMPLE_IDS, :],
            face_height,
            width,
            height,
        )

        return {
            "hairline_height": hairline_height,
            "left_temple_recession": left_recession,
            "right_temple_recession": right_recession,
            "temple_asymmetry": float(abs(left_recession - right_recession)),
        }

    def _hairline_height(
        self,
        hair_mask: np.ndarray,
        pixels: np.ndarray,
        face_height: float,
        face_width: float,
        width: int,
        height: int,
    ) -> float:
        """
        Distance from forehead landmark to the frontal hairline.

        In a central strip above the forehead, compute hair coverage per row.
        Scanning upward from the forehead, the first row whose coverage exceeds
        ``self.hairline_coverage_threshold`` is treated as the hairline. The vertical
        gap is then normalized by face height.
        """
        forehead = pixels[_FOREHEAD]
        fx, fy = float(forehead[0]), float(forehead[1])

        half_w = 0.12 * face_width
        x0, y0, x1, y1 = _clip_box(
            fx - half_w,
            0.0,
            fx + half_w,
            fy + 0.05 * face_height,
            width,
            height,
        )
        if x1 <= x0 or y1 <= y0:
            return 0.0

        strip = hair_mask[y0:y1, x0:x1]
        if strip.size == 0:
            return 0.0

        # Fraction of hair pixels in each row (vectorized).
        row_coverage = strip.mean(axis=1)

        # Scan from forehead (bottom of strip) toward the top of the image.
        for local_row in range(strip.shape[0] - 1, -1, -1):
            if row_coverage[local_row] >= self.hairline_coverage_threshold:
                hairline_y = float(y0 + local_row)
                distance = abs(fy - hairline_y)
                return _safe_ratio(distance, face_height)

        return 0.0

    def _temple_recession(
        self,
        hair_mask: np.ndarray,
        head_mask: np.ndarray,
        temple_points: np.ndarray,
        face_height: float,
        width: int,
        height: int,
    ) -> float:
        """
        Missing-hair mass in a temple ROI, normalized by temple width.

        recession = missing_pixels / (temple_width * face_height)
        """
        xs = temple_points[:, 0]
        temple_width = float(xs.max() - xs.min())

        x0, y0, x1, y1 = _temple_roi_bounds(
            temple_points, width, height, pad_x_fraction=0.15, pad_y_fraction=0.25
        )
        if x1 <= x0 or y1 <= y0:
            return 0.0

        region = head_mask[y0:y1, x0:x1]
        hair = hair_mask[y0:y1, x0:x1]
        if not np.any(region):
            region = np.ones_like(hair, dtype=bool)

        missing = int(np.count_nonzero(region & ~hair))
        # Comparable across resolutions: missing mass / (width * face_height).
        return _safe_ratio(float(missing), temple_width * face_height)
