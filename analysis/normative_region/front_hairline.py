"""Anatomical normative frontal hairline from proportional control stations.

Geometric principles (from reference hairline style — not a pixel template)
--------------------------------------------------------------------------
1. Vertical placement sits in the upper forehead band near the cranial rim,
   lower than a full facial-thirds trichion, scaled by brow–rim depth.
2. Mid-frontal contour is *broad and gently rounded* (low curvature), not a
   shallow circular arc or a dead-flat plateau.
3. Curvature increases toward the temples: a gradual rounded descent into
   the temporal insertions (no sharp corners).
4. Left/right halves are mathematical mirrors about the facial midline.
5. Span is clipped to the head silhouette so the envelope stays on scalp.

Model
-----
Mirrored cubic spline through anatomically placed stations. Station lateral
fractions and relative depths are proportions of face/forehead geometry —
the same construction adapts to any face size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis.front_features import (
    _CHIN,
    _FOREHEAD,
    _LEFT_TEMPLE_IDS,
    _RIGHT_TEMPLE_IDS,
)

_LEFT_BROW_IDS = (70, 63, 105, 66, 107)
_RIGHT_BROW_IDS = (300, 293, 334, 296, 336)
_FOREHEAD_RIM_IDS = (103, 67, 109, 10, 338, 297, 332)
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263

# Blend between full-thirds trichion (0) and forehead-rim (1) → lower placement.
_TRICHION_RIM_BLEND = 0.58
# Total temple descent as a fraction of brow–rim forehead band.
_TEMPLE_DEPTH_FRACTION = 0.32
# Relative depth along the half-span (0 at midline → 1 at temple).
# Early values stay near 0 → broad rounded center; later values rise → temples.
_STATION_LATERAL = (0.00, 0.28, 0.52, 0.74, 1.00)
_STATION_DEPTH = (0.00, 0.06, 0.28, 0.68, 1.00)


def _brow_y(pixels: np.ndarray) -> float:
    return float(
        np.mean(
            np.concatenate(
                [pixels[list(_LEFT_BROW_IDS), 1], pixels[list(_RIGHT_BROW_IDS), 1]],
            )
        )
    )


def _eye_y(pixels: np.ndarray) -> float:
    return float(
        0.5 * (pixels[_LEFT_EYE_OUTER, 1] + pixels[_RIGHT_EYE_OUTER, 1])
    )


def _rim_y_at_x(pixels: np.ndarray, x: float) -> float:
    rim = pixels[list(_FOREHEAD_RIM_IDS), :2]
    order = np.argsort(rim[:, 0])
    xs = rim[order, 0]
    ys = rim[order, 1]
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    i = int(np.searchsorted(xs, x))
    x0, y0 = float(xs[i - 1]), float(ys[i - 1])
    x1, y1 = float(xs[i]), float(ys[i])
    if abs(x1 - x0) < 1e-6:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _natural_cubic_spline_coeffs(
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Natural cubic spline on strictly increasing knots."""
    n = len(xs) - 1
    if n < 1:
        raise ValueError("spline requires at least two knots")
    h = np.diff(xs)
    if np.any(h <= 0):
        raise ValueError("spline knot x must be strictly increasing")

    a = ys.astype(np.float64).copy()
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (
            a[i] - a[i - 1]
        )

    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    for i in range(1, n):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    return xs, a[:-1], b, c[:-1], d


def _eval_cubic_spline(
    x: float,
    xs: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    if x <= xs[0]:
        i = 0
        dx = x - xs[0]
    elif x >= xs[-1]:
        i = len(a) - 1
        dx = x - xs[i]
    else:
        i = int(np.searchsorted(xs, x) - 1)
        i = max(0, min(i, len(a) - 1))
        dx = x - xs[i]
    return float(a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx)


@dataclass(frozen=True)
class NormativeHairline:
    """Mirror-symmetric anatomical frontal hairline (multi-station cubic spline)."""

    left_x: float
    left_y: float
    center_x: float
    center_y: float
    right_x: float
    right_y: float
    forehead_y: float
    brow_y: float
    half_width: float
    knot_xs: tuple[float, ...]
    knot_ys: tuple[float, ...]
    control_points: tuple[tuple[float, float], ...]
    ideal_trichion_y: float
    applied_offset: float
    forehead_corrected: bool
    eye_y: float
    chin_y: float
    normalized_forehead_height: float
    hairline_height_ratio: float
    trichion_rim_blend: float
    temple_depth_fraction: float

    def _spline_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xs = np.asarray(self.knot_xs, dtype=np.float64)
        ys = np.asarray(self.knot_ys, dtype=np.float64)
        return _natural_cubic_spline_coeffs(xs, ys)

    def y_at_x(self, x: float) -> float:
        """Evaluate the mirrored anatomical spline at column ``x``."""
        x = float(x)
        if x <= self.left_x:
            return self.left_y
        if x >= self.right_x:
            return self.right_y
        xs, a, b, c, d = self._spline_state()
        return _eval_cubic_spline(x, xs, a, b, c, d)

    def sample_curve(self, samples: int | None = None) -> list[tuple[float, float]]:
        span = max(1, int(np.ceil(self.right_x - self.left_x)))
        count = samples if samples is not None else max(80, span * 4)
        xs = np.linspace(self.left_x, self.right_x, count)
        return [(float(v), self.y_at_x(float(v))) for v in xs]

    def _last_row_above(self, x: float) -> int:
        return int(np.floor(self.y_at_x(x) - 1e-9))

    def above_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=bool)
        for x in range(width):
            y_last = self._last_row_above(float(x))
            if y_last >= 0:
                mask[: y_last + 1, x] = True
        return mask

    def forehead_region_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=bool)
        y_brow = int(np.clip(np.floor(self.brow_y), 0, height - 1))
        x0 = int(np.clip(np.floor(self.left_x), 0, width - 1))
        x1 = int(np.clip(np.ceil(self.right_x), 0, width))
        for x in range(x0, x1):
            y0 = self._last_row_above(float(x)) + 1
            if y0 <= y_brow:
                mask[y0 : y_brow + 1, x] = True
        return mask

    def to_metadata(self) -> dict[str, Any]:
        return {
            "left_x": self.left_x,
            "left_y": self.left_y,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "right_x": self.right_x,
            "right_y": self.right_y,
            "forehead_y": self.forehead_y,
            "brow_y": self.brow_y,
            "half_width": self.half_width,
            "knot_xs": list(self.knot_xs),
            "knot_ys": list(self.knot_ys),
            "control_points": [list(p) for p in self.control_points],
            "ideal_trichion_y": self.ideal_trichion_y,
            "applied_offset": self.applied_offset,
            "forehead_corrected": self.forehead_corrected,
            "eye_y": self.eye_y,
            "chin_y": self.chin_y,
            "normalized_forehead_height": self.normalized_forehead_height,
            "hairline_height_ratio": self.hairline_height_ratio,
            "trichion_rim_blend": self.trichion_rim_blend,
            "temple_depth_fraction": self.temple_depth_fraction,
            "profile_model": "mirrored_anatomical_spline",
            "plateau_lateral": 0.0,
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> NormativeHairline:
        knot_xs = meta.get("knot_xs")
        knot_ys = meta.get("knot_ys")
        if not knot_xs or not knot_ys:
            knot_xs = [meta["left_x"], meta["center_x"], meta["right_x"]]
            knot_ys = [meta["left_y"], meta["center_y"], meta["right_y"]]
        control = meta.get("control_points") or list(zip(knot_xs, knot_ys))
        half = float(
            meta.get(
                "half_width",
                0.5 * (float(meta["right_x"]) - float(meta["left_x"])),
            )
        )
        return cls(
            left_x=float(meta["left_x"]),
            left_y=float(meta["left_y"]),
            center_x=float(meta["center_x"]),
            center_y=float(meta["center_y"]),
            right_x=float(meta["right_x"]),
            right_y=float(meta["right_y"]),
            forehead_y=float(meta["forehead_y"]),
            brow_y=float(meta["brow_y"]),
            half_width=half,
            knot_xs=tuple(float(v) for v in knot_xs),
            knot_ys=tuple(float(v) for v in knot_ys),
            control_points=tuple((float(p[0]), float(p[1])) for p in control),
            ideal_trichion_y=float(meta.get("ideal_trichion_y", meta["center_y"])),
            applied_offset=float(meta.get("applied_offset", 0.0)),
            forehead_corrected=bool(meta.get("forehead_corrected", False)),
            eye_y=float(meta.get("eye_y", meta["brow_y"])),
            chin_y=float(meta.get("chin_y", meta["brow_y"])),
            normalized_forehead_height=float(
                meta.get("normalized_forehead_height", 0.0)
            ),
            hairline_height_ratio=float(meta.get("hairline_height_ratio", 0.0)),
            trichion_rim_blend=float(
                meta.get("trichion_rim_blend", _TRICHION_RIM_BLEND)
            ),
            temple_depth_fraction=float(
                meta.get("temple_depth_fraction", _TEMPLE_DEPTH_FRACTION)
            ),
        )


def _symmetric_half_width(
    center_x: float,
    raw_left: float,
    raw_right: float,
    face_width: float,
    head_mask: np.ndarray | None,
    probe_y: float,
) -> float:
    left_span = max(center_x - raw_left, 0.05 * face_width)
    right_span = max(raw_right - center_x, 0.05 * face_width)
    half = min(0.5 * (left_span + right_span), left_span, right_span)
    half = min(half, 0.58 * face_width)

    if head_mask is not None:
        head = np.asarray(head_mask).astype(bool)
        y = int(np.clip(round(probe_y), 0, head.shape[0] - 1))
        row = head[y]
        if np.any(row):
            xs = np.where(row)[0]
            head_half = min(center_x - float(xs.min()), float(xs.max()) - center_x)
            if head_half > 0:
                half = min(half, 0.98 * head_half)

    return max(half, 0.15 * face_width)


def _build_mirrored_stations(
    center_x: float,
    half_width: float,
    center_y: float,
    temple_drop: float,
    *,
    lateral: tuple[float, ...] = _STATION_LATERAL,
    depth: tuple[float, ...] = _STATION_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build strictly increasing (x, y) knots, mirrored about the midline.

    Right-half stations are placed at ``lateral[i] * half_width`` with depth
    ``depth[i] * temple_drop``; the left half is the exact mirror.
    """
    if len(lateral) != len(depth):
        raise ValueError("lateral and depth station tables must match")

    right_xs = [center_x + lat * half_width for lat in lateral]
    right_ys = [center_y + dep * temple_drop for dep in depth]

    # Mirror interior right stations to the left (skip midline duplicate).
    left_xs = [2.0 * center_x - x for x in reversed(right_xs[1:])]
    left_ys = list(reversed(right_ys[1:]))

    xs = np.asarray(left_xs + right_xs, dtype=np.float64)
    ys = np.asarray(left_ys + right_ys, dtype=np.float64)
    return xs, ys


def derive_normative_hairline(
    pixels: np.ndarray,
    face_height: float,
    face_width: float,
    *,
    head_mask: np.ndarray | None = None,
    trichion_rim_blend: float = _TRICHION_RIM_BLEND,
    temple_depth_fraction: float = _TEMPLE_DEPTH_FRACTION,
) -> NormativeHairline:
    """
    Derive a mirror-symmetric anatomical frontal hairline.

    Control stations are fractions of half-width and forehead-band depth, so
    the same construction scales across face sizes without a pixel template.
    """
    if face_height <= 0 or face_width <= 0:
        raise ValueError("face_height and face_width must be positive")
    if not 0.0 <= trichion_rim_blend <= 1.0:
        raise ValueError("trichion_rim_blend must be in [0, 1]")
    if not 0.15 <= temple_depth_fraction <= 0.50:
        raise ValueError("temple_depth_fraction out of anatomical range")

    brow = _brow_y(pixels)
    eye = _eye_y(pixels)
    chin_y = float(pixels[_CHIN, 1])
    forehead = pixels[_FOREHEAD]
    forehead_y = float(forehead[1])
    center_x = float(forehead[0])
    rim_center_y = _rim_y_at_x(pixels, center_x)

    mid_lower = max(chin_y - brow, 0.25 * face_height)
    upper_third = 0.5 * mid_lower
    ideal_trichion_y = brow - upper_third
    forehead_band = max(brow - rim_center_y, 0.05 * face_height)

    # Lower than full-thirds trichion by blending toward the cranial rim.
    center_y = (
        (1.0 - trichion_rim_blend) * ideal_trichion_y
        + trichion_rim_blend * rim_center_y
    )

    rim_pts = pixels[list(_FOREHEAD_RIM_IDS), :2]
    head_apex_y = float(np.min(rim_pts[:, 1]))
    if head_mask is not None:
        head = np.asarray(head_mask).astype(bool)
        mid0 = int(np.clip(center_x - 0.05 * face_width, 0, head.shape[1] - 1))
        mid1 = int(np.clip(center_x + 0.05 * face_width, 0, head.shape[1] - 1))
        tops = [
            int(np.where(head[:, x])[0].min())
            for x in range(mid0, mid1 + 1)
            if np.any(head[:, x])
        ]
        if tops:
            head_apex_y = float(min(tops))

    corrected = False
    # Stay in the upper forehead / lower-scalp band (anatomical envelope).
    superior_floor = head_apex_y + 0.03 * face_height
    caudal_limit = rim_center_y + 0.22 * forehead_band
    if center_y < superior_floor:
        center_y = superior_floor
        corrected = True
    if center_y > caudal_limit:
        center_y = caudal_limit
        corrected = True

    temple_drop = temple_depth_fraction * forehead_band
    temple_y = center_y + temple_drop
    if temple_y > brow - 0.15 * forehead_band:
        temple_y = brow - 0.15 * forehead_band
        temple_drop = max(temple_y - center_y, 0.08 * forehead_band)
        temple_y = center_y + temple_drop
        corrected = True

    left_temple = pixels[_LEFT_TEMPLE_IDS].mean(axis=0)
    right_temple = pixels[_RIGHT_TEMPLE_IDS].mean(axis=0)
    raw_left = float(min(float(np.min(rim_pts[:, 0])), left_temple[0]))
    raw_right = float(max(float(np.max(rim_pts[:, 0])), right_temple[0]))

    half = _symmetric_half_width(
        center_x,
        raw_left,
        raw_right,
        face_width,
        head_mask,
        center_y,
    )
    left_x = center_x - half
    right_x = center_x + half

    knot_xs, knot_ys = _build_mirrored_stations(
        center_x, half, center_y, temple_drop
    )
    # Endpoints must match span exactly.
    knot_xs[0] = left_x
    knot_xs[-1] = right_x
    knot_ys[0] = temple_y
    knot_ys[-1] = temple_y

    control_points = tuple(
        (float(x), float(y)) for x, y in zip(knot_xs, knot_ys)
    )

    return NormativeHairline(
        left_x=left_x,
        left_y=temple_y,
        center_x=center_x,
        center_y=center_y,
        right_x=right_x,
        right_y=temple_y,
        forehead_y=forehead_y,
        brow_y=brow,
        half_width=half,
        knot_xs=tuple(float(v) for v in knot_xs),
        knot_ys=tuple(float(v) for v in knot_ys),
        control_points=control_points,
        ideal_trichion_y=float(ideal_trichion_y),
        applied_offset=float(center_y - rim_center_y),
        forehead_corrected=corrected,
        eye_y=eye,
        chin_y=chin_y,
        normalized_forehead_height=float(forehead_band / face_height),
        hairline_height_ratio=float((brow - center_y) / face_height),
        trichion_rim_blend=trichion_rim_blend,
        temple_depth_fraction=temple_depth_fraction,
    )


def build_scalp_column_mask(
    head_mask: np.ndarray,
    hairline: NormativeHairline,
) -> np.ndarray:
    height, width = head_mask.shape
    head = np.asarray(head_mask).astype(bool)
    scalp = np.zeros((height, width), dtype=bool)

    for x in range(width):
        column = head[:, x]
        if not np.any(column):
            continue
        y_top = int(np.where(column)[0].min())
        y_last = min(height - 1, hairline._last_row_above(float(x)))
        if y_top <= y_last:
            scalp[y_top : y_last + 1, x] = True

    return scalp & head


def split_frontal_and_temples(
    scalp_mask: np.ndarray,
    hairline: NormativeHairline,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scalp = np.asarray(scalp_mask).astype(bool)
    _height, width = scalp.shape
    xs = np.arange(width)

    left_cut = int(np.clip(np.floor(hairline.left_x), 0, width - 1))
    right_cut = int(np.clip(np.ceil(hairline.right_x), 0, width))

    frontal_cols = (xs >= left_cut) & (xs < right_cut)
    left_cols = xs < left_cut
    right_cols = xs >= right_cut

    frontal = scalp.copy()
    frontal[:, ~frontal_cols] = False

    left_temple = scalp.copy()
    left_temple[:, ~left_cols] = False

    right_temple = scalp.copy()
    right_temple[:, ~right_cols] = False

    return frontal, left_temple, right_temple
