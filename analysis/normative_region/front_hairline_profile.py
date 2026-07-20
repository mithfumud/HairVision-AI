"""Anatomical frontal hairline profile for visualization.

The mature male normative hairline is modelled as a cranial contour, not as a
curve fitted through three anchor points. Landmarks define forehead width,
temple insertion, and brow position; a lateral depth prior places the hairline
higher on the mid-frontal scalp and lower at the temporal insertions with a
broad, shallow central section — matching the reference profile:

        _/────────────────\\_

Visualization only — the estimator is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.front_features import (
    _FOREHEAD,
    _LEFT_TEMPLE_IDS,
    _RIGHT_TEMPLE_IDS,
    _landmarks_to_pixels,
)
from analysis.normative_region.front_hairline import NormativeHairline

# Forehead rim arc (subject-relative left → right) for contour following.
_FOREHEAD_RIM_IDS = (103, 67, 109, 10, 338, 297, 332)

# Brow reference for forehead-zone depth (same as front_hairline).
_LEFT_BROW_IDS = (70, 63, 105, 66, 107)
_RIGHT_BROW_IDS = (300, 293, 334, 296, 336)


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _forehead_rim_y_at_x(pixels: np.ndarray, x: float) -> float:
    """Interpolate the forehead surface height at column ``x`` from landmarks."""
    rim = pixels[list(_FOREHEAD_RIM_IDS), :2]
    order = np.argsort(rim[:, 0])
    rim = rim[order]
    xs = rim[:, 0]
    ys = rim[:, 1]
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    idx = int(np.searchsorted(xs, x))
    x0, y0 = xs[idx - 1], ys[idx - 1]
    x1, y1 = xs[idx], ys[idx]
    if abs(x1 - x0) < 1e-6:
        return float(y0)
    t = (x - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


@dataclass(frozen=True)
class AnatomicalFrontalHairlineProfile:
    """
    Continuous normative hairline from anatomical landmarks and cranial depth.

    Stations are derived from forehead width — not chosen to satisfy a
    polynomial. The lateral depth prior encodes where hair sits on a healthy
    male scalp: higher across the broad mid-frontal section, descending toward
    the temporal insertions through smooth rounded transitions.
    """

    center_x: float
    left_x: float
    right_x: float
    half_width: float
    forehead_y: float
    brow_y: float
    center_inset: float
    temple_inset: float
    plateau_lateral: float
    vertical_lift: float
    stations: tuple[tuple[float, float], ...]

    @classmethod
    def from_landmarks(
        cls,
        landmarks: np.ndarray,
        image_width: int,
        image_height: int,
        hairline: NormativeHairline,
        face_height: float,
        face_width: float,
        *,
        center_inset_fraction: float = 0.24,
        temple_inset_fraction: float = 0.36,
        plateau_lateral: float = 0.68,
        hairline_lift_fraction: float = 0.09,
    ) -> AnatomicalFrontalHairlineProfile:
        """
        Build the profile from MediaPipe landmarks and estimator span metadata.

        Parameters
        ----------
        center_inset_fraction
            Lateral shape only: relative inset at mid-frontal vs temporal stations,
            expressed as a fraction of the forehead zone (brow − forehead).
        temple_inset_fraction
            Same, for temporal insertions (creates the reference ``_/──\\_`` shape).
        plateau_lateral
            Fraction of half-width over which the broad central section stays
            at centre inset before the temporal transition (~0.68 ≈ 70%).
        hairline_lift_fraction
            Uniform upward shift (fraction of face height) applied to the whole
            profile. The forehead landmark (index 10) is a geometric anchor —
            the normative hairline on a healthy male sits above it in the upper
            scalp, not on the brow-side forehead surface.
        """
        if not 0.0 < center_inset_fraction < temple_inset_fraction < 0.55:
            raise ValueError(
                "Require 0 < center_inset_fraction < temple_inset_fraction < 0.55"
            )
        if not 0.5 < plateau_lateral < 0.9:
            raise ValueError("plateau_lateral must be in (0.5, 0.9)")
        if hairline_lift_fraction <= 0:
            raise ValueError("hairline_lift_fraction must be positive")

        pixels = _landmarks_to_pixels(np.asarray(landmarks), image_width, image_height)
        brow_y = float(
            np.mean(
                np.concatenate(
                    [pixels[_LEFT_BROW_IDS, 1], pixels[_RIGHT_BROW_IDS, 1]],
                )
            )
        )
        forehead_y = float(pixels[_FOREHEAD, 1])
        forehead_zone = max(brow_y - forehead_y, 0.05 * face_height)

        center_x = float(pixels[_FOREHEAD, 0])
        left_x = float(hairline.left_x)
        right_x = float(hairline.right_x)
        half_width = max(0.5 * (right_x - left_x), 0.08 * face_width)

        center_inset = center_inset_fraction * forehead_zone
        temple_inset = temple_inset_fraction * forehead_zone
        vertical_lift = hairline_lift_fraction * face_height

        # Anatomical stations at fixed cranial fractions (not spline knots).
        lateral_fractions = (1.0, 0.86, 0.72, 0.50, 0.28, 0.14, 0.0)
        station_xs: list[float] = []
        for lat in lateral_fractions:
            x_left = center_x - lat * half_width
            x_right = center_x + lat * half_width
            if x_left >= left_x:
                station_xs.append(x_left)
            if lat == 0.0:
                station_xs.append(center_x)
            if x_right <= right_x:
                station_xs.append(x_right)

        dedup_x = sorted({round(x, 2) for x in station_xs})
        ordered = tuple((x, 0.0) for x in dedup_x)

        profile = cls(
            center_x=center_x,
            left_x=left_x,
            right_x=right_x,
            half_width=half_width,
            forehead_y=forehead_y,
            brow_y=brow_y,
            center_inset=center_inset,
            temple_inset=temple_inset,
            plateau_lateral=plateau_lateral,
            vertical_lift=vertical_lift,
            stations=ordered,
        )
        filled = tuple(
            (x, profile.y_at_x(x, pixels)) for x, _ in profile.stations
        )
        return cls(
            center_x=center_x,
            left_x=left_x,
            right_x=right_x,
            half_width=half_width,
            forehead_y=forehead_y,
            brow_y=brow_y,
            center_inset=center_inset,
            temple_inset=temple_inset,
            plateau_lateral=plateau_lateral,
            vertical_lift=vertical_lift,
            stations=filled,
        )

    def _lateral_normalized(self, x: float) -> float:
        """0 at midline, 1 at temporal half-width."""
        return float(np.clip(abs(float(x) - self.center_x) / self.half_width, 0.0, 1.0))

    def _inset_at_lateral(self, lateral: float) -> float:
        """
        Cranial depth prior: broad shallow centre, smooth temporal descent.

        Inside ``plateau_lateral`` the inset is constant (broad frontal section).
        Beyond that, inset increases smoothly toward the temple insertion.
        """
        if lateral <= self.plateau_lateral:
            return self.center_inset
        tail = 1.0 - self.plateau_lateral
        if tail <= 1e-6:
            return self.temple_inset
        t = (lateral - self.plateau_lateral) / tail
        return self.center_inset + (self.temple_inset - self.center_inset) * _smoothstep(t)

    def y_at_x(self, x: float, pixels: np.ndarray | None = None) -> float:
        """
        Lower boundary at column ``x`` from forehead contour + cranial inset.

        The forehead rim (including landmark 10) is only a geometric reference.
        A uniform ``vertical_lift`` shifts the entire profile upward so the
        normative hairline sits in the upper scalp; lateral inset values are
        unchanged and therefore preserve the profile shape.
        """
        if x <= self.left_x:
            x_eval = self.left_x
        elif x >= self.right_x:
            x_eval = self.right_x
        else:
            x_eval = float(x)

        lateral = self._lateral_normalized(x_eval)
        inset = self._inset_at_lateral(lateral)

        if pixels is not None:
            rim_y = _forehead_rim_y_at_x(pixels, x_eval)
        else:
            forehead_zone = max(self.brow_y - self.forehead_y, 1.0)
            rim_y = self.forehead_y + 0.08 * forehead_zone * (lateral**2)

        return rim_y + inset - self.vertical_lift

    def sample_curve(
        self,
        pixels: np.ndarray,
        samples: int | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> list[tuple[float, float]]:
        x_start = 0.0 if x_min is None else float(x_min)
        x_end = self.right_x if x_max is None else float(x_max)
        span = max(1, int(np.ceil(x_end - x_start)))
        count = samples if samples is not None else max(3, span * 4)
        xs = np.linspace(x_start, x_end, count)
        return [(float(v), self.y_at_x(float(v), pixels)) for v in xs]

    def _last_row_above(self, x: float, pixels: np.ndarray | None = None) -> int:
        return int(np.floor(self.y_at_x(x, pixels) - 1e-9))

    def above_mask(
        self,
        height: int,
        width: int,
        pixels: np.ndarray | None = None,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=bool)
        for x in range(width):
            y_last = self._last_row_above(float(x), pixels)
            if y_last >= 0:
                mask[: y_last + 1, x] = True
        return mask


def build_anatomical_scalp_mask(
    head_mask: np.ndarray,
    profile: AnatomicalFrontalHairlineProfile,
    pixels: np.ndarray,
) -> np.ndarray:
    """Scalp envelope for visualization using the anatomical hairline."""
    height, width = head_mask.shape
    head = np.asarray(head_mask).astype(bool)
    scalp = np.zeros((height, width), dtype=bool)

    for x in range(width):
        column = head[:, x]
        if not np.any(column):
            continue
        y_top = int(np.where(column)[0].min())
        y_last = min(height - 1, profile._last_row_above(float(x), pixels))
        if y_top <= y_last:
            scalp[y_top : y_last + 1, x] = True

    return scalp & head


# Backward-compatible aliases for the test harness.
PiecewiseFrontalHairlineBoundary = AnatomicalFrontalHairlineProfile
build_piecewise_scalp_mask = build_anatomical_scalp_mask
