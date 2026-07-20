"""Crown-view normative hair-bearing region from head geometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from analysis.front_features import _as_bool_mask
from analysis.normative_region.crown_vertex import estimate_vertex_from_image
from analysis.normative_region.types import NormativeRegionResult, RegionMasks

# Asymmetric crown envelope extents (fractions of head PCA spans from vertex).
_DEFAULT_POSTERIOR_EXTENT = 0.46
_DEFAULT_ANTERIOR_EXTENT = 0.34
# Wide enough for bilateral crown thinning (left + right of vertex).
_DEFAULT_LATERAL_EXTENT = 0.78
# Superellipse exponent for soft, non-circular boundary.
_DEFAULT_SHAPE_EXPONENT = 2.15
# Minimum inset from the head boundary (fraction of equivalent radius).
_DEFAULT_BOUNDARY_MARGIN = 0.030


@dataclass(frozen=True)
class HeadFrame:
    """Head-centric coordinate frame derived from the scalp contour."""

    centroid_x: float
    centroid_y: float
    major_axis_x: float
    major_axis_y: float
    minor_axis_x: float
    minor_axis_y: float
    major_span: float
    minor_span: float
    angle_rad: float
    posterior_axis_x: float
    posterior_axis_y: float


def _head_scale(head_mask: np.ndarray) -> tuple[float, float, float, float, int, int, int, int]:
    """Return head bbox, pixel count, equivalent radius, and y/x extrema."""
    ys, xs = np.where(head_mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    box_h = float(y_max - y_min + 1)
    box_w = float(x_max - x_min + 1)
    head_pixels = float(len(xs))
    radius = float(np.sqrt(head_pixels / np.pi))
    return box_w, box_h, head_pixels, radius, y_min, y_max, x_min, x_max


def estimate_head_frame(head_mask: np.ndarray) -> HeadFrame | None:
    """
    Derive a head-aligned frame from the scalp contour via PCA.

    The major axis follows the longest head dimension (anterior–posterior on
    crown photos); the minor axis is lateral. Used for ROI shape and scale.
    """
    head = np.asarray(head_mask).astype(bool)
    ys, xs = np.where(head)
    if len(xs) < 3:
        return None

    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return None

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    pc_major = evecs[:, order[0]]
    pc_minor = evecs[:, order[1]]

    proj_major = centered @ pc_major
    proj_minor = centered @ pc_minor
    major_span = float(proj_major.max() - proj_major.min())
    minor_span = float(proj_minor.max() - proj_minor.min())
    if major_span <= 1.0 or minor_span <= 1.0:
        return None

    end_a = centroid + pc_major * major_span * 0.5
    end_b = centroid - pc_major * major_span * 0.5
    if end_a[1] >= end_b[1]:
        posterior = pc_major
    else:
        posterior = -pc_major

    return HeadFrame(
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
        major_axis_x=float(pc_major[0]),
        major_axis_y=float(pc_major[1]),
        minor_axis_x=float(pc_minor[0]),
        minor_axis_y=float(pc_minor[1]),
        major_span=major_span,
        minor_span=minor_span,
        angle_rad=float(np.arctan2(pc_major[1], pc_major[0])),
        posterior_axis_x=float(posterior[0]),
        posterior_axis_y=float(posterior[1]),
    )


def build_crown_normative_mask(
    head_mask: np.ndarray,
    frame: HeadFrame,
    *,
    vertex_x: float,
    vertex_y: float,
    equivalent_radius: float,
    posterior_extent: float = _DEFAULT_POSTERIOR_EXTENT,
    anterior_extent: float = _DEFAULT_ANTERIOR_EXTENT,
    lateral_extent: float = _DEFAULT_LATERAL_EXTENT,
    shape_exponent: float = _DEFAULT_SHAPE_EXPONENT,
    boundary_margin: float = _DEFAULT_BOUNDARY_MARGIN,
) -> np.ndarray:
    """
    Build the expected crown hair-bearing envelope centered on the vertex.

    This is the anatomical region where crown hair is normally present — not
    a trace of existing baldness. The envelope is asymmetric (more posterior
    than anterior), scaled to the head PCA spans, softly bounded, and clipped
    by the head contour so the outline follows scalp anatomy.
    """
    head = np.asarray(head_mask).astype(bool)
    crown = np.zeros_like(head, dtype=bool)
    if not np.any(head):
        return crown

    semi_post = posterior_extent * frame.major_span
    semi_ant = anterior_extent * frame.major_span
    semi_lat = 0.5 * lateral_extent * frame.minor_span
    if min(semi_post, semi_ant, semi_lat) <= 1.0:
        return crown

    height, width = head.shape
    yy, xx = np.mgrid[:height, :width]
    dx = xx.astype(np.float64) - vertex_x
    dy = yy.astype(np.float64) - vertex_y

    # Signed coordinates: +major = posterior, +minor = lateral (right).
    local_major = dx * frame.posterior_axis_x + dy * frame.posterior_axis_y
    local_minor = dx * frame.minor_axis_x + dy * frame.minor_axis_y

    post_term = np.clip(local_major / semi_post, 0.0, None) ** shape_exponent
    ant_term = np.clip(-local_major / semi_ant, 0.0, None) ** shape_exponent
    lat_term = (np.abs(local_minor) / semi_lat) ** shape_exponent
    envelope = (post_term + ant_term + lat_term) <= 1.0

    margin_px = max(2.0, boundary_margin * equivalent_radius)
    dist = cv2.distanceTransform(head.astype(np.uint8), cv2.DIST_L2, 5)
    inset = dist >= margin_px

    crown = envelope & inset & head
    return crown


def _expand_crown_into_adjacent_visible_scalp(
    crown_mask: np.ndarray,
    head_mask: np.ndarray,
    gray: np.ndarray,
    *,
    max_distance_px: float,
    luminance_floor: float = 90.0,
    rim_margin_px: float = 4.0,
) -> np.ndarray:
    """
    Soft-extend the crown envelope into nearby bright exposed scalp.

    Grows geodesically through bright head pixels so a lit bald lobe that is
    separated from the oval by a thin dark-hair strip can still enter the
    gate. Dark dense hair is never a growth medium.
    """
    crown = np.asarray(crown_mask).astype(bool)
    head = np.asarray(head_mask).astype(bool)
    if not np.any(crown) or max_distance_px <= 0.0:
        return crown

    g = np.asarray(gray, dtype=np.float32)
    dist_in = cv2.distanceTransform(head.astype(np.uint8), cv2.DIST_L2, 5)
    growable = (
        head
        & ((g >= float(luminance_floor)) | crown)
        & (dist_in >= float(rim_margin_px))
    )
    if not np.any(growable & ~crown):
        return crown

    kernel = np.ones((3, 3), np.uint8)
    current = (crown & growable).astype(np.uint8)
    max_steps = max(1, int(round(max_distance_px)))
    for _ in range(max_steps):
        nxt = cv2.dilate(current, kernel, iterations=1)
        nxt = (nxt.astype(bool) & growable).astype(np.uint8)
        if np.array_equal(nxt, current):
            break
        current = nxt

    # Drop any expansion farther than max_distance from the original oval.
    dist_out = cv2.distanceTransform((~crown).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = current.astype(bool) & (
        crown | ((dist_out > 0.0) & (dist_out <= float(max_distance_px)))
    )
    return expanded


class CrownNormativeRegionEstimator:
    """
    Build a normative hair-bearing region for crown-view images.

    Step 1 — locate the vertex from image cues (whorl / hair flow / scalp).
    Step 2 — construct the full expected crown envelope around that point.
    Deficit analysis is deferred: ``normative − hair`` is handled elsewhere.
    """

    def __init__(
        self,
        posterior_extent: float = _DEFAULT_POSTERIOR_EXTENT,
        anterior_extent: float = _DEFAULT_ANTERIOR_EXTENT,
        lateral_extent: float = _DEFAULT_LATERAL_EXTENT,
        shape_exponent: float = _DEFAULT_SHAPE_EXPONENT,
        boundary_margin: float = _DEFAULT_BOUNDARY_MARGIN,
    ) -> None:
        if not 0.15 < posterior_extent <= 0.55:
            raise ValueError("posterior_extent must be in the interval (0.15, 0.55]")
        if not 0.10 < anterior_extent <= 0.45:
            raise ValueError("anterior_extent must be in the interval (0.10, 0.45]")
        if not 0.18 < lateral_extent <= 0.80:
            raise ValueError("lateral_extent must be in the interval (0.18, 0.80]")
        if not 2.0 < shape_exponent <= 4.0:
            raise ValueError("shape_exponent must be in the interval (2, 4]")
        if not 0.0 <= boundary_margin <= 0.12:
            raise ValueError("boundary_margin must be in the interval [0, 0.12]")

        self.posterior_extent = posterior_extent
        self.anterior_extent = anterior_extent
        self.lateral_extent = lateral_extent
        self.shape_exponent = shape_exponent
        self.boundary_margin = boundary_margin

    def estimate(
        self,
        image: Image.Image,
        head_mask: np.ndarray,
        hair_mask: np.ndarray,
    ) -> NormativeRegionResult:
        """
        Estimate the crown normative hair-bearing region.

        Parameters
        ----------
        image:
            RGB crown photo aligned with the segmentation masks.
        head_mask:
            Boolean head segmentation mask aligned with ``image``.
        hair_mask:
            Boolean hair segmentation mask aligned with ``image``.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        width, height = image.size
        head = _as_bool_mask(head_mask, "head_mask")
        hair = _as_bool_mask(hair_mask, "hair_mask")
        if head.shape != (height, width):
            raise ValueError(
                f"head_mask must match image dimensions (expected {(height, width)})"
            )
        if hair.shape != (height, width):
            raise ValueError(
                f"hair_mask must match image dimensions (expected {(height, width)})"
            )

        box_w, box_h, head_pixels, radius, y_min, y_max, x_min, x_max = _head_scale(head)
        frame = estimate_head_frame(head)
        if frame is None:
            raise ValueError("Could not derive a head coordinate frame from head_mask")

        vertex = estimate_vertex_from_image(
            image,
            head,
            hair,
            equivalent_radius=radius,
        )
        vertex_x, vertex_y = vertex.x, vertex.y

        crown = build_crown_normative_mask(
            head,
            frame,
            vertex_x=vertex_x,
            vertex_y=vertex_y,
            equivalent_radius=radius,
            posterior_extent=self.posterior_extent,
            anterior_extent=self.anterior_extent,
            lateral_extent=self.lateral_extent,
            shape_exponent=self.shape_exponent,
            boundary_margin=self.boundary_margin,
        )
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        expand_px = max(12.0, 0.28 * radius)
        crown = _expand_crown_into_adjacent_visible_scalp(
            crown,
            head,
            gray,
            max_distance_px=expand_px,
            luminance_floor=82.0,
        )

        empty = np.zeros((height, width), dtype=bool)
        crown_pixels = int(np.count_nonzero(crown))
        crown_fraction = crown_pixels / head_pixels if head_pixels > 0 else 0.0
        semi_post = self.posterior_extent * frame.major_span
        semi_ant = self.anterior_extent * frame.major_span
        semi_lat = 0.5 * self.lateral_extent * frame.minor_span
        axis_ratio = semi_post / semi_lat if semi_lat > 0 else 0.0

        return NormativeRegionResult(
            normative_region_mask=crown,
            region_masks=RegionMasks(
                frontal=empty,
                left_temple=empty,
                right_temple=empty,
                crown=crown,
            ),
            view="crown",
            metadata={
                "head_bbox_width": box_w,
                "head_bbox_height": box_h,
                "head_pixels": int(head_pixels),
                "equivalent_radius": radius,
                "head_major_span": frame.major_span,
                "head_minor_span": frame.minor_span,
                "head_angle_deg": float(np.degrees(frame.angle_rad)),
                "crown_pixels": crown_pixels,
                "crown_fraction_of_head": crown_fraction,
                "vertex_x": vertex_x,
                "vertex_y": vertex_y,
                "vertex_method": vertex.method,
                "vertex_confidence": vertex.confidence,
                "vertex_orientation_score": vertex.orientation_score,
                "vertex_scalp_weight": vertex.scalp_weight,
                "extent_posterior": semi_post,
                "extent_anterior": semi_ant,
                "extent_lateral": semi_lat,
                "axis_ratio_posterior_over_lateral": axis_ratio,
                "shape_exponent": self.shape_exponent,
                "posterior_extent": self.posterior_extent,
                "anterior_extent": self.anterior_extent,
                "lateral_extent": self.lateral_extent,
                "boundary_margin": self.boundary_margin,
                "head_y_min": y_min,
                "head_y_max": y_max,
                "head_x_min": x_min,
                "head_x_max": x_max,
            },
        )
