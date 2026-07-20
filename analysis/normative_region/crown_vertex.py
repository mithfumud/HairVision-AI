"""Image-based anatomical vertex (hair whorl) estimation for crown view."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from analysis.front_features import _as_bool_mask

# Minimum gradient magnitude percentile within the search region.
_DEFAULT_GRADIENT_PERCENTILE = 55.0
# Maximum orientation sample points for scoring.
_DEFAULT_MAX_SAMPLES = 6000
# Coarse grid steps along each axis for whorl search.
_DEFAULT_GRID_STEPS = 21
# Weight for scalp-exposure cue when fusing with orientation peak.
_DEFAULT_SCALP_FUSION_WEIGHT = 0.35
# Minimum orientation peak sharpness (z-score) to trust flow cue alone.
_DEFAULT_ORIENTATION_CONFIDENCE_Z = 0.75


@dataclass(frozen=True)
class VertexEstimate:
    """Detected anatomical crown center and cue diagnostics."""

    x: float
    y: float
    confidence: float
    method: str
    orientation_score: float
    scalp_weight: float


def _search_region(
    head_mask: np.ndarray,
    margin_fraction: float = 0.10,
) -> tuple[np.ndarray, int, int, int, int, float, float]:
    """Central scalp band where the whorl may appear (excludes head rim)."""
    head = np.asarray(head_mask).astype(bool)
    ys, xs = np.where(head)
    if len(xs) == 0:
        empty = np.zeros_like(head, dtype=bool)
        return empty, 0, 0, 0, 0, 0.0, 0.0

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    box_h = float(y_max - y_min + 1)
    box_w = float(x_max - x_min + 1)

    search = np.zeros_like(head, dtype=bool)
    my = int(round(margin_fraction * box_h))
    mx = int(round(margin_fraction * box_w))
    search[
        y_min + my : y_max - my + 1,
        x_min + mx : x_max - mx + 1,
    ] = True
    search &= head
    return search, y_min, y_max, x_min, x_max, box_w, box_h


def _hair_orientation_samples(
    gray: np.ndarray,
    search_mask: np.ndarray,
    *,
    gradient_percentile: float,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample pixel locations, hair-flow angles, and weights from image gradients."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.hypot(gx, gy)

    active = search_mask & (magnitude > 0.0)
    if not np.any(active):
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    mags = magnitude[active]
    threshold = float(np.percentile(mags, gradient_percentile))
    strong = active & (magnitude >= threshold)
    ys, xs = np.where(strong)
    if len(xs) == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    if len(xs) > max_samples:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(xs), size=max_samples, replace=False)
        ys = ys[pick]
        xs = xs[pick]

    # Gradient is across strands; hair flows along the isophote direction.
    flow_x = -gy[ys, xs]
    flow_y = gx[ys, xs]
    thetas = np.arctan2(flow_y, flow_x)
    weights = magnitude[ys, xs]
    weights = weights / max(float(weights.max()), 1e-6)
    return (
        xs.astype(np.float64),
        ys.astype(np.float64),
        thetas,
        weights.astype(np.float64),
    )


def _orientation_alignment_score(
    cx: float,
    cy: float,
    xs: np.ndarray,
    ys: np.ndarray,
    thetas: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Score how well local hair flow radiates from ``(cx, cy)``.

    At the whorl, strand directions align with rays from the convergence point.
    """
    if len(xs) == 0:
        return 0.0

    radial = np.arctan2(ys - cy, xs - cx)
    diff = thetas - radial
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    diff = np.minimum(np.abs(diff), np.pi - np.abs(diff))
    alignment = np.cos(2.0 * diff)
    return float(np.sum(weights * alignment))


def _score_grid(
    cx_values: np.ndarray,
    cy_values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    thetas: np.ndarray,
    weights: np.ndarray,
    head: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Evaluate orientation scores on a 2-D candidate grid."""
    height, width = head.shape
    best_score = -np.inf
    best_x = 0.0
    best_y = 0.0
    score_map = np.full((len(cy_values), len(cx_values)), -np.inf, dtype=np.float64)

    for row, cy in enumerate(cy_values):
        iy = int(round(cy))
        if iy < 0 or iy >= height:
            continue
        for col, cx in enumerate(cx_values):
            ix = int(round(cx))
            if ix < 0 or ix >= width or not head[iy, ix]:
                continue
            score = _orientation_alignment_score(cx, cy, xs, ys, thetas, weights)
            score_map[row, col] = score
            if score > best_score:
                best_score = score
                best_x = float(cx)
                best_y = float(cy)

    return best_x, best_y, best_score, score_map


def _scalp_exposure_centroid(
    head_mask: np.ndarray,
    hair_mask: np.ndarray,
    search_mask: np.ndarray,
    blur_sigma: float,
) -> tuple[float, float, float]:
    """Weighted centroid of exposed scalp — often peaks at the crown whorl."""
    head = np.asarray(head_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)
    scalp = (head & ~hair).astype(np.float32)
    scalp[~search_mask] = 0.0

    if not np.any(scalp):
        return 0.0, 0.0, 0.0

    ksize = max(3, int(round(blur_sigma * 6)) | 1)
    blurred = cv2.GaussianBlur(scalp, (ksize, ksize), blur_sigma)
    total = float(blurred.sum())
    if total <= 1e-3:
        return 0.0, 0.0, 0.0

    ys, xs = np.indices(blurred.shape)
    cx = float((xs * blurred).sum() / total)
    cy = float((ys * blurred).sum() / total)
    peak = float(blurred.max())
    weight = min(1.0, peak * 4.0)
    return cx, cy, weight


def _appearance_thinning_centroid(
    gray: np.ndarray,
    search_mask: np.ndarray,
    *,
    equivalent_radius: float,
) -> tuple[float, float, float]:
    """
    Centroid of appearance-based diffuse thinning inside the search band.

    When semantic ``hair_mask`` saturates the crown (sparse thinning still
    labelled hair), ``head ∧ ¬hair`` is empty and the scalp-exposure cue
    fails. Local dark-strand occupancy + elevated luminance recover the
    clinical thinning center:

        S(x) = (K ∗ 1[L ≤ P_40(L|search)])(x)
        w(x) = elev(L) · (1 − S)_+
        vertex ≈ centroid(G_σ ∗ w | search)
    """
    search = np.asarray(search_mask).astype(bool)
    if not np.any(search):
        return 0.0, 0.0, 0.0

    g = gray.astype(np.float32)
    vals = g[search]
    dark_cut = float(np.percentile(vals, 40.0))
    lum_lo = float(np.percentile(vals, 45.0))

    dark = search & (g <= dark_cut)
    radius = max(3, int(round(0.04 * max(equivalent_radius, 8.0))))
    k = 2 * radius + 1
    strand_occ = cv2.blur(dark.astype(np.float32), (k, k))

    elev = np.clip((g - lum_lo) / 40.0, 0.0, 1.0)
    sparse = np.clip(1.0 - strand_occ / 0.55, 0.0, 1.0)
    # Light-scalp mass (slightly softer floor so shadowed left patches count).
    light_scalp = (g >= 90.0).astype(np.float32)
    weight = np.zeros_like(g, dtype=np.float32)
    weight[search] = light_scalp[search] * (
        0.50 + 0.50 * np.maximum(sparse[search], 0.20)
    )
    # Mild boost from elevated sparse regions that are not quite L>=90.
    soft = (elev[search] > 0.35) & (sparse[search] > 0.35) & (light_scalp[search] < 0.5)
    weight[search] = np.where(
        soft,
        weight[search] + 0.25 * elev[search] * sparse[search],
        weight[search],
    )

    if float(weight[search].sum()) <= 1e-3:
        return 0.0, 0.0, 0.0

    blur_sigma = max(2.5, 0.05 * equivalent_radius)
    ksize = max(3, int(round(blur_sigma * 6)) | 1)
    blurred = cv2.GaussianBlur(weight, (ksize, ksize), blur_sigma)
    blurred[~search] = 0.0
    total = float(blurred.sum())
    if total <= 1e-3:
        return 0.0, 0.0, 0.0

    ys, xs = np.indices(blurred.shape)
    # Equal left/right lobe vote: mass-weighted global centroid is dominated
    # by the brighter/larger bald patch and leaves the opposite side outside
    # the crown envelope.
    search_xs = xs[search]
    mid_x = float(search_xs.mean())
    left = search & (xs < mid_x) & (blurred > 1e-4)
    right = search & (xs >= mid_x) & (blurred > 1e-4)
    left_mass = float(blurred[left].sum()) if np.any(left) else 0.0
    right_mass = float(blurred[right].sum()) if np.any(right) else 0.0
    min_lobe = 0.08 * total
    if left_mass >= min_lobe and right_mass >= min_lobe:
        lx = float((xs[left] * blurred[left]).sum() / left_mass)
        ly = float((ys[left] * blurred[left]).sum() / left_mass)
        rx = float((xs[right] * blurred[right]).sum() / right_mass)
        ry = float((ys[right] * blurred[right]).sum() / right_mass)
        cx = 0.5 * (lx + rx)
        cy = 0.5 * (ly + ry)
    else:
        cx = float((xs * blurred).sum() / total)
        cy = float((ys * blurred).sum() / total)

    peak = float(blurred.max())
    # Strength reflects how dominant the thinning mass is in the search band.
    mass_fraction = float(np.count_nonzero(weight[search] > 0.15)) / max(
        1, int(np.count_nonzero(search))
    )
    strength = float(np.clip(0.35 + 1.2 * mass_fraction + 0.4 * peak, 0.0, 1.0))
    return cx, cy, strength


def estimate_vertex_from_image(
    image: Image.Image,
    head_mask: np.ndarray,
    hair_mask: np.ndarray,
    *,
    equivalent_radius: float,
    gradient_percentile: float = _DEFAULT_GRADIENT_PERCENTILE,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    grid_steps: int = _DEFAULT_GRID_STEPS,
    scalp_fusion_weight: float = _DEFAULT_SCALP_FUSION_WEIGHT,
    orientation_confidence_z: float = _DEFAULT_ORIENTATION_CONFIDENCE_Z,
) -> VertexEstimate:
    """
    Locate the anatomical crown center from image cues.

    Combines hair-flow convergence (whorl radial pattern) with scalp /
    appearance thinning centroids. When segmentation labels sparse crown
    hair as hair everywhere, appearance thinning replaces the empty
    semantic-scalp cue so the envelope stays centered on the clinical vertex.
    """
    head = _as_bool_mask(head_mask, "head_mask")
    hair = _as_bool_mask(hair_mask, "hair_mask")
    if head.shape != hair.shape:
        raise ValueError("head_mask and hair_mask must have the same shape")

    search, y_min, y_max, x_min, x_max, box_w, box_h = _search_region(head)
    if not np.any(search):
        cy = 0.5 * (y_min + y_max)
        cx = 0.5 * (x_min + x_max)
        return VertexEstimate(cx, cy, 0.0, "fallback_center", 0.0, 0.0)

    rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != head.shape:
        raise ValueError("image dimensions must match head_mask")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    xs_s, ys_s, thetas, weights = _hair_orientation_samples(
        gray,
        search,
        gradient_percentile=gradient_percentile,
        max_samples=max_samples,
    )

    mx = int(round(0.10 * box_w))
    my = int(round(0.10 * box_h))
    cx_grid = np.linspace(x_min + mx, x_max - mx, grid_steps)
    cy_grid = np.linspace(y_min + my, y_max - my, grid_steps)

    ori_x, ori_y, ori_score, score_map = _score_grid(
        cx_grid, cy_grid, xs_s, ys_s, thetas, weights, head
    )

    finite = score_map[np.isfinite(score_map)]
    if len(finite) == 0 or ori_score <= -np.inf:
        ori_conf = 0.0
    else:
        mean = float(finite.mean())
        std = float(finite.std())
        ori_conf = (ori_score - mean) / std if std > 1e-6 else 0.0

    # Refine around the orientation peak.
    refine_span = max(8.0, 0.08 * equivalent_radius)
    refine_steps = 9
    cx_fine = np.linspace(ori_x - refine_span, ori_x + refine_span, refine_steps)
    cy_fine = np.linspace(ori_y - refine_span, ori_y + refine_span, refine_steps)
    ori_x, ori_y, ori_score, _ = _score_grid(
        cx_fine, cy_fine, xs_s, ys_s, thetas, weights, head
    )

    blur_sigma = max(2.5, 0.06 * equivalent_radius)
    scalp_x, scalp_y, scalp_strength = _scalp_exposure_centroid(
        head, hair, search, blur_sigma
    )

    scalp_fraction = float(np.count_nonzero((head & ~hair) & search)) / max(
        1, int(np.count_nonzero(search))
    )

    # Appearance thinning centroid: used when semantic scalp is saturated.
    app_x, app_y, app_strength = _appearance_thinning_centroid(
        gray,
        search,
        equivalent_radius=equivalent_radius,
    )

    # Prefer semantic scalp when present; otherwise fall back to appearance.
    if scalp_strength > 0.05 and scalp_fraction >= 0.01:
        cue_x, cue_y = scalp_x, scalp_y
        cue_strength = scalp_strength
        cue_name = "scalp"
        adaptive_cue_w = float(
            np.clip(
                scalp_fusion_weight * scalp_strength * (1.0 + 2.0 * scalp_fraction),
                0.0,
                0.70,
            )
        )
    elif app_strength > 0.05:
        cue_x, cue_y = app_x, app_y
        cue_strength = app_strength
        cue_name = "appearance"
        # Hair-mask saturation → trust appearance strongly over flow
        # (flow often locks onto one lit side of a bilateral bald pattern).
        saturation = 1.0 - float(np.clip(scalp_fraction * 20.0, 0.0, 1.0))
        adaptive_cue_w = float(
            np.clip(0.88 + 0.12 * app_strength * saturation, 0.85, 1.0)
        )
    else:
        cue_x, cue_y, cue_strength, cue_name = 0.0, 0.0, 0.0, "none"
        adaptive_cue_w = 0.0

    if len(xs_s) == 0 and cue_strength <= 0.0:
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        return VertexEstimate(cx, cy, 0.0, "fallback_center", 0.0, 0.0)

    if cue_strength <= 0.0:
        confidence = float(np.clip(ori_conf / 3.0, 0.0, 1.0))
        return VertexEstimate(
            ori_x, ori_y, confidence, "hair_flow", ori_score, 0.0
        )

    ori_trust = float(np.clip(ori_conf / orientation_confidence_z, 0.0, 1.0))
    cue_w = adaptive_cue_w
    # Appearance can fully dominate when semantic scalp is empty — hair-flow
    # often locks onto one lit lobe of a bilateral bald pattern.
    ori_floor = 0.0 if cue_name == "appearance" else 0.15
    ori_w = max(ori_floor, ori_trust * (1.0 - cue_w))
    total_w = ori_w + cue_w
    vertex_x = (ori_w * ori_x + cue_w * cue_x) / total_w
    vertex_y = (ori_w * ori_y + cue_w * cue_y) / total_w
    confidence = float(
        np.clip(max(ori_conf / 3.0, cue_strength * 0.5), 0.0, 1.0)
    )
    method = f"fused_flow_{cue_name}"

    return VertexEstimate(
        vertex_x,
        vertex_y,
        confidence,
        method,
        ori_score,
        cue_w,
    )
