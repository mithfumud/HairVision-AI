"""Hair coverage refinement for clinical crown deficit.

Semantic ``hair_mask`` often labels sparse thinning as hair. This module
demotes only pixels where the *scalp is visibly exposed* (light scalp through
hair), so clinical deficit ``normative ∩ ¬hair_coverage`` follows the
dermatologist outline of the thinning region — not merely lower hair density.

Does not change HairDeficitAnalyzer or HairLossMetricsExtractor formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def _head_radius_px(head_mask: np.ndarray) -> float:
    area = float(np.count_nonzero(head_mask))
    if area <= 0:
        return 16.0
    return max(8.0, float(np.sqrt(area / np.pi)))


def _odd_kernel(radius: int) -> int:
    r = max(1, int(radius))
    k = 2 * r + 1
    return k if k % 2 == 1 else k + 1


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    k = _odd_kernel(radius)
    return cv2.blur(values.astype(np.float32), (k, k))


@dataclass(frozen=True)
class CoverageRefineResult:
    """Outputs of coverage refinement for debugging and clinical use."""

    hair_coverage_mask: np.ndarray
    density_map: np.ndarray
    sparse_exposed_mask: np.ndarray
    density_threshold: float
    window_radius_px: int
    pixels_demoted: int
    scalp_visibility_map: np.ndarray | None = None


class HairCoverageRefiner:
    """
    Build a coverage mask using scalp *visibility* as the primary cue.

    Mathematical model (crown)
    --------------------------
    Dense-hair luminance from the full head (side/back dense strands):

        L_dense = P_75(L | dense_head)

    Continuous scalp-visibility map V ∈ [0, 1]:

        bright(x) = clip( (L(x) − L_dense − δ₀) / δ_scale , 0, 1 )
        exposed_frac(x) = (K_ρ ∗ 1[L > L_dense + δ_skin])(x)
        tophat(x) = L − open(L)     # bright holes amid darker strands
        V(x) = bright(x) · exposed_frac(x) · clip(tophat / τ_th, 0, 1)

    Interpretation:
      • High V  → light scalp visibly showing through hair (include)
      • Low V   → dark / uniformly lit covered hair (exclude)
      • Top-hat suppresses whole-side lighting that is bright but not scalp holes

    Seeds and growth (visibility-constrained, not density expansion):

        seeds = G ∧ (V ≥ τ_seed)      # τ_seed ≈ 0.60
        cand  = G ∧ (V ≥ τ_grow)      # τ_grow ≈ 0.42
        T     = geodesic_dilate(seeds | cand)   # only through significant V
        T     ← T ∧ (dist_to_seeds ≤ r_halo)    # stay near exposed core
        T     ← drop tiny components

    Clinical path (unchanged deficit & metrics formulas):

        hair_coverage = H ∧ ¬T   inside G
        deficit       = clean(normative ∩ ¬hair_coverage)
    """

    def __init__(
        self,
        *,
        occupancy_weight: float = 0.50,
        probability_weight: float = 0.30,
        visibility_weight: float = 0.20,
        window_radius_fraction: float = 0.035,
        dense_reference_percentile: float = 25.0,
        min_density_threshold: float = 0.42,
        low_probability_cut: float = 0.55,
        # Visibility model
        bright_offset: float = 28.0,
        bright_scale: float = 50.0,
        skin_offset: float = 45.0,
        seed_visibility: float = 0.52,
        grow_visibility: float = 0.32,
        halo_radius_fraction: float = 0.15,
        min_seed_luminance: float = 98.0,
        min_grow_luminance: float = 88.0,
        tophat_scale: float = 18.0,
        min_thinning_area_fraction: float = 0.002,
        close_radius_fraction: float = 0.010,
    ) -> None:
        w = occupancy_weight + probability_weight + visibility_weight
        if abs(w - 1.0) > 1e-6:
            raise ValueError("density weights must sum to 1")
        if not 0.0 < window_radius_fraction < 0.2:
            raise ValueError("window_radius_fraction out of range")
        if not 0.0 < grow_visibility <= seed_visibility <= 1.0:
            raise ValueError("invalid visibility thresholds")
        self.occupancy_weight = occupancy_weight
        self.probability_weight = probability_weight
        self.visibility_weight = visibility_weight
        self.window_radius_fraction = window_radius_fraction
        self.dense_reference_percentile = dense_reference_percentile
        self.min_density_threshold = min_density_threshold
        self.low_probability_cut = low_probability_cut
        self.bright_offset = bright_offset
        self.bright_scale = bright_scale
        self.skin_offset = skin_offset
        self.seed_visibility = seed_visibility
        self.grow_visibility = grow_visibility
        self.halo_radius_fraction = halo_radius_fraction
        self.min_seed_luminance = min_seed_luminance
        self.min_grow_luminance = min_grow_luminance
        self.tophat_scale = tophat_scale
        self.min_thinning_area_fraction = min_thinning_area_fraction
        self.close_radius_fraction = close_radius_fraction

    def refine(
        self,
        image: Image.Image | np.ndarray,
        hair_mask: np.ndarray,
        head_mask: np.ndarray,
        *,
        hair_probability: np.ndarray | None = None,
        skin_mask: np.ndarray | None = None,
        gate_mask: np.ndarray | None = None,
    ) -> CoverageRefineResult:
        """
        Return coverage-refined hair mask.

        Parameters
        ----------
        gate_mask
            Optional spatial gate (e.g. crown normative). Defaults to head.
        """
        del skin_mask  # not used; visibility is appearance-based
        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            rgb = np.asarray(image)
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)

        hair = _as_bool(hair_mask)
        head = _as_bool(head_mask)
        gate = head if gate_mask is None else (_as_bool(gate_mask) & head)

        if hair.shape != head.shape:
            raise ValueError("hair_mask and head_mask shapes must match")
        if rgb.shape[:2] != hair.shape:
            raise ValueError("image size must match mask shapes")

        head_r = _head_radius_px(head)
        radius = max(3, int(round(self.window_radius_fraction * head_r)))

        d_occ = _box_mean(hair.astype(np.float32), radius)
        if hair_probability is not None:
            p = np.asarray(hair_probability, dtype=np.float32)
            if p.shape != hair.shape:
                raise ValueError("hair_probability shape must match hair_mask")
            d_prob = _box_mean(p, radius)
        else:
            p = hair.astype(np.float32)
            d_prob = d_occ

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        visibility = self._scalp_visibility_map(gray, head, head_r)
        density = (
            self.occupancy_weight * d_occ
            + self.probability_weight * d_prob
            + self.visibility_weight * (1.0 - visibility)
        )
        density = np.clip(density, 0.0, 1.0).astype(np.float32)
        tau = self._adaptive_threshold(density, hair, p)

        # Visibility-primary exposed-scalp region (no density-side expansion).
        # Absolute luminance floors keep mid-tone lit hair from counting as scalp.
        demote = self._visibility_deficit_field(
            visibility,
            gray,
            gate,
            head,
            head_r,
        )
        demote = demote & hair & gate

        hair_coverage = hair & ~demote
        hair_coverage = np.where(gate, hair_coverage, hair).astype(bool)

        sparse_exposed = gate & ((visibility >= self.grow_visibility) | (~hair) | demote)

        return CoverageRefineResult(
            hair_coverage_mask=hair_coverage,
            density_map=density,
            sparse_exposed_mask=sparse_exposed.astype(bool),
            density_threshold=float(tau),
            window_radius_px=radius,
            pixels_demoted=int(np.count_nonzero(demote)),
            scalp_visibility_map=visibility,
        )

    def _dense_luminance_reference(
        self,
        gray: np.ndarray,
        head: np.ndarray,
        head_r: float,
    ) -> float:
        """Luminance of dense dark hair on the full head."""
        if not np.any(head):
            return 255.0
        dark_cut = float(np.percentile(gray[head], 35.0))
        dark = head & (gray <= dark_cut)
        strand_radius = max(3, int(round(self.window_radius_fraction * head_r)))
        strand_occ = _box_mean(dark.astype(np.float32), strand_radius)
        dense_floor = float(np.percentile(strand_occ[head], 70))
        dense_core = head & (strand_occ >= dense_floor)
        if int(np.count_nonzero(dense_core)) >= 200:
            return float(np.percentile(gray[dense_core], 75))
        return float(np.percentile(gray[head], 40.0))

    def _scalp_visibility_map(
        self,
        gray: np.ndarray,
        head: np.ndarray,
        head_r: float,
    ) -> np.ndarray:
        """
        Continuous scalp-visibility map V ∈ [0, 1].

        High only where luminance is well above dense hair *and* a local
        neighborhood contains exposed (light) scalp — residual strands allowed.
        Dark, well-covered hair maps near 0.
        """
        vis = np.zeros(gray.shape, dtype=np.float32)
        if not np.any(head):
            return vis

        lum_dense = self._dense_luminance_reference(gray, head, head_r)
        bright = np.clip(
            (gray - (lum_dense + self.bright_offset)) / max(self.bright_scale, 1.0),
            0.0,
            1.0,
        ).astype(np.float32)

        skin_cut = lum_dense + self.skin_offset
        exposed_bin = (gray >= skin_cut).astype(np.float32)
        strand_radius = max(3, int(round(self.window_radius_fraction * head_r)))
        exposed_frac = _box_mean(exposed_bin, strand_radius)

        # Top-hat: bright scalp holes relative to local darker hair structure.
        # Uniformly lit covered hair has low top-hat; speckled exposed scalp is high.
        open_r = max(3, int(round(0.045 * head_r)))
        open_k = _odd_kernel(open_r)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, open_kernel)
        tophat = np.clip(gray - opened, 0.0, None)
        tophat_score = np.clip(tophat / max(self.tophat_scale, 1.0), 0.0, 1.0)

        # Keep a floor so solid bright scalp interiors (low top-hat) still count,
        # while uniformly lit covered hair (low top-hat + weaker exposed_frac)
        # stays suppressed.
        structure = np.clip(0.55 + 0.45 * tophat_score, 0.0, 1.0)
        vis = bright * exposed_frac * structure
        vis = np.clip(vis, 0.0, 1.0).astype(np.float32)
        vis[~head] = 0.0
        return vis

    def _visibility_deficit_field(
        self,
        visibility: np.ndarray,
        gray: np.ndarray,
        gate: np.ndarray,
        head: np.ndarray,
        head_r: float,
    ) -> np.ndarray:
        """
        Contiguous exposed-scalp deficit from visibility seeds.

        Grows only through pixels with significant V *and* light absolute
        luminance (white/light scalp), within a limited halo of the seeds.
        """
        if not np.any(gate):
            return np.zeros_like(gate, dtype=bool)

        seeds = (
            gate
            & (visibility >= self.seed_visibility)
            & (gray >= self.min_seed_luminance)
        )
        candidates = (
            gate
            & (visibility >= self.grow_visibility)
            & (gray >= self.min_grow_luminance)
        )
        candidates = candidates | seeds

        if not np.any(seeds):
            if not np.any(candidates):
                return np.zeros_like(gate, dtype=bool)
            vals = visibility[candidates]
            cut = float(np.percentile(vals, 85))
            seeds = (
                candidates
                & (visibility >= max(cut, self.grow_visibility))
                & (gray >= self.min_grow_luminance)
            )

        thinning = self._geodesic_dilate(
            seeds,
            candidates,
            head_r,
            step_fraction=0.008,
            max_iters=40,
        )

        halo_r = max(4, int(round(self.halo_radius_fraction * head_r)))
        dist = cv2.distanceTransform(
            (~seeds).astype(np.uint8) * 255,
            cv2.DIST_L2,
            5,
        )
        thinning = thinning & (dist <= float(halo_r))

        close_r = max(1, int(round(self.close_radius_fraction * head_r)))
        close_k = _odd_kernel(close_r)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        thinning = cv2.morphologyEx(
            thinning.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            kernel,
        ).astype(bool)
        thinning &= candidates

        return self._remove_small_components(
            thinning,
            head,
            min_area_fraction=self.min_thinning_area_fraction,
        )

    @staticmethod
    def _geodesic_dilate(
        seeds: np.ndarray,
        mask: np.ndarray,
        head_r: float,
        *,
        step_fraction: float = 0.008,
        max_iters: int = 40,
    ) -> np.ndarray:
        """Small-step geodesic dilation under ``mask`` (visibility-limited)."""
        if not np.any(seeds) or not np.any(mask):
            return np.zeros_like(mask, dtype=bool)

        radius = max(1, int(round(step_fraction * head_r)))
        k = _odd_kernel(radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        current = (seeds & mask).astype(np.uint8) * 255
        limit = mask.astype(np.uint8) * 255
        for _ in range(max_iters):
            dilated = cv2.dilate(current, kernel)
            next_mask = cv2.bitwise_and(dilated, limit)
            if np.array_equal(next_mask, current):
                break
            current = next_mask
        return current.astype(bool)

    def _adaptive_threshold(
        self,
        density: np.ndarray,
        hair: np.ndarray,
        probability: np.ndarray,
    ) -> float:
        """τ from low percentile of density on dense-looking hair pixels."""
        dense_ref = hair & (probability >= max(0.70, self.low_probability_cut + 0.15))
        if np.count_nonzero(dense_ref) < 500:
            dense_ref = hair
        if not np.any(dense_ref):
            return float(self.min_density_threshold)

        tau_raw = float(
            np.percentile(
                density[dense_ref],
                self.dense_reference_percentile,
            )
        )
        tau_cap = 0.58
        return float(max(self.min_density_threshold, min(tau_cap, tau_raw)))

    @staticmethod
    def _remove_small_components(
        mask: np.ndarray,
        head_mask: np.ndarray,
        *,
        min_area_fraction: float = 0.0015,
    ) -> np.ndarray:
        """Keep only components large enough relative to head area."""
        area = np.asarray(mask).astype(bool)
        if not np.any(area):
            return area
        head_area = max(int(np.count_nonzero(head_mask)), 1)
        min_area = max(64, int(min_area_fraction * head_area))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            area.astype(np.uint8),
            connectivity=8,
        )
        kept = np.zeros_like(area, dtype=bool)
        for label in range(1, n_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                kept[labels == label] = True
        return kept
