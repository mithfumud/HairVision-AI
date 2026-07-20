"""
Manual visual QA for CrownNormativeRegionEstimator.

The annotated overlay shows hair-bearing areas in green and exposed scalp
(hair loss) in red — medical lesion-style. The normative crown mask is
internal and never drawn.

    python tests/test_normative_crown.py
    python tests/test_normative_crown.py path/to/crown_image.jpg
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.normative_region.crown_estimator import CrownNormativeRegionEstimator
from analysis.quality import ImageQualityChecker
from analysis.validation.view_validator import ViewValidator, print_view_validation_result
from models.segmentation import HairSegmenter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_STEM = "mi_crown"
DEFAULT_IMAGE_DIR = ROOT / "test_images" / "crown"
OUTPUT_DIR = ROOT / "outputs" / "normative_crown"

# Medical lesion-style annotation for exposed scalp (hair loss only).
LESION_FILL_COLOR = (255, 95, 55)
LESION_FILL_ALPHA = 0.50
LESION_OUTLINE_COLOR = (170, 35, 0)
LESION_OUTLINE_WIDTH = 2
MIN_EXPOSED_SCALP_AREA_RATIO = 0.002

# Semi-transparent green tint for hair-bearing regions.
HAIR_FILL_COLOR = (72, 200, 96)
HAIR_FILL_ALPHA = 0.38


def resolve_default_image() -> Path:
    """Return the default crown test image path, trying common extensions."""
    for ext in (".jpeg", ".jpg", ".png"):
        candidate = DEFAULT_IMAGE_DIR / f"{DEFAULT_IMAGE_STEM}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No default crown image found for stem '{DEFAULT_IMAGE_STEM}' "
        f"under {DEFAULT_IMAGE_DIR}"
    )


# Hair softmax below this value inside normative ⇒ likely visible scalp / sparse hair.
_HAIR_PROB_EXPOSED_THRESHOLD = 0.90
# Morphological close radius (px) to merge fragmented visible-scalp patches.
_EXPOSED_CLOSE_RADIUS = 18


def _image_visible_scalp_mask(
    image_rgb: np.ndarray,
    normative_mask: np.ndarray,
    head_mask: np.ndarray,
    hair_probability: np.ndarray | None,
) -> np.ndarray:
    """
    Find visibly exposed scalp from image appearance inside the crown gate.

    Sparse crown thinning often stays labelled as hair in segmentation, but it
    appears as bright, relatively smooth skin tone compared with surrounding
    dense hair stubble.
    """
    normative = np.asarray(normative_mask).astype(bool)
    head = np.asarray(head_mask).astype(bool)
    gate = normative & head
    if not np.any(gate):
        return np.zeros_like(normative, dtype=bool)

    gray = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    gate_vals = gray[gate]
    lum_cut = float(np.percentile(gate_vals, 52))

    mean = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray * gray, (9, 9))
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    std_cut = float(np.percentile(local_std[gate], 42))

    bright_smooth = gate & (gray >= lum_cut) & (local_std <= std_cut)

    if hair_probability is not None:
        hair_prob = np.asarray(hair_probability, dtype=np.float32)
        uncertain = gate & (hair_prob <= _HAIR_PROB_EXPOSED_THRESHOLD)
        bright_smooth &= uncertain | (hair_prob <= 0.96)

    return bright_smooth


def _detect_exposed_scalp(
    normative_mask: np.ndarray,
    hair_mask: np.ndarray,
    head_mask: np.ndarray,
    *,
    image_rgb: np.ndarray | None = None,
    hair_probability: np.ndarray | None = None,
    skin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Exposed scalp inside the anatomical crown: where hair is missing or visibly thin.

    Combines segmentation gaps with image appearance cues so sparse crown
    thinning (still labelled "hair") is included. Normative is a spatial gate
    only — never drawn.
    """
    normative = np.asarray(normative_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)
    head = np.asarray(head_mask).astype(bool)
    gate = normative & head

    exposed = gate & ~hair

    if hair_probability is not None:
        hair_prob = np.asarray(hair_probability, dtype=np.float32)
        exposed |= gate & (hair_prob <= _HAIR_PROB_EXPOSED_THRESHOLD)

    if skin_mask is not None:
        exposed |= gate & np.asarray(skin_mask).astype(bool)

    if image_rgb is not None:
        exposed |= _image_visible_scalp_mask(
            image_rgb,
            normative,
            head,
            hair_probability,
        )

    exposed &= gate

    if not np.any(exposed):
        return exposed

    k = _EXPOSED_CLOSE_RADIUS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(exposed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool) & gate


def _clean_exposed_scalp_mask(
    exposed_mask: np.ndarray,
    head_mask: np.ndarray,
    *,
    min_area_ratio: float = MIN_EXPOSED_SCALP_AREA_RATIO,
) -> np.ndarray:
    """Remove tiny speckles; keep clinically meaningful exposed-scalp patches."""
    exposed = np.asarray(exposed_mask).astype(bool)
    head = np.asarray(head_mask).astype(bool)
    if not np.any(exposed):
        return np.zeros_like(exposed, dtype=bool)

    head_pixels = int(np.count_nonzero(head))
    min_area = max(1, int(min_area_ratio * head_pixels))

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        exposed.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(exposed, dtype=bool)
    for label in range(1, label_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = True
    return cleaned & head


def _extract_exposed_scalp_contours(
    exposed_mask: np.ndarray,
) -> list[np.ndarray]:
    """Return external contours of the exposed-scalp region(s)."""
    exposed_u8 = np.asarray(exposed_mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(exposed_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def _draw_hair_annotation(canvas: np.ndarray, hair_mask: np.ndarray) -> None:
    """Fill hair-bearing pixels with a semi-transparent green tint (in-place)."""
    hair = np.asarray(hair_mask).astype(bool)
    if not np.any(hair):
        return

    fill_layer = canvas.astype(np.float32)
    color = np.array(HAIR_FILL_COLOR, dtype=np.float32)
    fill_layer[hair] = (1.0 - HAIR_FILL_ALPHA) * fill_layer[hair] + HAIR_FILL_ALPHA * color
    canvas[:] = np.clip(fill_layer, 0, 255).astype(np.uint8)


def _draw_lesion_annotation(canvas: np.ndarray, exposed_mask: np.ndarray) -> None:
    """Fill exposed scalp and draw a clean boundary — lesion style (in-place)."""
    exposed = np.asarray(exposed_mask).astype(bool)
    if not np.any(exposed):
        return

    contours = _extract_exposed_scalp_contours(exposed)

    fill_layer = canvas.astype(np.float32)
    color = np.array(LESION_FILL_COLOR, dtype=np.float32)
    fill_layer[exposed] = (
        (1.0 - LESION_FILL_ALPHA) * fill_layer[exposed] + LESION_FILL_ALPHA * color
    )
    canvas[:] = np.clip(fill_layer, 0, 255).astype(np.uint8)

    if contours:
        cv2.drawContours(
            canvas,
            contours,
            -1,
            LESION_OUTLINE_COLOR,
            LESION_OUTLINE_WIDTH,
        )


def create_annotated_overlay(
    image: Image.Image,
    normative_mask: np.ndarray,
    hair_mask: np.ndarray,
    head_mask: np.ndarray,
    *,
    hair_probability: np.ndarray | None = None,
    skin_mask: np.ndarray | None = None,
) -> Image.Image:
    """
    Crown overlay — green hair + red exposed scalp.

    Original image + semi-transparent green on hair + lesion-style exposed
    scalp fill and contour. No head outline; normative ROI is never drawn.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    head = np.asarray(head_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)

    raw_exposed = _detect_exposed_scalp(
        normative_mask,
        hair_mask,
        head_mask,
        image_rgb=base,
        hair_probability=hair_probability,
        skin_mask=skin_mask,
    )
    exposed = _clean_exposed_scalp_mask(raw_exposed, head)

    rgb = base.copy()
    _draw_hair_annotation(rgb, hair & head)
    _draw_lesion_annotation(rgb, exposed)
    return Image.fromarray(rgb)


def _save_bool_mask(path: Path, mask: np.ndarray) -> None:
    """Save a boolean mask as a white-on-black PNG."""
    arr = (np.asarray(mask).astype(bool).astype(np.uint8)) * 255
    Image.fromarray(arr).save(path)


def _mask_stats(
    head_mask: np.ndarray,
    hair_mask: np.ndarray,
    normative_mask: np.ndarray,
    raw_deficit_mask: np.ndarray,
    cleaned_deficit_mask: np.ndarray,
) -> dict[str, int | float]:
    """Compute pixel counts and deficit ratios for pipeline diagnostics."""
    head = np.asarray(head_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)
    normative = np.asarray(normative_mask).astype(bool)
    raw_deficit = np.asarray(raw_deficit_mask).astype(bool)
    cleaned_deficit = np.asarray(cleaned_deficit_mask).astype(bool)

    head_pixels = int(np.count_nonzero(head))
    hair_pixels = int(np.count_nonzero(hair))
    normative_pixels = int(np.count_nonzero(normative))
    raw_deficit_pixels = int(np.count_nonzero(raw_deficit))
    cleaned_deficit_pixels = int(np.count_nonzero(cleaned_deficit))

    raw_ratio = raw_deficit_pixels / normative_pixels if normative_pixels else 0.0
    cleaned_ratio = (
        cleaned_deficit_pixels / normative_pixels if normative_pixels else 0.0
    )

    return {
        "head_pixels": head_pixels,
        "hair_pixels": hair_pixels,
        "normative_pixels": normative_pixels,
        "raw_deficit_pixels": raw_deficit_pixels,
        "cleaned_deficit_pixels": cleaned_deficit_pixels,
        "hair_inside_normative_pixels": int(np.count_nonzero(hair & normative)),
        "scalp_inside_normative_pixels": int(np.count_nonzero(head & ~hair & normative)),
        "hair_outside_normative_pixels": int(np.count_nonzero(hair & ~normative)),
        "raw_deficit_ratio": raw_ratio,
        "cleaned_deficit_ratio": cleaned_ratio,
    }


def print_pipeline_diagnostics(
    stats: dict[str, int | float],
    *,
    min_deficit_area: int,
    raw_component_count: int,
    raw_components: list[tuple[int, int]],
    cleaned_component_count: int,
    hair_probability_mean_in_normative: float | None = None,
) -> None:
    """Print mask counts, ratios, and evidence-based failure analysis."""
    print("Pipeline mask counts")
    print(f"  Head mask pixels:           {stats['head_pixels']:,}")
    print(f"  Hair mask pixels:           {stats['hair_pixels']:,}")
    print(f"  Crown normative pixels:     {stats['normative_pixels']:,}")
    print(f"  Raw deficit pixels:         {stats['raw_deficit_pixels']:,}")
    print(f"  Cleaned deficit pixels:     {stats['cleaned_deficit_pixels']:,}")
    print()
    print("Pipeline ratios")
    print(
        "  Raw deficit ratio:          "
        f"{100.0 * float(stats['raw_deficit_ratio']):.2f}% of normative"
    )
    print(
        "  Cleaned deficit ratio:      "
        f"{100.0 * float(stats['cleaned_deficit_ratio']):.2f}% of normative"
    )
    print()
    print("Normative overlap breakdown")
    print(
        "  Hair inside normative:      "
        f"{stats['hair_inside_normative_pixels']:,}"
    )
    print(
        "  Scalp inside normative:     "
        f"{stats['scalp_inside_normative_pixels']:,}"
    )
    print(
        "  Hair outside normative:     "
        f"{stats['hair_outside_normative_pixels']:,}"
    )
    if hair_probability_mean_in_normative is not None:
        print(
            "  Mean hair probability in normative: "
            f"{hair_probability_mean_in_normative:.3f}"
        )
    print()
    print("Deficit component analysis")
    print(f"  Min area threshold:         {min_deficit_area:,} px")
    print(f"  Raw components:             {raw_component_count}")
    if raw_components:
        for idx, (area, _) in enumerate(raw_components[:8], start=1):
            kept = "kept" if area >= min_deficit_area else "removed"
            print(f"    component {idx}: {area:,} px ({kept})")
        if len(raw_components) > 8:
            print(f"    ... {len(raw_components) - 8} more")
    print(f"  Cleaned components:         {cleaned_component_count}")
    print()
    print("Pipeline diagnosis")

    normative_pixels = int(stats["normative_pixels"])
    raw_deficit_pixels = int(stats["raw_deficit_pixels"])
    cleaned_deficit_pixels = int(stats["cleaned_deficit_pixels"])
    hair_inside = int(stats["hair_inside_normative_pixels"])
    scalp_inside = int(stats["scalp_inside_normative_pixels"])

    if normative_pixels == 0:
        print(
            "  Stage failing: CrownNormativeRegionEstimator\n"
            "  Evidence: crown_normative_mask is empty — no normative envelope was produced."
        )
        return

    if raw_deficit_pixels == 0 or (
        float(stats["raw_deficit_ratio"]) < 0.01
        and hair_inside / normative_pixels >= 0.95
    ):
        hair_fraction = hair_inside / normative_pixels
        print(
            "  Stage failing: HairSegmenter (hair_mask)\n"
            "  Evidence: raw_deficit = normative & ~hair is nearly empty, while "
            "crown_normative_mask is non-empty.\n"
            f"  Hair mask covers {100.0 * hair_fraction:.2f}% of the crown normative region "
            f"({hair_inside:,} / {normative_pixels:,} px).\n"
            f"  Scalp pixels inside normative (head & ~hair): {scalp_inside:,}."
        )
        if hair_probability_mean_in_normative is not None:
            print(
                f"  Mean hair softmax probability inside normative: "
                f"{hair_probability_mean_in_normative:.3f}."
            )
        print(
            "  Conclusion: visible crown thinning is being classified as HAIR by "
            "jonathandinu/face-parsing, leaving almost no scalp pixels for deficit."
        )
        if (
            raw_deficit_pixels > 0
            and cleaned_deficit_pixels == 0
            and raw_component_count > 0
        ):
            print(
                "  Secondary note: deficit cleaning in test_normative_crown.py also "
                f"removed the remaining {raw_deficit_pixels:,} raw px "
                f"(largest component {raw_components[0][0]:,} px < min_area "
                f"{min_deficit_area:,} px)."
            )
        return

    if cleaned_deficit_pixels == 0 and raw_deficit_pixels > 0:
        print(
            "  Stage failing: deficit cleaning in test_normative_crown.py\n"
            f"  Evidence: raw deficit has {raw_deficit_pixels:,} px across "
            f"{raw_component_count} component(s), but all are below min_area="
            f"{min_deficit_area:,}."
        )
        return

    if cleaned_deficit_pixels > 0 and float(stats["cleaned_deficit_ratio"]) < 0.01:
        print(
            "  Stage note: deficit exists but is very small relative to normative envelope.\n"
            f"  Cleaned deficit is only {100.0 * float(stats['cleaned_deficit_ratio']):.2f}% "
            "of normative — likely sparse hair segmentation inside the crown zone."
        )
        return

    print(
        "  No upstream empty-mask failure detected.\n"
        f"  Cleaned deficit = {cleaned_deficit_pixels:,} px "
        f"({100.0 * float(stats['cleaned_deficit_ratio']):.2f}% of normative)."
    )


def _deficit_component_info(
    deficit_mask: np.ndarray,
) -> tuple[int, list[tuple[int, int]]]:
    """Return component count and (area, label_id) pairs for deficit speckles."""
    deficit = np.asarray(deficit_mask).astype(bool)
    if not np.any(deficit):
        return 0, []

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        deficit.astype(np.uint8),
        connectivity=8,
    )
    components: list[tuple[int, int]] = []
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        components.append((area, label))
    components.sort(reverse=True)
    return len(components), components


def _check(name: str, passed: bool, reason: str) -> tuple[str, bool, str]:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed:
        print(f"         {reason}")
    return name, passed, reason


def run_validations(result, head_mask: np.ndarray) -> list[tuple[str, bool, str]]:
    """Run automatic checks and print PASS/FAIL for each."""
    normative = np.asarray(result.normative_region_mask).astype(bool)
    head = np.asarray(head_mask).astype(bool)
    masks = result.region_masks
    meta = result.metadata

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        _check(
            'view == "crown"',
            result.view == "crown",
            f"Expected view 'crown', got {result.view!r}.",
        )
    )

    normative_pixels = int(np.count_nonzero(normative))
    head_pixels = int(np.count_nonzero(head))
    checks.append(
        _check(
            "normative region is not empty",
            normative_pixels > 0,
            "Crown normative mask has zero pixels.",
        )
    )

    crown_fraction = float(meta.get("crown_fraction_of_head", 0.0))
    checks.append(
        _check(
            "normative region covers the anatomical crown envelope",
            0.18 <= crown_fraction <= 0.72,
            (
                f"Crown normative covers {100.0 * crown_fraction:.1f}% of head "
                "(expected roughly 18–72% — full crown envelope, not full scalp)."
            ),
        )
    )

    outside_head = int(np.count_nonzero(normative & ~head))
    checks.append(
        _check(
            "normative region is inside head mask",
            outside_head == 0,
            f"{outside_head} normative pixels fall outside the head mask.",
        )
    )

    frontal_pixels = int(np.count_nonzero(masks.frontal))
    left_pixels = int(np.count_nonzero(masks.left_temple))
    right_pixels = int(np.count_nonzero(masks.right_temple))
    checks.append(
        _check(
            "front/temple subregions are empty",
            frontal_pixels == 0 and left_pixels == 0 and right_pixels == 0,
            "Crown view should only populate the crown subregion.",
        )
    )

    crown_pixels = int(np.count_nonzero(masks.crown))
    checks.append(
        _check(
            "crown subregion matches normative union",
            crown_pixels == normative_pixels,
            f"crown mask ({crown_pixels:,}) != union ({normative_pixels:,}).",
        )
    )

    if head_pixels > 0:
        checks.append(
            _check(
                "normative region does not cover the entire scalp",
                normative_pixels <= int(0.60 * head_pixels),
                (
                    f"{normative_pixels:,} / {head_pixels:,} head pixels "
                    "in normative region — envelope may cover too much scalp."
                ),
            )
        )

    vertex_y = float(meta.get("vertex_y", 0.0))
    head_y_min = float(meta.get("head_y_min", 0.0))
    head_y_max = float(meta.get("head_y_max", 0.0))
    head_span = max(1.0, head_y_max - head_y_min)
    vertex_depth = (vertex_y - head_y_min) / head_span
    checks.append(
        _check(
            "vertex lies inside the central scalp",
            0.10 <= vertex_depth <= 0.75,
            (
                f"Vertex depth is {100.0 * vertex_depth:.1f}% from head top "
                "(expected within central 10–75% of head span)."
            ),
        )
    )

    checks.append(
        _check(
            "vertex detection has non-zero confidence",
            float(meta.get("vertex_confidence", 0.0)) > 0.0,
            "Vertex confidence is zero — image cues may have failed.",
        )
    )

    checks.append(
        _check(
            "crown envelope is asymmetric (posterior > lateral)",
            float(meta.get("axis_ratio_posterior_over_lateral", 1.0)) >= 1.05,
            (
                f"Posterior/lateral ratio is "
                f"{meta.get('axis_ratio_posterior_over_lateral', 1.0):.2f} "
                "(expected >= 1.05 — crown extends further posteriorly)."
            ),
        )
    )

    return checks


def print_metrics(
    result,
    hair_mask: np.ndarray,
    head_mask: np.ndarray,
    *,
    raw_exposed: np.ndarray | None = None,
    cleaned_exposed: np.ndarray | None = None,
) -> None:
    """Print head scale and region pixel counts."""
    meta = result.metadata
    normative = np.asarray(result.normative_region_mask).astype(bool)
    head = np.asarray(head_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)

    head_pixels = int(np.count_nonzero(head))
    normative_pixels = int(np.count_nonzero(normative))

    print("Head scale")
    print(f"  Bbox width:          {meta['head_bbox_width']:.1f} px")
    print(f"  Bbox height:         {meta['head_bbox_height']:.1f} px")
    print(f"  Head pixels:         {head_pixels:,}")
    print(f"  Equivalent radius:   {meta['equivalent_radius']:.1f} px")
    print(f"  Vertex (detected):   ({meta['vertex_x']:.1f}, {meta['vertex_y']:.1f})")
    print(f"  Vertex method:       {meta.get('vertex_method', 'unknown')}")
    print(f"  Vertex confidence:   {meta.get('vertex_confidence', 0.0):.2f}")
    print()
    print("Crown envelope parameters")
    print(f"  Head major span:     {meta['head_major_span']:.1f} px")
    print(f"  Head minor span:     {meta['head_minor_span']:.1f} px")
    print(f"  Head angle:          {meta['head_angle_deg']:.1f}°")
    print(f"  Extent posterior:    {meta['extent_posterior']:.1f} px")
    print(f"  Extent anterior:     {meta['extent_anterior']:.1f} px")
    print(f"  Extent lateral:      {meta['extent_lateral']:.1f} px")
    print(
        "  Post/lateral ratio:  "
        f"{meta['axis_ratio_posterior_over_lateral']:.2f}"
    )
    print(f"  Shape exponent:      {meta['shape_exponent']:.2f}")
    print(f"  Boundary margin:     {meta['boundary_margin']:.3f}")
    print()
    print("Region pixel counts")
    print(f"  Crown normative:     {normative_pixels:,}")
    print(
        f"  Fraction of head:    {100.0 * meta['crown_fraction_of_head']:.1f}%"
    )
    if raw_exposed is not None:
        deficit_pixels = int(np.count_nonzero(raw_exposed))
        cleaned_deficit_pixels = int(np.count_nonzero(cleaned_exposed))
    else:
        deficit_pixels = int(np.count_nonzero(normative & ~hair))
        cleaned_deficit_pixels = int(
            np.count_nonzero(_clean_exposed_scalp_mask(normative & ~hair, head))
        )
    print(
        "  Hair inside normative (green): "
        f"{int(np.count_nonzero(hair & normative)):,}"
    )
    print(f"  Exposed scalp (raw):             {deficit_pixels:,}")
    print(f"  Exposed scalp (cleaned):         {cleaned_deficit_pixels:,}")


def process_image(image_path: Path) -> int:
    """Run crown normative validation for one image. Returns 0 if all checks pass."""
    print(f"Image: {image_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    image = Image.open(image_path).convert("RGB")

    print("Running ViewValidator...")
    with ViewValidator() as view_validator:
        view_result = view_validator.validate(image, "crown")
    print_view_validation_result(view_result)
    print()
    if not view_result.is_valid:
        print("Crown analysis stopped — incompatible image view.")
        return 2

    print("Running ImageQualityChecker...")
    with ImageQualityChecker() as quality_checker:
        quality = quality_checker.validate(image, "crown")

    if quality["warnings"]:
        print("Quality warnings:")
        for warning in quality["warnings"]:
            print(f"  - {warning}")
        print()
    if not quality["valid"]:
        print(
            "Warning: quality checker marked image invalid; "
            "continuing for visual QA."
        )
        print()

    print("Running HairSegmenter...")
    segmenter = HairSegmenter()
    seg = segmenter.segment(image)
    head_mask = seg["head_mask"]
    hair_mask = seg["hair_mask"]
    print("✓ Segmentation complete")
    print()

    print("Running CrownNormativeRegionEstimator...")
    estimator = CrownNormativeRegionEstimator()
    result = estimator.estimate(image, head_mask, hair_mask)
    print("✓ Crown normative region estimated")
    print()

    head = np.asarray(head_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)
    normative = np.asarray(result.normative_region_mask).astype(bool)
    image_rgb = np.asarray(image.convert("RGB"))
    hair_probability = seg.get("hair_probability")
    skin_mask = seg.get("skin_mask")

    raw_deficit = _detect_exposed_scalp(
        normative,
        hair,
        head,
        image_rgb=image_rgb,
        hair_probability=hair_probability,
        skin_mask=skin_mask,
    )
    cleaned_deficit = _clean_exposed_scalp_mask(raw_deficit, head)

    print_metrics(
        result,
        hair_mask,
        head_mask,
        raw_exposed=raw_deficit,
        cleaned_exposed=cleaned_deficit,
    )
    print()

    head_pixels = int(np.count_nonzero(head))
    min_deficit_area = max(1, int(MIN_EXPOSED_SCALP_AREA_RATIO * head_pixels))
    raw_component_count, raw_components = _deficit_component_info(raw_deficit)
    cleaned_component_count, _ = _deficit_component_info(cleaned_deficit)

    stats = _mask_stats(head, hair, normative, raw_deficit, cleaned_deficit)

    hair_prob_mean = None
    if hair_probability is not None and np.any(normative):
        hair_prob_mean = float(np.mean(hair_probability[normative]))

    print_pipeline_diagnostics(
        stats,
        min_deficit_area=min_deficit_area,
        raw_component_count=raw_component_count,
        raw_components=raw_components,
        cleaned_component_count=cleaned_component_count,
        hair_probability_mean_in_normative=hair_prob_mean,
    )
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image.save(OUTPUT_DIR / "normal.png")
    _save_bool_mask(OUTPUT_DIR / "head_mask.png", head)
    _save_bool_mask(OUTPUT_DIR / "hair_mask.png", hair)
    _save_bool_mask(OUTPUT_DIR / "crown_normative_mask.png", normative)
    _save_bool_mask(OUTPUT_DIR / "raw_deficit_mask.png", raw_deficit)
    _save_bool_mask(OUTPUT_DIR / "cleaned_deficit_mask.png", cleaned_deficit)

    annotated = create_annotated_overlay(
        image,
        result.normative_region_mask,
        hair_mask,
        head_mask,
        hair_probability=hair_probability,
        skin_mask=skin_mask,
    )
    annotated.save(OUTPUT_DIR / "annotated.png")
    print("✓ Saved pipeline debug masks:")
    print("    normal.png")
    print("    head_mask.png")
    print("    hair_mask.png")
    print("    crown_normative_mask.png")
    print("    raw_deficit_mask.png")
    print("    cleaned_deficit_mask.png")
    print("    annotated.png")
    print()

    print("Validations")
    checks = run_validations(result, head_mask)
    print()

    all_passed = all(passed for _, passed, _ in checks)
    if all_passed:
        print("Overall: PASS — all validations succeeded.")
        return 0

    failed = [name for name, passed, _ in checks if not passed]
    print(f"Overall: FAIL — {len(failed)} validation(s) failed: {', '.join(failed)}")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual visual QA for CrownNormativeRegionEstimator.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help=f"Crown test image (default: {DEFAULT_IMAGE_STEM} under test_images/crown/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        image_path = args.image if args.image is not None else resolve_default_image()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension {image_path.suffix!r}; "
                f"expected one of {sorted(IMAGE_EXTENSIONS)}"
            )

        exit_code = process_image(image_path.resolve())
        sys.exit(exit_code)
    except Exception as exc:  # noqa: BLE001 — top-level manual script
        print(f"Error: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
