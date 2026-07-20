"""
Manual visual QA for FrontNormativeRegionEstimator.

Not a unit test — run manually to inspect normative region geometry:

    python tests/test_normative_front.py
    python tests/test_normative_front.py path/to/front_image.jpg
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running as ``python tests/test_normative_front.py`` from HairVision-AI/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.normative_region.front_estimator import FrontNormativeRegionEstimator
from analysis.normative_region.front_hairline import NormativeHairline
from analysis.quality import ImageQualityChecker
from analysis.validation.view_validator import ViewValidator, print_view_validation_result
from models.segmentation import HairSegmenter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_STEM = "mi_front"
DEFAULT_IMAGE_DIR = ROOT / "test_images" / "front"
OUTPUT_DIR = ROOT / "outputs" / "normative_front"
DEBUG_DIR = OUTPUT_DIR / "debug"

HAIR_OVERLAY_COLOR = (40, 180, 60)
HAIR_OVERLAY_ALPHA = 0.45
DEFICIT_OVERLAY_COLOR = (255, 140, 0)
DEFICIT_OVERLAY_ALPHA = 0.45
HAIRLINE_COLOR = (255, 0, 0)
HAIRLINE_WIDTH = 2
LANDMARK_COLOR = (0, 200, 255)
ANCHOR_COLOR = (255, 255, 0)
CONTROL_COLOR = (255, 0, 255)
FOREHEAD_COLOR = (180, 80, 255)
FOREHEAD_ALPHA = 0.35
NORMATIVE_COLOR = (40, 120, 255)
NORMATIVE_ALPHA = 0.35


def resolve_default_image() -> Path:
    """Return the default front test image path, trying common extensions."""
    for ext in (".jpeg", ".jpg", ".png"):
        candidate = DEFAULT_IMAGE_DIR / f"{DEFAULT_IMAGE_STEM}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No default front image found for stem '{DEFAULT_IMAGE_STEM}' "
        f"under {DEFAULT_IMAGE_DIR}"
    )



def _blend_mask(
    base: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Alpha-blend ``color`` onto ``base`` where ``mask`` is True."""
    out = base.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    selected = np.asarray(mask).astype(bool)
    if np.any(selected):
        out[selected] = (1.0 - alpha) * out[selected] + alpha * color_arr
    return out


def _draw_polyline(
    canvas: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    """Draw a polyline on an RGB canvas (in-place)."""
    if len(points) < 2:
        return
    try:
        import cv2

        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in points],
            dtype=np.int32,
        )
        cv2.polylines(
            bgr,
            [pts],
            isClosed=False,
            color=(color[2], color[1], color[0]),
            thickness=width,
            lineType=cv2.LINE_AA,
        )
        canvas[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return
    except ImportError:
        pass

    from PIL import ImageDraw

    pil = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    draw.line([(x, y) for x, y in points], fill=color, width=width)
    canvas[:] = np.asarray(pil)


def _draw_points(
    canvas: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    radius: int = 5,
) -> None:
    """Draw filled circles on an RGB canvas (in-place)."""
    try:
        import cv2

        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        for x, y in points:
            cv2.circle(
                bgr,
                (int(round(x)), int(round(y))),
                radius,
                (color[2], color[1], color[0]),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        canvas[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return
    except ImportError:
        pass

    from PIL import ImageDraw

    pil = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    for x, y in points:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
        )
    canvas[:] = np.asarray(pil)


def _draw_hairline_curve(
    canvas: np.ndarray,
    hairline: NormativeHairline,
) -> None:
    """Draw the estimator normative hairline in red (in-place on RGB canvas)."""
    _draw_polyline(canvas, hairline.sample_curve(), HAIRLINE_COLOR, HAIRLINE_WIDTH)


def create_normative_overlay(
    image: Image.Image,
    normative_mask: np.ndarray,
    hair_mask: np.ndarray,
    hairline_anchors: dict,
) -> Image.Image:
    """
    Overlay hair (green) and deficit (orange) using the estimator normative mask.

    Red curve = lower boundary from ``FrontNormativeRegionEstimator`` hairline.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    normative = np.asarray(normative_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)

    hair_inside_normative = hair & normative
    deficit_mask = normative & ~hair

    blended = _blend_mask(
        base,
        hair_inside_normative,
        HAIR_OVERLAY_COLOR,
        HAIR_OVERLAY_ALPHA,
    )
    blended = _blend_mask(
        blended,
        deficit_mask,
        DEFICIT_OVERLAY_COLOR,
        DEFICIT_OVERLAY_ALPHA,
    )

    rgb = np.clip(blended, 0, 255).astype(np.uint8)
    _draw_hairline_curve(rgb, NormativeHairline.from_metadata(hairline_anchors))
    return Image.fromarray(rgb)


def _save_debug_stages(
    image: Image.Image,
    result,
    head_mask: np.ndarray | None = None,
) -> None:
    """Write separate debug stages for geometric QA."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    meta = result.metadata
    anchors = meta["hairline_anchors"]
    hairline = NormativeHairline.from_metadata(anchors)
    debug_lm = meta.get("debug_landmarks", {})
    controls = [tuple(p) for p in meta.get("hairline_control_points", hairline.control_points)]

    # 1. Facial landmarks + centerline
    stage = base.copy()
    rim = [tuple(p) for p in debug_lm.get("forehead_rim", [])]
    brows = [tuple(p) for p in debug_lm.get("brow_points", [])]
    _draw_points(stage, rim, LANDMARK_COLOR, radius=4)
    _draw_points(stage, brows, (0, 255, 128), radius=3)
    if "forehead" in debug_lm:
        _draw_points(stage, [tuple(debug_lm["forehead"])], (255, 255, 255), radius=6)
    centerline = [tuple(p) for p in debug_lm.get("facial_centerline", [])]
    if len(centerline) >= 2:
        _draw_polyline(stage, centerline, (0, 255, 255), width=1)
    Image.fromarray(stage).save(DEBUG_DIR / "01_landmarks.png")

    # 2. Head contour
    stage = base.copy()
    contour = [tuple(p) for p in debug_lm.get("head_contour", [])]
    if contour:
        _draw_polyline(stage, contour, (0, 200, 255), width=1)
    elif head_mask is not None:
        # Fallback outline from mask columns.
        head = np.asarray(head_mask).astype(bool)
        tops = []
        for x in range(0, head.shape[1], 3):
            ys = np.where(head[:, x])[0]
            if ys.size:
                tops.append((float(x), float(ys.min())))
        _draw_polyline(stage, tops, (0, 200, 255), width=1)
    Image.fromarray(stage).save(DEBUG_DIR / "02_head_contour.png")

    # 3. Control points (mirrored construction)
    stage = base.copy()
    if len(centerline) >= 2:
        _draw_polyline(stage, centerline, (0, 255, 255), width=1)
    _draw_points(stage, controls, CONTROL_COLOR, radius=6)
    _draw_polyline(stage, controls, CONTROL_COLOR, width=1)
    Image.fromarray(stage).save(DEBUG_DIR / "03_control_points.png")

    # 4. Generated smooth curve
    stage = base.copy()
    _draw_hairline_curve(stage, hairline)
    _draw_points(stage, controls, CONTROL_COLOR, radius=4)
    Image.fromarray(stage).save(DEBUG_DIR / "04_curve.png")

    # 5. Composite: centerline + controls + contour + hairline
    stage = base.copy()
    if contour:
        _draw_polyline(stage, contour, (0, 200, 255), width=1)
    if len(centerline) >= 2:
        _draw_polyline(stage, centerline, (0, 255, 255), width=1)
    _draw_points(stage, controls, CONTROL_COLOR, radius=5)
    _draw_hairline_curve(stage, hairline)
    Image.fromarray(stage).save(DEBUG_DIR / "05_composite_geometry.png")

    # 6. Final normative region
    stage = base.astype(np.float32)
    stage = _blend_mask(
        stage,
        result.normative_region_mask,
        NORMATIVE_COLOR,
        NORMATIVE_ALPHA,
    )
    stage_u8 = np.clip(stage, 0, 255).astype(np.uint8)
    _draw_hairline_curve(stage_u8, hairline)
    _draw_points(stage_u8, controls, CONTROL_COLOR, radius=4)
    Image.fromarray(stage_u8).save(DEBUG_DIR / "06_normative_region.png")

    print(f"✓ Saved debug stages under {DEBUG_DIR}")
    print("Geometry ratios")
    print(f"  Normalized forehead height: {anchors.get('normalized_forehead_height', float('nan')):.4f}")
    print(f"  Hairline height ratio:      {anchors.get('hairline_height_ratio', float('nan')):.4f}")
    print("  Control points:")
    for i, (x, y) in enumerate(controls):
        print(f"    [{i}] ({x:.1f}, {y:.1f})")


def _check(
    name: str,
    passed: bool,
    reason: str,
) -> tuple[str, bool, str]:
    """Format a single validation result."""
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed:
        print(f"         {reason}")
    return name, passed, reason


def run_validations(
    result,
    head_mask: np.ndarray,
) -> list[tuple[str, bool, str]]:
    """Run automatic checks and print PASS/FAIL for each."""
    normative = np.asarray(result.normative_region_mask).astype(bool)
    head = np.asarray(head_mask).astype(bool)
    masks = result.region_masks

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        _check(
            'view == "front"',
            result.view == "front",
            f"Expected view 'front', got {result.view!r}.",
        )
    )

    normative_pixels = int(np.count_nonzero(normative))
    checks.append(
        _check(
            "normative_region_mask is not empty",
            normative_pixels > 0,
            "Normative union mask has zero pixels — anchors or head clip may be wrong.",
        )
    )

    crown_pixels = int(np.count_nonzero(masks.crown))
    checks.append(
        _check(
            "crown mask is empty",
            crown_pixels == 0,
            f"Front view crown mask should be empty but has {crown_pixels} pixels.",
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
    checks.append(
        _check(
            "frontal region is non-empty",
            frontal_pixels > 0,
            "Frontal band has zero pixels — hairline curve or head top may be misaligned.",
        )
    )

    left_pixels = int(np.count_nonzero(masks.left_temple))
    checks.append(
        _check(
            "left temple region is non-empty",
            left_pixels > 0,
            "Left temple box is empty — temple landmarks or ROI scaling may be wrong.",
        )
    )

    right_pixels = int(np.count_nonzero(masks.right_temple))
    checks.append(
        _check(
            "right temple region is non-empty",
            right_pixels > 0,
            "Right temple zone is empty — temporal boundary or head clip may be wrong.",
        )
    )

    anchors = result.metadata.get("hairline_anchors", {})
    if anchors:
        hairline = NormativeHairline.from_metadata(anchors)
        above = hairline.above_mask(normative.shape[0], normative.shape[1])
        below = int(np.count_nonzero(normative & ~above))
        checks.append(
            _check(
                "normative region excludes forehead below hairline",
                below == 0,
                f"{below} normative pixels fall below the hairline on the forehead.",
            )
        )
        checks.append(
            _check(
                "hairline sits at or above forehead landmark",
                hairline.center_y <= hairline.forehead_y + 1.0,
                "Midline hairline falls below the forehead landmark onto forehead skin.",
            )
        )
        # Smooth arch: measurable curvature in the central half (no flat plateau).
        half = 0.5 * (hairline.right_x - hairline.left_x)
        xs_mid = np.linspace(
            hairline.center_x - 0.35 * half,
            hairline.center_x + 0.35 * half,
            21,
        )
        ys_mid = [hairline.y_at_x(float(x)) for x in xs_mid]
        mid_span = float(max(ys_mid) - min(ys_mid))
        forehead_zone = max(hairline.brow_y - hairline.forehead_y, 1.0)
        checks.append(
            _check(
                "central band has continuous curvature",
                mid_span >= 0.015 * forehead_zone,
                f"Central arch varies only {mid_span:.2f}px — still too flat.",
            )
        )
        checks.append(
            _check(
                "hairline is mirror-symmetric",
                abs(hairline.left_y - hairline.right_y) < 0.5
                and abs(
                    (hairline.center_x - hairline.left_x)
                    - (hairline.right_x - hairline.center_x)
                )
                < 0.5,
                "Left/right temple height or half-width mismatch.",
            )
        )
        checks.append(
            _check(
                "temples sit below frontal apex",
                hairline.left_y > hairline.center_y
                and hairline.right_y > hairline.center_y,
                "Temple endpoints must be caudal to the frontal apex.",
            )
        )
        checks.append(
            _check(
                "hairline does not invade mid-forehead",
                (hairline.center_y - hairline.forehead_y) / forehead_zone <= 0.30,
                "Center hairline penetrates too far into the brow–forehead zone.",
            )
        )

    return checks


def print_metrics(
    result,
    hair_mask: np.ndarray,
) -> None:
    """Print face scale, anchors, and region pixel counts."""
    meta = result.metadata
    anchors = meta["hairline_anchors"]
    masks = result.region_masks
    normative = np.asarray(result.normative_region_mask).astype(bool)
    hair = np.asarray(hair_mask).astype(bool)

    print("Face scale")
    print(f"  Face width:  {meta['face_width']:.1f} px")
    print(f"  Face height: {meta['face_height']:.1f} px")
    print()
    print("Hairline anchors (image coordinates; smaller y is higher)")
    print(f"  Left:   ({anchors['left_x']:.1f}, {anchors['left_y']:.1f})")
    print(f"  Center: ({anchors['center_x']:.1f}, {anchors['center_y']:.1f})")
    print(f"  Right:  ({anchors['right_x']:.1f}, {anchors['right_y']:.1f})")
    print(f"  Forehead reference y: {anchors['forehead_y']:.1f}")
    print(f"  Brow reference y:     {anchors['brow_y']:.1f}")
    print(f"  Ideal trichion y:     {anchors.get('ideal_trichion_y', float('nan')):.1f}")
    print(f"  Applied rim offset:   {anchors.get('applied_offset', float('nan')):.1f}")
    print(f"  Norm. forehead h:     {anchors.get('normalized_forehead_height', float('nan')):.4f}")
    print(f"  Hairline height ratio:{anchors.get('hairline_height_ratio', float('nan')):.4f}")
    print(f"  Trichion–rim blend:   {anchors.get('trichion_rim_blend', float('nan')):.2f}")
    print(f"  Temple depth frac:    {anchors.get('temple_depth_fraction', float('nan')):.2f}")
    print(f"  Profile model:        {anchors.get('profile_model', 'n/a')}")
    print(f"  Forehead corrected:   {anchors.get('forehead_corrected', False)}")
    print()
    print("Region pixel counts")
    frontal_pixels = int(np.count_nonzero(masks.frontal))
    left_pixels = int(np.count_nonzero(masks.left_temple))
    right_pixels = int(np.count_nonzero(masks.right_temple))
    normative_pixels = int(np.count_nonzero(normative))
    hair_inside = int(np.count_nonzero(hair & normative))
    deficit_pixels = int(np.count_nonzero(normative & ~hair))
    print(f"  Frontal:                     {frontal_pixels:,}")
    print(f"  Left temple:                 {left_pixels:,}")
    print(f"  Right temple:                {right_pixels:,}")
    print(f"  Total normative:             {normative_pixels:,}")
    print(f"  Hair inside normative:       {hair_inside:,}")
    print(f"  Hair deficit:                {deficit_pixels:,}")
    print(
        "  Check (hair + deficit):      "
        f"{hair_inside + deficit_pixels:,} (should equal total normative)"
    )


def process_image(image_path: Path) -> int:
    """
    Run the full normative-front validation workflow for one image.

    Returns 0 on success (all validations passed), 1 otherwise.
    """
    print(f"Image: {image_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    image = Image.open(image_path).convert("RGB")

    print("Running ViewValidator...")
    with ViewValidator() as view_validator:
        view_result = view_validator.validate(image, "front")
    print_view_validation_result(view_result)
    print()
    if not view_result.is_valid:
        print("Front analysis stopped — incompatible image view.")
        return 2

    print("Running ImageQualityChecker...")
    with ImageQualityChecker() as quality_checker:
        quality = quality_checker.validate(image, "front")
        landmarks = quality_checker.detect_landmarks(image)

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
    segmentation_mask = seg["segmentation_mask"]
    print("✓ Segmentation complete")
    print()

    print("Running FrontNormativeRegionEstimator...")
    estimator = FrontNormativeRegionEstimator()
    result = estimator.estimate(
        image,
        landmarks,
        head_mask,
        segmentation_mask=segmentation_mask,
    )
    print("✓ Normative region estimated")
    print()

    print_metrics(result, hair_mask)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image.save(OUTPUT_DIR / "original.png")

    overlay = create_normative_overlay(
        image,
        result.normative_region_mask,
        hair_mask,
        result.metadata["hairline_anchors"],
    )
    overlay.save(OUTPUT_DIR / "overlay_normative.png")
    print("✓ Saved original.png and overlay_normative.png")
    _save_debug_stages(image, result, head_mask=head_mask)
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
        description="Manual visual QA for FrontNormativeRegionEstimator.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help=f"Front test image (default: {DEFAULT_IMAGE_STEM} under test_images/front/)",
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
