"""Unit tests for HairCoverageRefiner (visibility-primary)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from analysis.coverage.refiner import HairCoverageRefiner


def test_dense_hair_unchanged() -> None:
    """Textured dense dark hair should not be demoted."""
    h, w = 64, 64
    rng = np.random.default_rng(1)
    rgb = (30 + rng.integers(0, 25, size=(h, w, 3))).astype(np.uint8)
    hair = np.ones((h, w), dtype=bool)
    head = np.ones((h, w), dtype=bool)
    prob = np.full((h, w), 0.95, dtype=np.float32)

    result = HairCoverageRefiner().refine(
        Image.fromarray(rgb),
        hair,
        head,
        hair_probability=prob,
        gate_mask=head,
    )
    assert int(np.count_nonzero(result.hair_coverage_mask)) >= int(0.90 * h * w)


def test_bright_exposed_scalp_demoted() -> None:
    """Bright exposed scalp (with residual strand speckles) should be demoted."""
    h, w = 96, 96
    rng = np.random.default_rng(0)
    rgb = np.full((h, w, 3), 40, dtype=np.uint8)
    rgb[30:66, 30:66] = 200
    speckles = np.zeros((h, w), dtype=bool)
    speckles[30:66, 30:66] = rng.random((36, 36)) < 0.15
    rgb[speckles] = 45
    hair = np.ones((h, w), dtype=bool)
    head = np.ones((h, w), dtype=bool)
    prob = np.full((h, w), 0.92, dtype=np.float32)

    result = HairCoverageRefiner().refine(
        Image.fromarray(rgb),
        hair,
        head,
        hair_probability=prob,
        gate_mask=head,
    )
    patch = result.hair_coverage_mask[30:66, 30:66]
    assert int(np.count_nonzero(~patch)) > 100
    assert result.pixels_demoted > 0


def test_dark_covered_hair_not_demoted_beside_scalp() -> None:
    """Dark well-covered hair next to a bright scalp patch must stay hair."""
    h, w = 128, 128
    rng = np.random.default_rng(2)
    rgb = np.full((h, w, 3), 35, dtype=np.uint8)
    # Bright exposed scalp disk on the left (with residual strands).
    yy, xx = np.ogrid[:h, :w]
    scalp = (yy - 64) ** 2 + (xx - 40) ** 2 <= 18**2
    rgb[scalp] = 200
    speckles = scalp & (rng.random((h, w)) < 0.18)
    rgb[speckles] = 45
    # Right side stays dark dense hair.
    right = xx > 80
    hair = np.ones((h, w), dtype=bool)
    head = np.ones((h, w), dtype=bool)
    prob = np.full((h, w), 0.95, dtype=np.float32)

    result = HairCoverageRefiner().refine(
        Image.fromarray(rgb),
        hair,
        head,
        hair_probability=prob,
        gate_mask=head,
    )
    demoted = hair & ~result.hair_coverage_mask
    assert int(np.count_nonzero(demoted & scalp)) > 150
    # Most of the dark right side must remain covered (not deficit).
    assert int(np.count_nonzero(demoted & right)) < 80
