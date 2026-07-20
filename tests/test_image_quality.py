"""Unit tests for soft/hard image quality validation and upscaling."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from analysis.quality import (
    HARD_MIN_HEIGHT,
    HARD_MIN_WIDTH,
    RECOMMENDED_MIN_HEIGHT,
    RECOMMENDED_MIN_WIDTH,
    ImageQualityChecker,
    upscale_to_recommended,
)


class UpscaleTests(unittest.TestCase):
    def test_preserves_aspect_and_meets_recommended(self) -> None:
        image = Image.new("RGB", (401, 431), color=(100, 100, 100))
        out, upscaled = upscale_to_recommended(image)
        self.assertTrue(upscaled)
        self.assertGreaterEqual(out.size[0], RECOMMENDED_MIN_WIDTH)
        self.assertGreaterEqual(out.size[1], RECOMMENDED_MIN_HEIGHT)
        # Aspect ratio preserved within rounding.
        src_ratio = 401 / 431
        out_ratio = out.size[0] / out.size[1]
        self.assertAlmostEqual(src_ratio, out_ratio, places=2)

    def test_no_upscale_when_already_recommended(self) -> None:
        image = Image.new("RGB", (640, 640), color=(100, 100, 100))
        out, upscaled = upscale_to_recommended(image)
        self.assertFalse(upscaled)
        self.assertEqual(out.size, (640, 640))


class QualityValidateTests(unittest.TestCase):
    def test_sub_recommended_is_soft_warning_and_upscales(self) -> None:
        checker = ImageQualityChecker.__new__(ImageQualityChecker)
        checker._landmarker = None
        image = Image.new("RGB", (401, 431), color=(120, 120, 120))
        with patch.object(ImageQualityChecker, "_face_detected", return_value=True):
            result = ImageQualityChecker.validate(checker, image, "front")
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(any("lower than recommended" in w for w in result["warnings"]))
        self.assertTrue(result["upscaled"])
        prep = result["prepared_image"]
        self.assertGreaterEqual(prep.size[0], RECOMMENDED_MIN_WIDTH)
        self.assertGreaterEqual(prep.size[1], RECOMMENDED_MIN_HEIGHT)

    def test_extremely_small_is_hard_failure(self) -> None:
        checker = ImageQualityChecker.__new__(ImageQualityChecker)
        checker._landmarker = None
        image = Image.new(
            "RGB",
            (HARD_MIN_WIDTH - 1, HARD_MIN_HEIGHT - 1),
            color=(120, 120, 120),
        )
        result = ImageQualityChecker.validate(checker, image, "crown")
        self.assertFalse(result["valid"])
        self.assertTrue(any("resolution_unusable" in e for e in result["errors"]))
        self.assertFalse(result["upscaled"])

    def test_blur_is_soft_warning(self) -> None:
        checker = ImageQualityChecker.__new__(ImageQualityChecker)
        checker._landmarker = None
        # Near-uniform image → low Laplacian variance.
        image = Image.new("RGB", (640, 640), color=(128, 128, 128))
        result = ImageQualityChecker.validate(checker, image, "crown")
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(any("blurry" in w for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
