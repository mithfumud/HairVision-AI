"""Unit tests for HairDeficitAnalyzer using synthetic masks."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.deficit.analyzer import HairDeficitAnalyzer
from analysis.deficit.metrics import DEFAULT_MIN_COMPONENT_FRACTION
from analysis.normative_region.types import RegionMasks


def _empty_regions(shape: tuple[int, int]) -> RegionMasks:
    empty = np.zeros(shape, dtype=bool)
    return RegionMasks(
        frontal=empty.copy(),
        left_temple=empty.copy(),
        right_temple=empty.copy(),
        crown=empty.copy(),
    )


def _crown_regions(shape: tuple[int, int]) -> RegionMasks:
    height, width = shape
    crown = np.zeros(shape, dtype=bool)
    crown[20:80, 20:80] = True
    empty = np.zeros(shape, dtype=bool)
    return RegionMasks(
        frontal=empty.copy(),
        left_temple=empty.copy(),
        right_temple=empty.copy(),
        crown=crown,
    )


class HairDeficitAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = HairDeficitAnalyzer(
            min_component_fraction=DEFAULT_MIN_COMPONENT_FRACTION,
        )
        self.shape = (100, 100)

    def test_empty_deficit_when_normative_is_fully_haired(self) -> None:
        normative = np.zeros(self.shape, dtype=bool)
        normative[25:75, 25:75] = True
        hair = normative.copy()
        head = np.ones(self.shape, dtype=bool)

        result = self.analyzer.analyze(
            normative,
            hair,
            head,
            _crown_regions(self.shape),
        )

        self.assertFalse(np.any(result.deficit_mask))
        self.assertEqual(result.components, ())
        self.assertIsNone(result.largest_component)
        self.assertEqual(result.statistics.total_deficit_area_px, 0)
        self.assertEqual(result.statistics.component_count, 0)
        self.assertEqual(result.statistics.deficit_percentage, 0.0)

    def test_single_component(self) -> None:
        normative = np.zeros(self.shape, dtype=bool)
        normative[30:70, 30:70] = True
        hair = normative.copy()
        hair[45:55, 45:55] = False
        head = np.ones(self.shape, dtype=bool)

        result = self.analyzer.analyze(
            normative,
            hair,
            head,
            _crown_regions(self.shape),
        )

        self.assertEqual(result.statistics.component_count, 1)
        self.assertEqual(result.statistics.total_deficit_area_px, 100)
        self.assertIsNotNone(result.largest_component)
        assert result.largest_component is not None
        self.assertEqual(result.largest_component.area_px, 100)
        self.assertEqual(result.largest_component.zone, "crown")
        self.assertGreater(result.largest_component.zone_area_ratio, 0.0)
        self.assertAlmostEqual(result.largest_component.zone_overlap_ratio, 1.0)
        self.assertEqual(result.largest_component.contour.shape[-1], 2)

    def test_multiple_components(self) -> None:
        normative = np.zeros(self.shape, dtype=bool)
        normative[20:80, 20:80] = True
        hair = normative.copy()
        hair[30:40, 30:40] = False
        hair[60:70, 60:70] = False
        head = np.ones(self.shape, dtype=bool)

        result = self.analyzer.analyze(
            normative,
            hair,
            head,
            _crown_regions(self.shape),
        )

        self.assertEqual(result.statistics.component_count, 2)
        self.assertEqual(result.statistics.total_deficit_area_px, 200)
        self.assertIsNotNone(result.largest_component)
        assert result.largest_component is not None
        self.assertEqual(result.largest_component.area_px, 100)
        areas = sorted(component.area_px for component in result.components)
        self.assertEqual(areas, [100, 100])

    def test_tiny_noise_removal(self) -> None:
        normative = np.zeros(self.shape, dtype=bool)
        normative[20:80, 20:80] = True
        hair = normative.copy()
        hair[40:60, 40:60] = False
        hair[22:23, 22:23] = False
        head = np.ones(self.shape, dtype=bool)

        result = self.analyzer.analyze(
            normative,
            hair,
            head,
            _crown_regions(self.shape),
        )

        self.assertEqual(result.statistics.raw_deficit_area_px, 401)
        self.assertEqual(result.statistics.component_count, 1)
        self.assertEqual(result.statistics.total_deficit_area_px, 400)

    def test_largest_component_detection(self) -> None:
        normative = np.zeros(self.shape, dtype=bool)
        normative[10:90, 10:90] = True
        hair = normative.copy()
        hair[20:50, 20:50] = False
        hair[60:65, 60:65] = False
        head = np.ones(self.shape, dtype=bool)

        result = self.analyzer.analyze(
            normative,
            hair,
            head,
            _crown_regions(self.shape),
        )

        self.assertEqual(result.statistics.component_count, 2)
        self.assertIsNotNone(result.largest_component)
        assert result.largest_component is not None
        self.assertEqual(result.largest_component.area_px, 900)
        self.assertEqual(
            result.largest_component.label_id,
            result.components[0].label_id,
        )
        self.assertEqual(
            result.statistics.largest_component_area_px,
            result.largest_component.area_px,
        )

    def test_front_zone_assignment(self) -> None:
        shape = (100, 100)
        normative = np.zeros(shape, dtype=bool)
        normative[20:80, 20:80] = True
        hair = normative.copy()
        hair[30:40, 25:35] = False

        frontal = np.zeros(shape, dtype=bool)
        frontal[20:80, 35:65] = True
        left_temple = np.zeros(shape, dtype=bool)
        left_temple[20:80, 20:35] = True
        right_temple = np.zeros(shape, dtype=bool)
        right_temple[20:80, 65:80] = True
        regions = RegionMasks(
            frontal=frontal,
            left_temple=left_temple,
            right_temple=right_temple,
            crown=np.zeros(shape, dtype=bool),
        )

        result = self.analyzer.analyze(
            normative,
            hair,
            np.ones(shape, dtype=bool),
            regions,
        )

        self.assertEqual(result.statistics.component_count, 1)
        self.assertEqual(result.components[0].zone, "left_temple")
        self.assertAlmostEqual(result.components[0].zone_overlap_ratio, 1.0)

    def test_configurable_min_component_fraction(self) -> None:
        normative = np.zeros(self.shape, dtype=bool)
        normative[20:80, 20:80] = True
        hair = normative.copy()
        hair[40:60, 40:60] = False
        hair[22:27, 22:27] = False
        head = np.ones(self.shape, dtype=bool)

        default_result = self.analyzer.analyze(
            normative,
            hair,
            head,
            _crown_regions(self.shape),
        )
        strict_analyzer = HairDeficitAnalyzer(min_component_fraction=0.01)
        strict_result = strict_analyzer.analyze(
            normative,
            hair,
            head,
            _crown_regions(self.shape),
        )

        self.assertEqual(default_result.statistics.component_count, 2)
        self.assertEqual(default_result.statistics.total_deficit_area_px, 425)
        self.assertEqual(strict_result.statistics.component_count, 1)
        self.assertEqual(strict_result.statistics.total_deficit_area_px, 400)

    def test_zone_overlap_ratio_for_mixed_zone_component(self) -> None:
        shape = (100, 100)
        normative = np.zeros(shape, dtype=bool)
        normative[20:80, 20:80] = True
        hair = normative.copy()
        hair[30:40, 44:54] = False

        left_temple = np.zeros(shape, dtype=bool)
        left_temple[20:80, 20:50] = True
        frontal = np.zeros(shape, dtype=bool)
        frontal[20:80, 50:80] = True
        regions = RegionMasks(
            frontal=frontal,
            left_temple=left_temple,
            right_temple=np.zeros(shape, dtype=bool),
            crown=np.zeros(shape, dtype=bool),
        )

        result = self.analyzer.analyze(
            normative,
            hair,
            np.ones(shape, dtype=bool),
            regions,
        )

        component = result.components[0]
        self.assertEqual(component.area_px, 100)
        self.assertEqual(component.zone, "left_temple")
        self.assertAlmostEqual(component.zone_overlap_ratio, 0.6)

    def test_unknown_zone_when_no_region_overlap(self) -> None:
        normative = np.zeros(self.shape, dtype=bool)
        normative[40:60, 40:60] = True
        hair = normative.copy()
        hair[45:55, 45:55] = False

        result = self.analyzer.analyze(
            normative,
            hair,
            np.ones(self.shape, dtype=bool),
            _empty_regions(self.shape),
        )

        self.assertEqual(result.components[0].zone, "unknown")
        self.assertEqual(result.components[0].zone_area_ratio, 0.0)
        self.assertEqual(result.components[0].zone_overlap_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
