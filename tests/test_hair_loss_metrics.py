"""Unit tests for HairLossMetricsExtractor."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.deficit.types import (
    DeficitComponent,
    DeficitStatistics,
    HairDeficitResult,
)
from analysis.metrics.extractor import HairLossMetricsExtractor
from analysis.metrics.types import AnalysisMode
from analysis.normative_region.types import RegionMasks


def _empty_regions(shape: tuple[int, int]) -> RegionMasks:
    empty = np.zeros(shape, dtype=bool)
    return RegionMasks(
        frontal=empty.copy(),
        left_temple=empty.copy(),
        right_temple=empty.copy(),
        crown=empty.copy(),
    )


def _front_regions(shape: tuple[int, int]) -> RegionMasks:
    height, width = shape
    frontal = np.zeros(shape, dtype=bool)
    frontal[:, width // 3 : 2 * width // 3] = True
    left = np.zeros(shape, dtype=bool)
    left[:, : width // 3] = True
    right = np.zeros(shape, dtype=bool)
    right[:, 2 * width // 3 :] = True
    return RegionMasks(
        frontal=frontal,
        left_temple=left,
        right_temple=right,
        crown=np.zeros(shape, dtype=bool),
    )


def _crown_regions(shape: tuple[int, int]) -> RegionMasks:
    crown = np.zeros(shape, dtype=bool)
    crown[20:80, 20:80] = True
    empty = np.zeros(shape, dtype=bool)
    return RegionMasks(
        frontal=empty.copy(),
        left_temple=empty.copy(),
        right_temple=empty.copy(),
        crown=crown,
    )


def _combined_regions(shape: tuple[int, int]) -> RegionMasks:
    front = _front_regions(shape)
    crown = _crown_regions(shape)
    return RegionMasks(
        frontal=front.frontal,
        left_temple=front.left_temple,
        right_temple=front.right_temple,
        crown=crown.crown,
    )


def _make_result(
    deficit_mask: np.ndarray,
    *,
    components: tuple[DeficitComponent, ...] = (),
    normative_area_px: int | None = None,
) -> HairDeficitResult:
    deficit = np.asarray(deficit_mask).astype(bool)
    total_deficit = int(np.count_nonzero(deficit))
    normative_area = (
        normative_area_px
        if normative_area_px is not None
        else max(total_deficit, 1)
    )
    largest = max(components, key=lambda item: item.area_px) if components else None
    stats = DeficitStatistics(
        total_deficit_area_px=total_deficit,
        deficit_percentage=float(total_deficit / normative_area) if normative_area else 0.0,
        normative_area_px=normative_area,
        component_count=len(components),
        largest_component_area_px=largest.area_px if largest else 0,
        raw_deficit_area_px=total_deficit,
    )
    return HairDeficitResult(
        deficit_mask=deficit,
        components=components,
        largest_component=largest,
        statistics=stats,
    )


def _component(
    label_id: int,
    zone: str,
    area_px: int,
    zone_area_ratio: float,
) -> DeficitComponent:
    return DeficitComponent(
        label_id=label_id,
        zone=zone,  # type: ignore[arg-type]
        area_px=area_px,
        zone_area_ratio=zone_area_ratio,
        zone_overlap_ratio=1.0,
        centroid=(0.0, 0.0),
        bounding_box=(0, 0, 1, 1),
        contour=np.empty((0, 1, 2), dtype=np.int32),
    )


class HairLossMetricsExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = HairLossMetricsExtractor()
        self.shape = (100, 100)

    def test_no_deficit(self) -> None:
        regions = _front_regions(self.shape)
        normative_area = sum(
            int(np.count_nonzero(getattr(regions, name)))
            for name in ("frontal", "left_temple", "right_temple")
        )
        result = _make_result(
            np.zeros(self.shape, dtype=bool),
            normative_area_px=normative_area,
        )

        metrics = self.extractor.extract(result, regions)

        self.assertEqual(metrics.front_loss_percentage, 0.0)
        self.assertEqual(metrics.left_temple_loss_percentage, 0.0)
        self.assertEqual(metrics.right_temple_loss_percentage, 0.0)
        self.assertIsNone(metrics.crown_loss_percentage)
        self.assertEqual(metrics.overall_hair_loss_percentage, 0.0)
        self.assertEqual(metrics.overall_hair_coverage_percentage, 100.0)
        self.assertEqual(metrics.component_count, 0)
        self.assertIsNone(metrics.largest_deficit_zone)
        self.assertIsNone(metrics.largest_deficit_percentage)
        self.assertEqual(metrics.analysis_mode, AnalysisMode.FRONT_ONLY)
        self.assertTrue(metrics.front_available)
        self.assertFalse(metrics.crown_available)

    def test_frontal_only_deficit(self) -> None:
        regions = _front_regions(self.shape)
        deficit = np.zeros(self.shape, dtype=bool)
        deficit[10:30, 40:60] = True
        frontal_normative = int(np.count_nonzero(regions.frontal))
        deficit_pixels = int(np.count_nonzero(deficit & regions.frontal))
        normative_area = (
            int(np.count_nonzero(regions.frontal))
            + int(np.count_nonzero(regions.left_temple))
            + int(np.count_nonzero(regions.right_temple))
        )
        expected_front = 100.0 * deficit_pixels / frontal_normative

        metrics = self.extractor.extract(
            _make_result(deficit, normative_area_px=normative_area),
            regions,
        )

        self.assertAlmostEqual(metrics.front_loss_percentage, expected_front)
        self.assertEqual(metrics.left_temple_loss_percentage, 0.0)
        self.assertEqual(metrics.right_temple_loss_percentage, 0.0)
        self.assertIsNone(metrics.crown_loss_percentage)
        self.assertGreater(metrics.overall_hair_loss_percentage or 0.0, 0.0)
        self.assertEqual(metrics.analysis_mode, AnalysisMode.FRONT_ONLY)
        self.assertTrue(metrics.front_available)
        self.assertFalse(metrics.crown_available)

    def test_crown_only_deficit(self) -> None:
        regions = _crown_regions(self.shape)
        deficit = np.zeros(self.shape, dtype=bool)
        deficit[30:50, 30:50] = True
        crown_normative = int(np.count_nonzero(regions.crown))
        deficit_pixels = int(np.count_nonzero(deficit))
        expected_crown = 100.0 * deficit_pixels / crown_normative

        metrics = self.extractor.extract(
            _make_result(
                deficit,
                normative_area_px=crown_normative,
                components=(_component(1, "crown", deficit_pixels, expected_crown / 100.0),),
            ),
            regions,
        )

        self.assertIsNone(metrics.front_loss_percentage)
        self.assertIsNone(metrics.left_temple_loss_percentage)
        self.assertIsNone(metrics.right_temple_loss_percentage)
        self.assertAlmostEqual(metrics.crown_loss_percentage, expected_crown)
        self.assertEqual(metrics.largest_deficit_zone, "crown")
        self.assertAlmostEqual(metrics.largest_deficit_percentage, expected_crown)
        self.assertEqual(metrics.analysis_mode, AnalysisMode.CROWN_ONLY)
        self.assertFalse(metrics.front_available)
        self.assertTrue(metrics.crown_available)

    def test_mixed_deficits(self) -> None:
        regions = _front_regions(self.shape)
        deficit = np.zeros(self.shape, dtype=bool)
        deficit[10:20, 10:20] = True
        deficit[10:20, 70:80] = True

        left_deficit = int(np.count_nonzero(deficit & regions.left_temple))
        right_deficit = int(np.count_nonzero(deficit & regions.right_temple))
        left_norm = int(np.count_nonzero(regions.left_temple))
        right_norm = int(np.count_nonzero(regions.right_temple))
        normative_area = left_norm + right_norm + int(np.count_nonzero(regions.frontal))

        metrics = self.extractor.extract(
            _make_result(deficit, normative_area_px=normative_area),
            regions,
        )

        self.assertAlmostEqual(
            metrics.left_temple_loss_percentage,
            100.0 * left_deficit / left_norm,
        )
        self.assertAlmostEqual(
            metrics.right_temple_loss_percentage,
            100.0 * right_deficit / right_norm,
        )
        self.assertEqual(metrics.front_loss_percentage, 0.0)

    def test_missing_regions_return_none(self) -> None:
        regions = _crown_regions(self.shape)
        metrics = self.extractor.extract(
            _make_result(np.zeros(self.shape, dtype=bool), normative_area_px=3600),
            regions,
        )

        self.assertIsNone(metrics.front_loss_percentage)
        self.assertIsNone(metrics.left_temple_loss_percentage)
        self.assertIsNone(metrics.right_temple_loss_percentage)
        self.assertIsNotNone(metrics.crown_loss_percentage)
        self.assertEqual(metrics.analysis_mode, AnalysisMode.CROWN_ONLY)
        self.assertFalse(metrics.front_available)
        self.assertTrue(metrics.crown_available)

    def test_combined_analysis_mode(self) -> None:
        regions = _combined_regions(self.shape)
        normative_area = (
            int(np.count_nonzero(regions.frontal))
            + int(np.count_nonzero(regions.left_temple))
            + int(np.count_nonzero(regions.right_temple))
            + int(np.count_nonzero(regions.crown))
        )
        metrics = self.extractor.extract(
            _make_result(np.zeros(self.shape, dtype=bool), normative_area_px=normative_area),
            regions,
        )

        self.assertEqual(metrics.analysis_mode, AnalysisMode.COMBINED)
        self.assertTrue(metrics.front_available)
        self.assertTrue(metrics.crown_available)
        self.assertIsNotNone(metrics.front_loss_percentage)
        self.assertIsNotNone(metrics.crown_loss_percentage)

    def test_percentages_stay_within_bounds(self) -> None:
        regions = _front_regions(self.shape)
        deficit = regions.frontal.copy()
        normative_area = int(
            np.count_nonzero(regions.frontal | regions.left_temple | regions.right_temple)
        )

        metrics = self.extractor.extract(
            _make_result(deficit, normative_area_px=normative_area),
            regions,
        )

        for value in (
            metrics.front_loss_percentage,
            metrics.left_temple_loss_percentage,
            metrics.right_temple_loss_percentage,
            metrics.overall_hair_loss_percentage,
            metrics.overall_hair_coverage_percentage,
        ):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)

        self.assertEqual(metrics.front_loss_percentage, 100.0)
        self.assertEqual(metrics.left_temple_loss_percentage, 0.0)
        self.assertEqual(metrics.right_temple_loss_percentage, 0.0)
        expected_overall = 100.0 * int(np.count_nonzero(deficit)) / normative_area
        self.assertAlmostEqual(metrics.overall_hair_loss_percentage, expected_overall)
        self.assertAlmostEqual(
            metrics.overall_hair_coverage_percentage,
            100.0 - expected_overall,
        )


if __name__ == "__main__":
    unittest.main()
