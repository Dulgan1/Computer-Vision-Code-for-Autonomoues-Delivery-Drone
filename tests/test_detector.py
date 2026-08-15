import unittest

import cv2
import numpy as np

import config
from detector import MarkerDetector


class FilterContoursTests(unittest.TestCase):
    def test_reuses_circle_masks_for_matching_radius_and_style(self):
        detector = MarkerDetector()

        first = detector._circle_mask(50, thickness=-1)
        second = detector._circle_mask(50, thickness=-1)
        outline = detector._circle_mask(50, thickness=2)

        self.assertIs(first, second)
        self.assertIsNot(first, outline)
        self.assertEqual(first.shape, (100, 100))

    def test_extract_roi_preserves_binary_edge_pixels(self):
        edges = np.zeros((20, 20), dtype=np.uint8)
        edges[10, 10] = 255
        candidate = {"center": (10, 10), "radius": 5}

        MarkerDetector().extract_roi(edges, candidate)

        self.assertEqual(candidate["roi"][5, 5], 255)
        self.assertTrue(np.all(np.isin(candidate["roi"], [0, 255])))

    def test_keeps_large_contour_and_enriches_its_geometry(self):
        small = np.array([[[0, 0]], [[3, 0]], [[3, 3]], [[0, 3]]], dtype=np.int32)
        large = np.array(
            [[[10, 20]], [[30, 20]], [[30, 40]], [[10, 40]]], dtype=np.int32
        )
        candidate = {"contours": [small, large]}

        result = MarkerDetector().filter_contours(candidate)

        self.assertIs(result, candidate)
        self.assertEqual(len(result["contours"]), 1)

        contour_object = result["contours"][0]
        self.assertIs(contour_object["contour"], large)
        self.assertEqual(contour_object["area"], cv2.contourArea(large))
        self.assertEqual(contour_object["perimeter"], cv2.arcLength(large, True))
        self.assertEqual(contour_object["bbox"], (10, 20, 21, 21))
        self.assertGreater(contour_object["area"], config.MIN_CONTOUR_AREA)


class FitLinesTests(unittest.TestCase):
    def test_fits_a_unit_direction_to_a_contour(self):
        contour = np.array(
            [[[0, 0]], [[10, 10]], [[20, 20]], [[30, 30]]], dtype=np.int32
        )
        candidate = {"contours": [{"contour": contour}]}

        result = MarkerDetector().fit_lines(candidate)

        self.assertIs(result, candidate)
        vx, vy, x0, y0 = result["contours"][0]["line"]
        self.assertAlmostEqual(vx * vx + vy * vy, 1.0, places=5)
        self.assertAlmostEqual(abs(vx), abs(vy), places=5)
        self.assertAlmostEqual(x0, y0, places=5)

    def test_marks_degenerate_contour_without_discarding_it(self):
        contour = np.array([[[5, 5]]], dtype=np.int32)
        candidate = {"contours": [{"contour": contour}]}

        MarkerDetector().fit_lines(candidate)

        self.assertIsNone(candidate["contours"][0]["line"])


class LineFeatureTests(unittest.TestCase):
    def test_computes_diagonal_orientation_length_and_midpoint(self):
        contour = np.array(
            [[[0, 0]], [[10, 10]], [[20, 20]]], dtype=np.int32
        )
        candidate = {"contours": [{"contour": contour}]}
        detector = MarkerDetector()

        detector.fit_lines(candidate)
        detector.compute_line_features(candidate)

        contour_object = candidate["contours"][0]
        self.assertAlmostEqual(contour_object["angle"], 45.0, places=4)
        self.assertAlmostEqual(contour_object["length"], 20 * np.sqrt(2), places=4)
        self.assertAlmostEqual(contour_object["midpoint"][0], 10.0, places=4)
        self.assertAlmostEqual(contour_object["midpoint"][1], 10.0, places=4)

    def test_normalizes_opposite_horizontal_directions_to_zero_degrees(self):
        contour = np.array(
            [[[20, 5]], [[10, 5]], [[0, 5]]], dtype=np.int32
        )
        candidate = {"contours": [{"contour": contour}]}
        detector = MarkerDetector()

        detector.fit_lines(candidate)
        detector.compute_line_features(candidate)

        self.assertAlmostEqual(candidate["contours"][0]["angle"], 0.0, places=4)


class OrientationClusteringTests(unittest.TestCase):
    def test_clusters_nearby_orientations_by_total_support_length(self):
        candidate = {
            "contours": [
                {"angle": 43.0, "length": 30.0},
                {"angle": 47.0, "length": 20.0},
                {"angle": 133.0, "length": 25.0},
                {"angle": 137.0, "length": 15.0},
            ]
        }

        MarkerDetector().cluster_orientations(candidate)

        clusters = candidate["orientation_clusters"]
        self.assertEqual(len(clusters), 2)
        self.assertEqual([cluster["count"] for cluster in clusters], [2, 2])
        self.assertAlmostEqual(clusters[0]["total_length"], 50.0)
        self.assertAlmostEqual(clusters[0]["angle"], 44.6, places=1)
        self.assertAlmostEqual(clusters[1]["angle"], 134.5, places=1)

    def test_clusters_orientations_across_zero_and_180_degrees(self):
        candidate = {
            "contours": [
                {"angle": 179.0, "length": 10.0},
                {"angle": 2.0, "length": 10.0},
            ]
        }

        MarkerDetector().cluster_orientations(candidate)

        clusters = candidate["orientation_clusters"]
        self.assertEqual(len(clusters), 1)
        self.assertLess(
            MarkerDetector._orientation_distance(clusters[0]["angle"], 0.0),
            2.0,
        )


class SymbolClassificationTests(unittest.TestCase):
    def test_classifies_diagonal_perpendicular_clusters_as_x(self):
        candidate = {
            "orientation_clusters": [
                {"angle": 44.0, "total_length": 40.0},
                {"angle": 136.0, "total_length": 35.0},
            ]
        }

        MarkerDetector().classify_symbol(candidate)

        self.assertEqual(candidate["symbol"], "X")
        self.assertGreater(candidate["symbol_confidence"], 0.8)
        self.assertAlmostEqual(candidate["orientation_separation"], 88.0)

    def test_classifies_horizontal_vertical_clusters_as_plus(self):
        candidate = {
            "orientation_clusters": [
                {"angle": 3.0, "total_length": 30.0},
                {"angle": 88.0, "total_length": 30.0},
            ]
        }

        MarkerDetector().classify_symbol(candidate)

        self.assertEqual(candidate["symbol"], "+")
        self.assertGreater(candidate["symbol_confidence"], 0.8)

    def test_rejects_candidate_without_two_dominant_directions(self):
        candidate = {"orientation_clusters": [{"angle": 45.0, "total_length": 30.0}]}

        MarkerDetector().classify_symbol(candidate)

        self.assertEqual(candidate["symbol"], "unknown")
        self.assertEqual(candidate["symbol_confidence"], 0.0)

    def test_accepts_rotated_crossing_strokes_as_generic_cross(self):
        first = {
            "line": (np.cos(np.radians(25)), np.sin(np.radians(25)), 50.0, 50.0),
            "line_endpoints": ((20.0, 36.0), (80.0, 64.0)),
            "length": 66.0,
        }
        second = {
            "line": (np.cos(np.radians(115)), np.sin(np.radians(115)), 50.0, 50.0),
            "line_endpoints": ((64.0, 20.0), (36.0, 80.0)),
            "length": 66.0,
        }
        candidate = {
            "radius": 50,
            "orientation_clusters": [
                {"angle": 25.0, "total_length": 66.0, "contours": [first]},
                {"angle": 115.0, "total_length": 66.0, "contours": [second]},
            ],
        }

        MarkerDetector().classify_symbol(candidate)

        self.assertEqual(candidate["symbol"], "cross")
        self.assertGreater(candidate["symbol_confidence"], 0.9)
        self.assertAlmostEqual(candidate["crossing_point"][0], 50.0, places=4)
        self.assertAlmostEqual(candidate["crossing_point"][1], 50.0, places=4)

    def test_rejects_intersection_outside_the_circle(self):
        first = {
            "line": (1.0, 0.0, 10.0, 10.0),
            "line_endpoints": ((0.0, 10.0), (100.0, 10.0)),
            "length": 100.0,
        }
        second = {
            "line": (0.0, 1.0, 10.0, 10.0),
            "line_endpoints": ((10.0, 0.0), (10.0, 100.0)),
            "length": 100.0,
        }
        candidate = {
            "radius": 50,
            "orientation_clusters": [
                {"angle": 0.0, "total_length": 100.0, "contours": [first]},
                {"angle": 90.0, "total_length": 100.0, "contours": [second]},
            ],
        }

        MarkerDetector().score_crossing_lines(candidate)

        self.assertIsNone(candidate["crossing_point"])
        self.assertEqual(candidate["cross_confidence"], 0.0)


class CandidateValidationTests(unittest.TestCase):
    def test_accepts_centered_well_supported_crossing_candidate(self):
        candidate = {
            "radius": 50,
            "confidence": 0.90,
            "crossing_point": (50.0, 50.0),
            "cross_confidence": 0.90,
            "crossing_support_lengths": (80.0, 85.0),
        }

        MarkerDetector().validate_candidate(candidate)

        self.assertTrue(candidate["is_marker"])
        self.assertGreater(candidate["marker_confidence"], 0.80)
        self.assertEqual(candidate["cross_center_offset_ratio"], 0.0)
        self.assertAlmostEqual(candidate["line_support_ratio"], 0.80)

    def test_rejects_crossing_far_from_circle_center(self):
        candidate = {
            "radius": 50,
            "confidence": 0.95,
            "crossing_point": (80.0, 50.0),
            "cross_confidence": 0.95,
            "crossing_support_lengths": (90.0, 90.0),
        }

        MarkerDetector().validate_candidate(candidate)

        self.assertFalse(candidate["is_marker"])
        self.assertGreater(candidate["cross_center_offset_ratio"], 0.45)


if __name__ == "__main__":
    unittest.main()
