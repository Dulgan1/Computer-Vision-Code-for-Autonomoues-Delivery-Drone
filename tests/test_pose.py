import unittest

import numpy as np

from pose import CameraModel, TargetPoseEstimator


class TargetPoseEstimatorTests(unittest.TestCase):
    def setUp(self):
        camera_matrix = np.array(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
        )
        self.estimator = TargetPoseEstimator(
            CameraModel(camera_matrix, np.zeros(5)),
            marker_diameter_meters=0.5,
        )

    def test_returns_zero_angles_for_image_center(self):
        target = {
            "is_stable": True,
            "crossing_point": (320.0, 240.0),
            "center": (320.0, 240.0),
            "radius": 100.0,
        }

        estimate = self.estimator.estimate(target, altitude_meters=10.0)

        self.assertAlmostEqual(estimate["horizontal_angle_rad"], 0.0)
        self.assertAlmostEqual(estimate["vertical_angle_rad"], 0.0)
        self.assertEqual(estimate["position_camera_m"], (0.0, 0.0, 10.0))
        self.assertAlmostEqual(estimate["range_from_marker_radius_m"], 1.0)

    def test_projects_offset_target_onto_ground_plane_at_known_altitude(self):
        target = {
            "is_stable": True,
            "crossing_point": (420.0, 240.0),
            "center": (420.0, 240.0),
            "radius": 100.0,
        }

        estimate = self.estimator.estimate(target, altitude_meters=10.0)

        self.assertAlmostEqual(estimate["position_camera_m"][0], 2.5)
        self.assertAlmostEqual(estimate["position_camera_m"][1], 0.0)
        self.assertAlmostEqual(estimate["horizontal_angle_rad"], np.arctan(0.25))

    def test_ignores_unstable_track(self):
        target = {
            "is_stable": False,
            "crossing_point": (320.0, 240.0),
            "center": (320.0, 240.0),
            "radius": 100.0,
        }

        self.assertIsNone(self.estimator.estimate(target))


if __name__ == "__main__":
    unittest.main()
