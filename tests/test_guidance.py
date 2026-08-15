import unittest

from guidance import ImageGuidance


def stable_target(center=(320.0, 240.0), crossing_point=None):
    return {
        "track_id": 1,
        "center": center,
        "crossing_point": crossing_point,
        "radius": 60.0,
        "marker_confidence": 0.9,
        "is_stable": True,
        "visible": True,
    }


class ImageGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.guidance = ImageGuidance()
        self.frame_shape = (480, 640, 3)

    def test_centered_target_has_zero_error(self):
        estimate = self.guidance.estimate(stable_target(), self.frame_shape)

        self.assertEqual(estimate["horizontal_error"], 0.0)
        self.assertEqual(estimate["vertical_error"], 0.0)
        self.assertAlmostEqual(estimate["normalized_radius"], 0.25)

    def test_uses_crossing_point_for_target_error(self):
        estimate = self.guidance.estimate(
            stable_target(crossing_point=(480.0, 120.0)), self.frame_shape
        )

        self.assertEqual(estimate["horizontal_error"], 0.5)
        self.assertEqual(estimate["vertical_error"], -0.5)

    def test_does_not_emit_guidance_for_held_or_unstable_track(self):
        target = stable_target()
        target["visible"] = False
        self.assertIsNone(self.guidance.estimate(target, self.frame_shape))

        target["visible"] = True
        target["is_stable"] = False
        self.assertIsNone(self.guidance.estimate(target, self.frame_shape))


if __name__ == "__main__":
    unittest.main()
