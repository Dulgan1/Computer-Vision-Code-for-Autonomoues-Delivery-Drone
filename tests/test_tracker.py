import unittest

import config
from tracker import MarkerTracker


def marker(center, confidence=0.9, radius=50, symbol="cross"):
    return {
        "is_marker": True,
        "center": center,
        "radius": radius,
        "marker_confidence": confidence,
        "symbol": symbol,
        "crossing_point": (radius, radius),
    }


class MarkerTrackerTests(unittest.TestCase):
    def test_confirms_track_after_consistent_detections(self):
        tracker = MarkerTracker()

        first = tracker.update([marker((100, 100))])
        second = tracker.update([marker((102, 100))])
        third = tracker.update([marker((104, 100))])

        self.assertFalse(first["is_stable"])
        self.assertFalse(second["is_stable"])
        self.assertTrue(third["is_stable"])
        self.assertEqual(third["hit_count"], config.TRACK_CONFIRMATION_FRAMES)

    def test_smooths_position_measurements(self):
        tracker = MarkerTracker()
        tracker.update([marker((100, 100))])

        target = tracker.update([marker((120, 100))])

        self.assertAlmostEqual(
            target["center"][0],
            100 + config.TRACK_SMOOTHING_ALPHA * 20,
        )
        self.assertEqual(target["center"][1], 100.0)

    def test_holds_then_expires_track_after_missed_frames(self):
        tracker = MarkerTracker()
        tracker.update([marker((100, 100))])

        held_target = tracker.update([])
        self.assertFalse(held_target["visible"])
        self.assertEqual(held_target["miss_count"], 1)

        for _ in range(config.TRACK_MAX_MISSED_FRAMES):
            expired_target = tracker.update([])

        self.assertIsNone(expired_target)

    def test_ignores_distant_candidate_while_track_is_active(self):
        tracker = MarkerTracker()
        tracker.update([marker((100, 100))])

        target = tracker.update([marker((250, 100), confidence=1.0)])

        self.assertFalse(target["visible"])
        self.assertEqual(target["center"], (100.0, 100.0))


if __name__ == "__main__":
    unittest.main()
