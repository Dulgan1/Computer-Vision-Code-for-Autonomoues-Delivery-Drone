import json
import unittest

from autonomy_adapter import AutonomyTargetAdapter


class AutonomyTargetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = AutonomyTargetAdapter()

    def test_reports_target_unavailable_without_fresh_guidance(self):
        message = self.adapter.build(None)

        self.assertFalse(message["target_available"])
        self.assertIsNone(message["horizontal_error"])
        self.assertIsNone(message["track_id"])

    def test_converts_guidance_to_json_compatible_message(self):
        guidance = {
            "track_id": 4,
            "target_point": (321.5, 119.0),
            "horizontal_error": 0.1,
            "vertical_error": -0.2,
            "normalized_radius": 0.25,
            "marker_confidence": 0.92,
        }

        message = self.adapter.build(guidance)
        decoded = json.loads(self.adapter.to_json(guidance))

        self.assertTrue(message["target_available"])
        self.assertEqual(message["target_point"], [321.5, 119.0])
        self.assertEqual(decoded, message)
