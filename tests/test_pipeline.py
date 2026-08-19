import unittest

import numpy as np

from pipeline import LandingMarkerPipeline


class LandingMarkerPipelineTests(unittest.TestCase):
    def test_blank_frame_returns_no_target_or_guidance(self):
        pipeline = LandingMarkerPipeline()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        result = pipeline.process(frame)

        self.assertEqual(result["gray"].shape, (240, 320))
        self.assertEqual(result["edges"].shape, (240, 320))
        self.assertEqual(result["candidates"], [])
        self.assertIsNone(result["tracked_target"])
        self.assertIsNone(result["guidance"])
