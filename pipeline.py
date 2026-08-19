"""Reusable, non-UI implementation of the DroneCV perception pipeline."""

from guidance import ImageGuidance
from detector import MarkerDetector
from preprocess import Preprocessor
from tracker import MarkerTracker


class LandingMarkerPipeline:
    """Process frames into marker candidates, a track, and image guidance."""

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.detector = MarkerDetector()
        self.tracker = MarkerTracker()
        self.guidance = ImageGuidance()

    def process(self, frame):
        """Run one frame through the production perception stages.

        The returned values contain no OpenCV windows or drawing operations,
        making this class usable for offline video evaluation and later service
        integration.
        """

        gray, edges = self.preprocessor.process(frame)
        candidates = self.detector.detect_circles(gray, edges)

        for candidate in candidates:
            self.detector.extract_roi(edges, candidate)
            self.detector.find_contours(candidate)
            self.detector.filter_contours(candidate)
            self.detector.fit_lines(candidate)
            self.detector.compute_line_features(candidate)
            self.detector.cluster_orientations(candidate)
            self.detector.classify_symbol(candidate)
            self.detector.validate_candidate(candidate)

        tracked_target = self.tracker.update(candidates)
        guidance = self.guidance.estimate(tracked_target, frame.shape)

        return {
            "gray": gray,
            "edges": edges,
            "candidates": candidates,
            "tracked_target": tracked_target,
            "guidance": guidance,
        }
