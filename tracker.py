"""Lightweight temporal tracker for validated landing-marker candidates."""

from math import hypot

import config


class MarkerTracker:
    """Track one landing marker with exponential smoothing.

    This intentionally tracks a single high-confidence target. It is cheap,
    deterministic, and sufficient for the one intended landing marker while
    avoiding the complexity of a multi-object tracker on the Raspberry Pi.
    """

    def __init__(self):
        self._target = None
        self._next_track_id = 1

    @staticmethod
    def _global_crossing_point(candidate):
        """Convert an ROI-local crossing point to full-image coordinates."""

        crossing_point = candidate.get("crossing_point")
        if crossing_point is None:
            return None

        center_x, center_y = candidate["center"]
        radius = candidate["radius"]
        return (
            center_x - radius + crossing_point[0],
            center_y - radius + crossing_point[1],
        )

    @staticmethod
    def _candidate_target(candidate):
        return {
            "center": tuple(float(value) for value in candidate["center"]),
            "radius": float(candidate["radius"]),
            "marker_confidence": float(candidate["marker_confidence"]),
            "symbol": candidate["symbol"],
            "crossing_point": MarkerTracker._global_crossing_point(candidate),
        }

    def _best_match(self, candidates):
        valid_candidates = [item for item in candidates if item.get("is_marker")]

        if self._target is None:
            return max(
                valid_candidates,
                key=lambda item: item["marker_confidence"],
                default=None,
            )

        target_x, target_y = self._target["center"]
        nearby_candidates = [
            item
            for item in valid_candidates
            if hypot(item["center"][0] - target_x, item["center"][1] - target_y)
            <= config.TRACK_ASSOCIATION_DISTANCE
        ]
        return max(
            nearby_candidates,
            key=lambda item: item["marker_confidence"],
            default=None,
        )

    def _start_track(self, candidate):
        target = self._candidate_target(candidate)
        target.update(
            {
                "track_id": self._next_track_id,
                "hit_count": 1,
                "miss_count": 0,
                "is_stable": False,
                "visible": True,
            }
        )
        self._next_track_id += 1
        self._target = target

    def _update_track(self, candidate):
        measurement = self._candidate_target(candidate)
        alpha = config.TRACK_SMOOTHING_ALPHA

        self._target["center"] = tuple(
            (1.0 - alpha) * self._target["center"][axis]
            + alpha * measurement["center"][axis]
            for axis in range(2)
        )

        self._target["radius"] = (
            (1.0 - alpha) * self._target["radius"]
            + alpha * measurement["radius"]
        )
        self._target["marker_confidence"] = (
            (1.0 - alpha) * self._target["marker_confidence"]
            + alpha * measurement["marker_confidence"]
        )

        if measurement["crossing_point"] is not None:
            previous_crossing = self._target["crossing_point"]
            if previous_crossing is None:
                self._target["crossing_point"] = measurement["crossing_point"]
            else:
                self._target["crossing_point"] = tuple(
                    (1.0 - alpha) * previous_crossing[axis]
                    + alpha * measurement["crossing_point"][axis]
                    for axis in range(2)
                )

        self._target["symbol"] = measurement["symbol"]
        self._target["hit_count"] += 1
        self._target["miss_count"] = 0
        self._target["is_stable"] = (
            self._target["hit_count"] >= config.TRACK_CONFIRMATION_FRAMES
        )
        self._target["visible"] = True

    def update(self, candidates):
        """Update tracking state and return a stable target snapshot or None."""

        candidate = self._best_match(candidates)

        if candidate is None:
            if self._target is None:
                return None

            self._target["miss_count"] += 1
            self._target["visible"] = False

            if self._target["miss_count"] > config.TRACK_MAX_MISSED_FRAMES:
                self._target = None
                return None

            return self._target.copy()

        if self._target is None:
            self._start_track(candidate)
        else:
            self._update_track(candidate)

        return self._target.copy()
