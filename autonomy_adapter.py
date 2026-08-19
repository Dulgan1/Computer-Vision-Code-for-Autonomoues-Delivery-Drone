"""Safe, serializable perception messages for a separate autonomy subsystem."""

import json


class AutonomyTargetAdapter:
    """Expose fresh image guidance without embedding vehicle-control logic.

    The adapter intentionally returns an explicit unavailable message whenever
    guidance is absent. A consumer therefore cannot mistake an old detection
    for a current visual measurement.
    """

    SCHEMA_VERSION = 1

    def build(self, guidance):
        """Return a JSON-compatible perception message from image guidance."""

        message = {
            "schema_version": self.SCHEMA_VERSION,
            "target_available": guidance is not None,
            "track_id": None,
            "target_point": None,
            "horizontal_error": None,
            "vertical_error": None,
            "normalized_radius": None,
            "marker_confidence": None,
        }

        if guidance is None:
            return message

        message.update(
            {
                "track_id": guidance["track_id"],
                "target_point": list(guidance["target_point"]),
                "horizontal_error": guidance["horizontal_error"],
                "vertical_error": guidance["vertical_error"],
                "normalized_radius": guidance["normalized_radius"],
                "marker_confidence": guidance["marker_confidence"],
            }
        )
        return message

    def to_json(self, guidance):
        """Serialize the current perception state for a future transport layer."""

        return json.dumps(self.build(guidance), separators=(",", ":"))
