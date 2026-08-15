"""Uncalibrated image-space guidance signals for a stable landing target."""


class ImageGuidance:
    """Convert a stable visible track into normalized image-space errors.

    These values are deliberately unitless. They are suitable for simulation
    and for a future autonomy interface, but are not metre offsets or motor
    commands.
    """

    def estimate(self, tracked_target, frame_shape):
        """Return normalized target error, or None without a fresh stable target.

        Positive horizontal error means the target is right of image centre.
        Positive vertical error means the target is below image centre.
        """

        if (
            tracked_target is None
            or not tracked_target["is_stable"]
            or not tracked_target["visible"]
        ):
            return None

        frame_height, frame_width = frame_shape[:2]
        target_point = tracked_target["crossing_point"] or tracked_target["center"]
        image_center = (frame_width / 2.0, frame_height / 2.0)
        horizontal_error = (target_point[0] - image_center[0]) / image_center[0]
        vertical_error = (target_point[1] - image_center[1]) / image_center[1]

        return {
            "track_id": tracked_target["track_id"],
            "target_point": target_point,
            "horizontal_error": max(-1.0, min(1.0, horizontal_error)),
            "vertical_error": max(-1.0, min(1.0, vertical_error)),
            "normalized_radius": tracked_target["radius"] / min(image_center),
            "marker_confidence": tracked_target["marker_confidence"],
        }
