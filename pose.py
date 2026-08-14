"""Calibrated image-target to camera-relative-position conversion."""

from pathlib import Path

import cv2
import numpy as np


class CameraModel:
    """Camera intrinsics and distortion parameters for the deployment camera."""

    def __init__(self, camera_matrix, distortion):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(distortion, dtype=np.float64)

        if self.camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3).")

    @classmethod
    def load(cls, calibration_path):
        calibration_path = Path(calibration_path)
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

        with np.load(calibration_path) as calibration:
            return cls(calibration["camera_matrix"], calibration["distortion"])

    def normalized_ray(self, image_point):
        """Return the undistorted camera ray (x, y, 1) through an image point."""

        point = np.array([[image_point]], dtype=np.float64)
        normalized = cv2.undistortPoints(point, self.camera_matrix, self.distortion)
        x, y = normalized.reshape(2)
        return float(x), float(y), 1.0


class TargetPoseEstimator:
    """Estimate calibrated angular and camera-relative target measurements."""

    def __init__(self, camera_model, marker_diameter_meters=None):
        self.camera_model = camera_model
        self.marker_diameter_meters = marker_diameter_meters

    def estimate(self, tracked_target, altitude_meters=None):
        """Return target angles and optional metric estimates.

        If ``altitude_meters`` is supplied from a rangefinder or flight
        controller, the ray is intersected with the level ground plane. If a
        marker diameter is supplied, a fronto-parallel range approximation is
        also reported. It is not a substitute for full solvePnP under tilt.
        """

        if tracked_target is None or not tracked_target["is_stable"]:
            return None

        image_point = tracked_target["crossing_point"] or tracked_target["center"]
        ray_x, ray_y, ray_z = self.camera_model.normalized_ray(image_point)
        estimate = {
            "image_point": image_point,
            "ray": (ray_x, ray_y, ray_z),
            "horizontal_angle_rad": float(np.arctan(ray_x)),
            "vertical_angle_rad": float(np.arctan(ray_y)),
            "position_camera_m": None,
            "range_from_marker_radius_m": None,
        }

        if altitude_meters is not None:
            estimate["position_camera_m"] = (
                ray_x * altitude_meters,
                ray_y * altitude_meters,
                float(altitude_meters),
            )

        if self.marker_diameter_meters is not None:
            focal_length_pixels = (self.camera_model.camera_matrix[0, 0] + self.camera_model.camera_matrix[1, 1]) / 2.0
            estimate["range_from_marker_radius_m"] = float(
                focal_length_pixels * self.marker_diameter_meters
                / (2.0 * tracked_target["radius"])
            )

        return estimate
