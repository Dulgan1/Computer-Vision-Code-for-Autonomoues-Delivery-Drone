"""Calibrate the deployment camera from checkerboard photographs.

Example:
    venv/bin/python calibrate_camera.py "calibration_images/*.png" \
        --square-size 0.024 --output camera_calibration.npz

``--square-size`` is the physical side length of one checkerboard square in
metres. ``--board-columns`` and ``--board-rows`` are *inner* corner counts.
"""

import argparse
from glob import glob

import cv2
import numpy as np


def calibrate(image_paths, board_size, square_size_meters):
    """Return intrinsics, distortion coefficients, RMS error, and image size."""

    object_template = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_template[:, :2] = np.indices(board_size).T.reshape(-1, 2)
    object_template *= square_size_meters

    object_points = []
    image_points = []
    image_size = None

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size)

        if not found:
            continue

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            ),
        )
        object_points.append(object_template)
        image_points.append(refined_corners)
        image_size = gray.shape[::-1]

    if len(object_points) < 10:
        raise ValueError(
            "At least 10 checkerboard detections are required; "
            f"found {len(object_points)}."
        )

    rms_error, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    return camera_matrix, distortion, rms_error, image_size, len(object_points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_glob", help="Glob for checkerboard images")
    parser.add_argument("--square-size", required=True, type=float, help="Square side in metres")
    parser.add_argument("--board-columns", type=int, default=9, help="Inner corners across")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner corners down")
    parser.add_argument("--output", default="camera_calibration.npz")
    arguments = parser.parse_args()

    image_paths = sorted(glob(arguments.image_glob))
    if not image_paths:
        raise ValueError("No calibration images match the supplied glob.")

    result = calibrate(
        image_paths,
        (arguments.board_columns, arguments.board_rows),
        arguments.square_size,
    )
    camera_matrix, distortion, rms_error, image_size, used_images = result
    np.savez(
        arguments.output,
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=np.array(image_size),
        rms_error=rms_error,
    )
    print(f"Saved {arguments.output} from {used_images} images; RMS error: {rms_error:.3f} px")


if __name__ == "__main__":
    main()
