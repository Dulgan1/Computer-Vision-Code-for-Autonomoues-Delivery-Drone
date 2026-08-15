"""import cv2
import time

from camera import Camera
from preprocess import Preprocessor
import detector

cam = Camera()
prep = Preprocessor()
#marker_detector = detector.MarkerDetector()
previous = time.time()

while True:

    frame = cam.read()

    if frame is None:
        break

    current = time.time()
    fps = 1/ (current - previous)
    previous = current
    
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    gray, edges = prep.process(frame)
    circles = detector.MarkerDetector.detect_circles(edges, gray)

    display = frame.copy()

    #cv2.imshow("Original", frame)
    for candidate in circles:
        x, y, r = candidate["center"][0], candidate["center"][1], candidate["radius"]
        cv2.circle(display, (x, y), r, (0, 255, 0), 2)
        cv2.circle(display, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Circle Candidates", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()"""
from camera import Camera
from preprocess import Preprocessor
from detector import MarkerDetector
from tracker import MarkerTracker
from pose import CameraModel, TargetPoseEstimator
from guidance import ImageGuidance

import cv2
import config
import numpy as np
from time import perf_counter


camera = Camera()

preprocessor = Preprocessor()

detector = MarkerDetector()

tracker = MarkerTracker()

image_guidance = ImageGuidance()

pipeline_times = []

pose_estimator = None
if config.POSE_ENABLED:
    if config.MARKER_DIAMETER_METERS is None:
        raise ValueError("MARKER_DIAMETER_METERS is required when POSE_ENABLED is True.")

    pose_estimator = TargetPoseEstimator(
        CameraModel.load(config.CAMERA_CALIBRATION_FILE),
        config.MARKER_DIAMETER_METERS,
    )


while True:

    frame = camera.read()

    if frame is None:
        break

    pipeline_start = perf_counter()

    # Phase 2 & 3
    gray, edges = preprocessor.process(frame)

    # Phase 4
    candidates = detector.detect_circles(gray, edges)

    # Phase 5: symbol pipeline.  The binary edge image is used here rather
    # than grayscale so contours represent marker edges instead of the whole
    # non-zero image region.  Keep this outside DEBUG for production parity.
    for candidate in candidates:
        candidate = detector.extract_roi(edges, candidate)
        candidate = detector.find_contours(candidate)
        candidate = detector.filter_contours(candidate)
        candidate = detector.fit_lines(candidate)
        candidate = detector.compute_line_features(candidate)
        candidate = detector.cluster_orientations(candidate)
        detector.classify_symbol(candidate)
        detector.validate_candidate(candidate)

    tracked_target = tracker.update(candidates)
    guidance_estimate = image_guidance.estimate(tracked_target, frame.shape)
    pose_estimate = None
    if pose_estimator is not None:
        pose_estimate = pose_estimator.estimate(tracked_target)

    if config.PERFORMANCE_LOGGING:
        pipeline_times.append(perf_counter() - pipeline_start)

        if len(pipeline_times) >= config.PERFORMANCE_LOG_INTERVAL:
            average_ms = 1000.0 * sum(pipeline_times) / len(pipeline_times)
            print(f"Average detection pipeline: {average_ms:.1f} ms/frame")
            pipeline_times.clear()

    # Debug drawing
    if config.DEBUG:

        debug_frame = frame.copy()

        for candidate in candidates:

            # ----------------------------------
            # Draw Circle Detection
            # ----------------------------------
            x, y = candidate["center"]
            r = candidate["radius"]

            confidence = candidate["confidence"]
            density = candidate["density"]

            marker_color = (0, 255, 0) if candidate["is_marker"] else (0, 165, 255)
            cv2.circle(
                debug_frame,
                (x, y),
                r,
                marker_color,
                2
            )

            cv2.circle(
                debug_frame,
                (x, y),
                2,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                debug_frame,
                f"Score:{confidence:.2f}",
                (x - 35, y - r - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

            cv2.putText(
                debug_frame,
                f"Marker:{candidate['marker_confidence']:.2f}",
                (x - 35, y - r - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                marker_color,
                2,
            )

            cv2.putText(
                debug_frame,
                f"Density:{density:.2f}",
                (x - 35, y - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255,255,0),
                1
            )

            roi_display = cv2.cvtColor(
                candidate["roi"],
                cv2.COLOR_GRAY2BGR
            )

            cv2.drawContours(
                roi_display,
                [item["contour"] for item in candidate["contours"]],
                -1,
                (0,255,0),
                2
            )

            for contour_object in candidate["contours"]:
                endpoints = contour_object["line_endpoints"]

                if endpoints is None:
                    continue

                start, end = endpoints
                start = tuple(round(value) for value in start)
                end = tuple(round(value) for value in end)
                cv2.line(roi_display, start, end, (255, 0, 0), 1)

                midpoint = tuple(round(value) for value in contour_object["midpoint"])
                cv2.putText(
                    roi_display,
                    f"{contour_object['angle']:.0f} deg",
                    midpoint,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 0, 0),
                    1,
                )

            cv2.putText(
                roi_display,
                f"Contours: {len(candidate['contours'])}",
                (5,20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,255),
                1
            )

            cluster_angles = ", ".join(
                f"{cluster['angle']:.0f} deg ({cluster['count']})"
                for cluster in candidate["orientation_clusters"]
            )
            cv2.putText(
                roi_display,
                f"Directions: {cluster_angles or 'none'}",
                (5, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                roi_display,
                f"Symbol: {candidate['symbol']} ({candidate['symbol_confidence']:.2f})",
                (5, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
            )

            if candidate["crossing_point"] is not None:
                crossing_point = tuple(round(value) for value in candidate["crossing_point"])
                cv2.circle(roi_display, crossing_point, 3, (0, 0, 255), -1)

            cv2.imshow(
                "ROI",
                roi_display
            )

        if tracked_target is not None:
            track_x, track_y = (round(value) for value in tracked_target["center"])
            track_color = (255, 255, 0) if tracked_target["is_stable"] else (0, 255, 255)
            track_state = "stable" if tracked_target["is_stable"] else "acquiring"
            visibility = "visible" if tracked_target["visible"] else "held"
            cv2.circle(debug_frame, (track_x, track_y), 6, track_color, -1)
            cv2.putText(
                debug_frame,
                f"Track {tracked_target['track_id']}: {track_state}, {visibility}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                track_color,
                2,
            )

        if pose_estimate is not None:
            cv2.putText(
                debug_frame,
                f"Bearing: {np.degrees(pose_estimate['horizontal_angle_rad']):.1f} deg",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        if guidance_estimate is not None:
            cv2.putText(
                debug_frame,
                (
                    "Image error: "
                    f"x={guidance_estimate['horizontal_error']:+.2f}, "
                    f"y={guidance_estimate['vertical_error']:+.2f}"
                ),
                (10, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        cv2.imshow(
            "Camera",
            debug_frame
        )

        cv2.imshow(
            "Edges",
            edges
        )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


camera.release()

cv2.destroyAllWindows()
