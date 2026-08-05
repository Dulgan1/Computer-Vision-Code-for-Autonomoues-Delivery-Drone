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

import cv2
import config


camera = Camera()

preprocessor = Preprocessor()

detector = MarkerDetector()


while True:

    frame = camera.read()

    if frame is None:
        break

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

            cv2.circle(
                debug_frame,
                (x, y),
                r,
                (0, 255, 0),
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
