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

    # Debug drawing
    if config.DEBUG:

        for candidate in candidates:
            x, y = candidate["center"]
            r = candidate["radius"]

            confidence = candidate["confidence"]
            density = candidate["density"]

            cv2.circle(
                frame,
                (x, y),
                r,
                (0, 255, 0),
                2
            )
            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 0, 255),
                -1
            )

            # Confidence
            cv2.putText(
                frame,
                f"Score:{confidence:.2f}",
                (x - 35, y - r - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Edge density
            cv2.putText(
                frame,
                f"Density:{density:.2f}",
                (x - 35, y - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1
            )

            candidate = detector.extract_roi(gray, candidate)

            """if roi is not None:
                cv2.imshow("ROI", roi)"""
            candidate = detector.find_contours(
                candidate
            )

            roi_display = cv2.cvtColor(
                candidate["roi"],
                cv2.COLOR_GRAY2BGR
            )

            cv2.drawContours(
                roi_display,
                candidate["contours"],
                -1,
                (0, 255, 0),
                1
            )

            cv2.imshow("ROI Lines", roi_display)

        cv2.imshow("Camera", frame)
        cv2.imshow("Edges", edges)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


camera.release()

cv2.destroyAllWindows()