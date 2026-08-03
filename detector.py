"""import cv2
import numpy as np
import config


class MarkerDetector:

    def detect_circles(self, edge_image, gray_image):

        candidates = []

        circles = cv2.HoughCircles(
            edge_image,
            cv2.HOUGH_GRADIENT,
            dp=config.HOUGH_DP,
            minDist=config.HOUGH_MIN_DIST,
            param1=config.CANNY_HIGH,
            param2=config.HOUGH_PARAM2,
            minRadius=config.MIN_RADIUS,
            maxRadius=config.MAX_RADIUS,
        )

        if circles is None:
            return candidates

        circles = np.round(circles[0]).astype(int)

        h, w = gray_image.shape

        for x, y, r in circles:

            if (
                x - r < 0 or
                y - r < 0 or
                x + r >= w or
                y + r >= h
            ):
                continue

            roi = gray_image[
                y-r:y+r,
                x-r:x+r
            ]

            candidates.append({
                "center": (x, y),
                "radius": r,
                "roi": roi,
            })

        return candidates"""
import cv2
import numpy as np
import config


class MarkerDetector:
    def __init__(self):
        pass

    def detect_circles(self, gray, edges):
        """
        Detect circles using Hough Circle Transform.
        Returns:
            List of circles [(x, y, r), ...]
        """

        detected = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=config.HOUGH_DP,
            minDist=config.HOUGH_MIN_DIST,
            param1=config.CANNY_HIGH,
            param2=config.HOUGH_PARAM2,
            minRadius=config.MIN_RADIUS,
            maxRadius=config.MAX_RADIUS
        )

        if detected is None:
            return []

        detected = np.round(detected[0]).astype(int)

        h, w = gray.shape

        candidates = []
        expected_radius = (config.MIN_RADIUS + config.MAX_RADIUS) / 2

        for x, y, r in detected:
            # Border check: ensure the circle is fully within the image boundaries
            if (
                x - r < 0 or
                y - r < 0 or
                x + r >= w or
                y + r >= h
            ):
                continue

            duplicate = False

            for candidate in candidates:
                cx, cy = candidate["center"]
                distance = np.hypot((x - cx), (y - cy))
                if distance < config.DUPLICATE_DISTANCE:
                    duplicate = True
                    break

            if duplicate:
                continue

            # ROI Extraction
            roi_edges = edges[y-r:y+r, x-r:x+r]
            mask = np.zeros(
                roi_edges.shape,
                dtype=np.uint8
                )
            
            cv2.circle(
                mask, 
                (r, r),
                r,
                255,
                thickness=2
                )

            edge_pixels = cv2.countNonZero(
                cv2.bitwise_and(
                    roi_edges,
                    mask
                )
            )

            circumference = 2 * np.pi * r

            density = edge_pixels / circumference

            if density < config.MIN_EDGE_DENSITY:
                continue

            density_score = min(density, 1.0)

            radius_score = 1.0 - (
            abs(r - expected_radius)
            /
            (config.MAX_RADIUS - config.MIN_RADIUS)
        )
            radius_score = max(0.0, radius_score)

            confidence = (
            0.7 * density_score +
            0.3 * radius_score
        )



            candidates.append({

            "center": (x, y),

            "radius": r,

            "density": density,

            "confidence": confidence

        })

        return candidates

    def extract_roi(self, gray, candidate):
        """
        Extract circular ROI from grayscale image.
        """

        x, y = candidate["center"]
        r = candidate["radius"]

        roi = gray[y-r:y+r, x-r:x+r]

        # Circular mask
        mask = np.zeros(roi.shape, dtype=np.uint8)

        cv2.circle(
            mask,
            (r, r),
            r,
            255,
            -1
        )

        roi = cv2.bitwise_and(roi, mask)

        candidate["roi"] = roi

        return candidate
    
    def find_contours(self, candidate):

        roi = candidate["roi"]
        contours, hierarchy = cv2.findContours(
            roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        candidate["contours"] = contours
        candidate["hierarchy"] = hierarchy

        return candidate

"""    def detect_lines(self, roi):

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=config.HOUGH_LINE_THRESHOLD,
            minLineLength=config.MIN_LINE_LENGTH,
            maxLineGap=config.MAX_LINE_GAP,
        )

        if lines is None:
            return []

        return lines.reshape(-1, 4) # [line[0] for line in lines]"""