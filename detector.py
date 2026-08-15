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
        # Radius is bounded by configuration, so this remains a small cache.
        # Reusing masks avoids two array allocations per circle candidate.
        self._circle_masks = {}

    def _circle_mask(self, radius, thickness):
        """Return a reusable circle mask for a square ROI of side ``2 * radius``."""

        cache_key = (radius, thickness)
        mask = self._circle_masks.get(cache_key)

        if mask is None:
            mask = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)
            cv2.circle(mask, (radius, radius), radius, 255, thickness)
            self._circle_masks[cache_key] = mask

        return mask

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
            mask = self._circle_mask(r, thickness=2)

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

    def extract_roi(self, image, candidate):
        """Extract a circular ROI from a grayscale or binary input image."""

        x, y = candidate["center"]
        r = candidate["radius"]

        roi = image[y-r:y+r, x-r:x+r]

        mask = self._circle_mask(r, thickness=-1)

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

    def filter_contours(self, candidate):
        """Discard small contours and attach reusable geometric features.

        ``find_contours`` remains the OpenCV-facing stage and stores raw
        contour arrays. After this method, ``candidate[\"contours\"]`` contains
        contour objects for the geometry stages that follow.
        """

        filtered = []

        for contour in candidate["contours"]:
            area = cv2.contourArea(contour)

            if area < config.MIN_CONTOUR_AREA:
                continue

            perimeter = cv2.arcLength(contour, closed=True)
            x, y, width, height = cv2.boundingRect(contour)

            filtered.append({
                "contour": contour,
                "area": area,
                "perimeter": perimeter,
                "bbox": (x, y, width, height),
            })

        candidate["contours"] = filtered

        return candidate

    def fit_lines(self, candidate):
        """Fit a direction line to each retained contour.

        OpenCV returns a unit direction vector ``(vx, vy)`` and a point
        ``(x0, y0)`` on the fitted line. The next stage uses that direction to
        compute orientation; keeping the point makes the fit inspectable in
        the debug view and available for later geometric checks.
        """

        for contour_object in candidate["contours"]:
            contour = contour_object["contour"]

            if len(contour) < 2:
                contour_object["line"] = None
                continue

            line = cv2.fitLine(
                contour,
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            ).reshape(4)

            contour_object["line"] = tuple(float(value) for value in line)

        return candidate

    def compute_line_features(self, candidate):
        """Derive finite, orientation-aware features from fitted contour lines.

        A fitted line is infinite, so its useful marker-stroke length comes
        from projecting every contour point onto its unit direction vector.
        The minimum and maximum projections define the two support endpoints.
        """

        for contour_object in candidate["contours"]:
            line = contour_object["line"]

            if line is None:
                contour_object["angle"] = None
                contour_object["length"] = None
                contour_object["midpoint"] = None
                contour_object["line_endpoints"] = None
                continue

            vx, vy, x0, y0 = line
            direction = np.array([vx, vy], dtype=np.float32)
            point_on_line = np.array([x0, y0], dtype=np.float32)
            points = contour_object["contour"].reshape(-1, 2).astype(np.float32)

            projections = (points - point_on_line) @ direction
            start = point_on_line + projections.min() * direction
            end = point_on_line + projections.max() * direction
            midpoint = (start + end) / 2.0

            contour_object["angle"] = float(np.degrees(np.arctan2(vy, vx)) % 180.0)
            contour_object["length"] = float(np.linalg.norm(end - start))
            contour_object["midpoint"] = tuple(float(value) for value in midpoint)
            contour_object["line_endpoints"] = (
                tuple(float(value) for value in start),
                tuple(float(value) for value in end),
            )

        return candidate

    @staticmethod
    def _orientation_distance(first_angle, second_angle):
        """Return the smallest difference between two orientations in degrees.

        Orientations have a 180-degree period: 0 and 180 degrees describe the
        same undirected line, unlike a heading where they are opposites.
        """

        return abs((first_angle - second_angle + 90.0) % 180.0 - 90.0)

    @staticmethod
    def _mean_orientation(contours):
        """Compute a length-weighted mean for orientations with 180-degree period."""

        angles = np.radians([2.0 * item["angle"] for item in contours])
        weights = np.array([item["length"] for item in contours])
        mean_angle = 0.5 * np.degrees(
            np.arctan2(
                np.sum(weights * np.sin(angles)),
                np.sum(weights * np.cos(angles)),
            )
        )
        return float(mean_angle % 180.0)

    def cluster_orientations(self, candidate):
        """Group contour directions into the marker's dominant orientations.

        Longer contours establish clusters first, so substantial marker strokes
        outweigh short residual fragments. Each cluster keeps its member
        contours, count, total support length, and a length-weighted angle.
        """

        usable_contours = [
            item
            for item in candidate["contours"]
            if item.get("angle") is not None and item.get("length", 0.0) > 0.0
        ]
        usable_contours.sort(key=lambda item: item["length"], reverse=True)

        clusters = []

        for contour_object in usable_contours:
            nearest_cluster = None
            nearest_distance = float("inf")

            for cluster in clusters:
                distance = self._orientation_distance(
                    contour_object["angle"], cluster["angle"]
                )

                if distance < nearest_distance:
                    nearest_cluster = cluster
                    nearest_distance = distance

            if nearest_distance > config.ORIENTATION_CLUSTER_TOLERANCE_DEGREES:
                nearest_cluster = {"contours": []}
                clusters.append(nearest_cluster)

            nearest_cluster["contours"].append(contour_object)
            nearest_cluster["angle"] = self._mean_orientation(
                nearest_cluster["contours"]
            )
            nearest_cluster["count"] = len(nearest_cluster["contours"])
            nearest_cluster["total_length"] = sum(
                item["length"] for item in nearest_cluster["contours"]
            )

        candidate["orientation_clusters"] = sorted(
            clusters,
            key=lambda cluster: cluster["total_length"],
            reverse=True,
        )

        return candidate

    def _template_error(self, observed_angles, target_angles):
        """Return the best mean error for either pairing of two orientations."""

        direct_error = (
            self._orientation_distance(observed_angles[0], target_angles[0])
            + self._orientation_distance(observed_angles[1], target_angles[1])
        ) / 2.0
        swapped_error = (
            self._orientation_distance(observed_angles[0], target_angles[1])
            + self._orientation_distance(observed_angles[1], target_angles[0])
        ) / 2.0
        return min(direct_error, swapped_error)

    @staticmethod
    def _cross_product(first, second):
        return first[0] * second[1] - first[1] * second[0]

    def score_crossing_lines(self, candidate):
        """Score the strongest pair of distinct directions that crosses in the ROI.

        Unlike X/+ templates, this accepts a rotated or perspective-distorted
        marker. The intersection must lie on both finite contour supports and
        inside the candidate circle, not merely where two infinite lines meet.
        """

        candidate["crossing_point"] = None
        candidate["crossing_angle"] = None
        candidate["cross_confidence"] = 0.0
        candidate["crossing_support_lengths"] = None

        radius = candidate.get("radius")
        if radius is None:
            return candidate

        best_crossing = None
        clusters = candidate.get("orientation_clusters", [])

        for first_index, first_cluster in enumerate(clusters):
            for second_cluster in clusters[first_index + 1:]:
                first_contours = [
                    item for item in first_cluster.get("contours", [])
                    if item.get("line") is not None
                ]
                second_contours = [
                    item for item in second_cluster.get("contours", [])
                    if item.get("line") is not None
                ]

                if not first_contours or not second_contours:
                    continue

                first = max(first_contours, key=lambda item: item["length"])
                second = max(second_contours, key=lambda item: item["length"])
                first_direction = np.array(first["line"][:2], dtype=np.float32)
                second_direction = np.array(second["line"][:2], dtype=np.float32)
                first_point = np.array(first["line"][2:], dtype=np.float32)
                second_point = np.array(second["line"][2:], dtype=np.float32)

                determinant = self._cross_product(first_direction, second_direction)
                crossing_angle = np.degrees(np.arcsin(min(1.0, abs(determinant))))

                if crossing_angle < config.MIN_CROSSING_ANGLE_DEGREES:
                    continue

                point_difference = second_point - first_point
                first_parameter = self._cross_product(point_difference, second_direction) / determinant
                second_parameter = self._cross_product(point_difference, first_direction) / determinant
                intersection = first_point + first_parameter * first_direction

                def is_on_support(contour_object, parameter):
                    start, end = contour_object["line_endpoints"]
                    point_on_line = np.array(contour_object["line"][2:], dtype=np.float32)
                    direction = np.array(contour_object["line"][:2], dtype=np.float32)
                    start_parameter = np.dot(np.array(start) - point_on_line, direction)
                    end_parameter = np.dot(np.array(end) - point_on_line, direction)
                    extension = contour_object["length"] * config.CROSS_SEGMENT_EXTENSION_RATIO
                    return min(start_parameter, end_parameter) - extension <= parameter <= max(
                        start_parameter, end_parameter
                    ) + extension

                if not (
                    is_on_support(first, first_parameter)
                    and is_on_support(second, second_parameter)
                ):
                    continue

                center_offset = np.linalg.norm(intersection - np.array([radius, radius]))
                maximum_offset = radius * config.MAX_CROSS_CENTER_OFFSET_RATIO

                if center_offset > maximum_offset:
                    continue

                angle_score = (
                    crossing_angle - config.MIN_CROSSING_ANGLE_DEGREES
                ) / (90.0 - config.MIN_CROSSING_ANGLE_DEGREES)
                center_score = 1.0 - center_offset / maximum_offset
                support_balance = min(
                    first_cluster["total_length"], second_cluster["total_length"]
                ) / max(first_cluster["total_length"], second_cluster["total_length"])
                confidence = (
                    0.55 * angle_score
                    + 0.30 * center_score
                    + 0.15 * support_balance
                )

                if best_crossing is None or confidence > best_crossing["confidence"]:
                    best_crossing = {
                        "point": tuple(float(value) for value in intersection),
                        "angle": float(crossing_angle),
                        "confidence": float(confidence),
                        "support_lengths": (
                            float(first_cluster["total_length"]),
                            float(second_cluster["total_length"]),
                        ),
                    }

        if best_crossing is not None:
            candidate["crossing_point"] = best_crossing["point"]
            candidate["crossing_angle"] = best_crossing["angle"]
            candidate["cross_confidence"] = best_crossing["confidence"]
            candidate["crossing_support_lengths"] = best_crossing["support_lengths"]

        return candidate

    def classify_symbol(self, candidate):
        """Classify the two dominant directions as an X, +, cross, or unknown.

        The score is a deterministic heuristic combining template agreement,
        near-perpendicularity, and the balance of line support. It is useful
        for ranking candidates, but is not a probability.
        """

        clusters = candidate.get("orientation_clusters", [])

        candidate["symbol"] = "unknown"
        candidate["symbol_confidence"] = 0.0
        candidate["orientation_separation"] = None
        self.score_crossing_lines(candidate)

        if len(clusters) < 2:
            return candidate

        first, second = clusters[:2]
        observed_angles = (first["angle"], second["angle"])
        separation = self._orientation_distance(*observed_angles)
        candidate["orientation_separation"] = separation

        templates = {
            "X": (45.0, 135.0),
            "+": (0.0, 90.0),
        }
        symbol, template_error = min(
            (
                (name, self._template_error(observed_angles, target_angles))
                for name, target_angles in templates.items()
            ),
            key=lambda result: result[1],
        )

        tolerance = config.SYMBOL_ANGLE_TOLERANCE_DEGREES
        template_score = max(0.0, 1.0 - template_error / tolerance)
        perpendicular_score = max(0.0, 1.0 - abs(90.0 - separation) / tolerance)
        support_balance = min(first["total_length"], second["total_length"]) / max(
            first["total_length"], second["total_length"]
        )
        confidence = (
            0.5 * template_score
            + 0.3 * perpendicular_score
            + 0.2 * support_balance
        )

        candidate["symbol_confidence"] = confidence

        if (
            template_score > 0.0
            and confidence >= config.MIN_SYMBOL_CONFIDENCE
        ):
            candidate["symbol"] = symbol
        elif candidate["cross_confidence"] >= config.MIN_CROSS_CONFIDENCE:
            candidate["symbol"] = "cross"
            candidate["symbol_confidence"] = candidate["cross_confidence"]

        return candidate

    def validate_candidate(self, candidate):
        """Combine circle, crossing, centering, and support evidence.

        This is the final per-frame acceptance gate. Its confidence ranks
        candidates; it is deliberately not a probability estimate.
        """

        candidate["marker_confidence"] = 0.0
        candidate["is_marker"] = False
        candidate["cross_center_offset_ratio"] = None
        candidate["line_support_ratio"] = None

        radius = candidate.get("radius")
        crossing_point = candidate.get("crossing_point")
        support_lengths = candidate.get("crossing_support_lengths")

        if radius is None or crossing_point is None or support_lengths is None:
            return candidate

        crossing_point = np.array(crossing_point, dtype=np.float32)
        roi_center = np.array([radius, radius], dtype=np.float32)
        center_offset_ratio = float(np.linalg.norm(crossing_point - roi_center) / radius)
        support_ratio = min(support_lengths) / (2.0 * radius)
        candidate["cross_center_offset_ratio"] = center_offset_ratio
        candidate["line_support_ratio"] = support_ratio

        circle_score = float(np.clip(candidate.get("confidence", 0.0), 0.0, 1.0))
        crossing_score = float(candidate.get("cross_confidence", 0.0))
        center_score = max(
            0.0,
            1.0 - center_offset_ratio / config.MAX_INTERSECTION_CENTER_OFFSET_RATIO,
        )
        support_score = min(1.0, support_ratio)
        marker_confidence = (
            0.30 * circle_score
            + 0.35 * crossing_score
            + 0.20 * center_score
            + 0.15 * support_score
        )
        candidate["marker_confidence"] = marker_confidence

        candidate["is_marker"] = bool(
            crossing_score >= config.MIN_CROSS_CONFIDENCE
            and center_offset_ratio <= config.MAX_INTERSECTION_CENTER_OFFSET_RATIO
            and support_ratio >= config.MIN_LINE_SUPPORT_RATIO
            and marker_confidence >= config.MIN_MARKER_CONFIDENCE
        )

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
