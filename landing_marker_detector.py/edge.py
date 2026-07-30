import cv2
import numpy as np


class EdgeDetector:

    def compute_gradient(self, gray):

        gx = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(gx, gy)

        angle = cv2.phase(
            gx,
            gy,
            angleInDegrees=True
        )

        return gx, gy, magnitude, angle