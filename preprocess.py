import cv2
import config
import numpy as np


class Preprocessor:
    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)

    def process(self, frame):
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Reduce sensor noise
        blur = cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL, 0)

        # Step 3: Edge Detection
        edges = cv2.Canny(
            blur,
            config.CANNY_LOW,
            config.CANNY_HIGH
        )

        
        # Step 4: Fill small gaps, repair broken edges
        #clean = cv2.morphologyEx(
        #    edges,
        #    cv2.MORPH_CLOSE,
        #    self.kernel
        #)

        return gray, edges