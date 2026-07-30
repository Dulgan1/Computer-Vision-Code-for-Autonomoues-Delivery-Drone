"""
Landing Marker Detector Configuration
Target Platform:
Raspberry Pi 3 Model A+ (512 MB RAM)
"""

# Mode

DEBUG = True
# ===========================
# CAMERA
# ===========================

CAMERA_INDEX = 0

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

FPS = 15


# ===========================
# PREPROCESSING
# ===========================

GAUSSIAN_KERNEL = (5, 5)

ADAPTIVE_BLOCK_SIZE = 11

ADAPTIVE_C = 2


# ===========================
# MORPHOLOGY
# ===========================

MORPH_KERNEL_SIZE = 3


# ===========================
# EDGE DETECTION
# ===========================

CANNY_LOW = 50
CANNY_HIGH = 120


# ===========================
# CIRCLE DETECTION
# ===========================


HOUGH_DP = 1.2
HOUGH_MIN_DIST = 40
HOUGH_PARAM2 = 20
MIN_RADIUS = 40
MAX_RADIUS = 100

# ==========================
# Circle Validation
# ==========================

MIN_EDGE_DENSITY = 0.50      # Tune experimentally
DUPLICATE_DISTANCE = 10       # pixels

# ===========================
# TRACKING
# ===========================

SEARCH_WINDOW = 80

FULL_SCAN_INTERVAL = 15

#===========================
# Line Detection
#===========================
HOUGH_LINE_THRESHOLD = 20

MIN_LINE_LENGTH = 20

MAX_LINE_GAP = 5
