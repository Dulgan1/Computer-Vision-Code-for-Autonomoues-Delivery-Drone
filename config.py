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

CAMERA_INDEX = 1

FRAME_WIDTH = 640
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

# Temporal tracking of final, validated marker candidates.
TRACK_SMOOTHING_ALPHA = 0.35
TRACK_ASSOCIATION_DISTANCE = 80
TRACK_CONFIRMATION_FRAMES = 3
TRACK_MAX_MISSED_FRAMES = 5

# =========================
# Camera Calibration and Relative Position
# =========================
# Generate this file with calibrate_camera.py using images from the deployment
# camera. Pose output remains disabled until these values are supplied.
POSE_ENABLED = False
CAMERA_CALIBRATION_FILE = "camera_calibration.npz"
MARKER_DIAMETER_METERS = None

#===========================
# Line Detection
#===========================
HOUGH_LINE_THRESHOLD = 20

MIN_LINE_LENGTH = 20

MAX_LINE_GAP = 5

# =========================
# Contour Filtering
# ==========================
MIN_CONTOUR_AREA = 40

# =========================
# Orientation Clustering
# =========================
# Maximum axial angle difference for contours in the same stroke direction.
ORIENTATION_CLUSTER_TOLERANCE_DEGREES = 15.0

# =========================
# Symbol Classification
# =========================
SYMBOL_ANGLE_TOLERANCE_DEGREES = 20.0
MIN_SYMBOL_CONFIDENCE = 0.60

# A valid generic marker contains two non-parallel strokes that intersect in
# the circular ROI. These limits reject nearly parallel fragments and
# intersections outside the landing marker.
MIN_CROSSING_ANGLE_DEGREES = 20.0
MAX_CROSS_CENTER_OFFSET_RATIO = 0.95
CROSS_SEGMENT_EXTENSION_RATIO = 0.15
MIN_CROSS_CONFIDENCE = 0.45

# =========================
# Final Candidate Validation
# =========================
# The crossing should be close to the marker centre and both directions should
# span a meaningful fraction of the marker diameter.
MAX_INTERSECTION_CENTER_OFFSET_RATIO = 0.45
MIN_LINE_SUPPORT_RATIO = 0.30
MIN_MARKER_CONFIDENCE = 0.65
