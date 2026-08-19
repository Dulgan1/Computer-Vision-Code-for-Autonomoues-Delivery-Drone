# DroneCV: Classical Vision for a Drone Landing Marker

## Executive summary

DroneCV is a deterministic OpenCV perception subsystem for recognising a
circular drone landing marker with two crossing strokes. It accepts conventional
`X` and `+` symbols as well as a rotated or perspective-distorted `cross` when
their geometry is consistent with the marker.

The system is designed for a Raspberry Pi 3 Model A+ (512 MB RAM). It uses no
machine learning, GPU processing, CUDA, TensorRT, or cloud services. Every
result comes from inspectable image evidence and geometry.

```text
camera frame -> validated marker -> stable temporal track -> image alignment error
```

DroneCV is perception only. It does not command motors or bypass a flight
controller. A separate autonomy subsystem must apply safety gates before using
its output.

## Marker and project scope

The marker consists of a circular outer boundary and two substantial strokes
that intersect inside it. A valid crossing must be supported by visible contour
evidence and lie near the circle centre.

The current system detects, validates, tracks, and provides uncalibrated
image-space alignment guidance. It does not yet provide trusted metre-level
position unless camera calibration, a real marker diameter, and altitude data
are supplied.

## Pipeline architecture

```text
USB/onboard camera
  -> grayscale -> Gaussian blur -> Canny edge image
  -> Hough circle proposals -> circle-quality checks
  -> masked binary circular ROI -> contours
  -> fitted lines -> orientation clusters -> crossing verification
  -> final per-frame marker validation -> temporal tracking
  -> normalized image guidance / optional calibrated camera ray
```

| Module | Responsibility |
| --- | --- |
| `camera.py` | Opens the selected camera and requests resolution, FPS, and a small capture buffer. |
| `preprocess.py` | Builds the grayscale and Canny edge images. |
| `detector.py` | Owns circle detection, ROI geometry, crossing detection, and validation. |
| `tracker.py` | Tracks and smooths one validated marker through time. |
| `guidance.py` | Emits unitless image-centre error only for a fresh stable target. |
| `pipeline.py` | Single non-UI processing interface shared by live capture and video evaluation. |
| `autonomy_adapter.py` | Converts fresh guidance into a JSON-compatible perception message; it sends no vehicle commands. |
| `calibrate_camera.py` | Calibrates the deployment camera from checkerboard photographs. |
| `pose.py` | Creates calibrated camera rays and optional relative-position estimates. |
| `main.py` | Runs the pipeline and its debug views. |

## Image processing workflow

### 1. Image acquisition

OpenCV `VideoCapture` opens the configured camera. Width, height, and FPS are
requests to the driver; the camera can return a different supported mode. The
actual resolution must be checked before calibration and deployment.

The code requests a one-frame capture buffer. Where supported by the camera
backend, this reduces latency caused by processing stale frames.

### 2. Grayscale conversion and Gaussian smoothing

Colour is unnecessary because the marker is recognised from intensity changes
and geometry. The BGR frame is converted to grayscale, then smoothed with a
Gaussian kernel:

```math
G(x,y) = \frac{1}{2\pi\sigma^2}
\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
```

This suppresses high-frequency sensor noise. Excessive smoothing would remove
thin marker edges; insufficient smoothing would create false edges.

> The active code path is grayscale -> Gaussian blur -> Canny. Configuration
> contains placeholders for adaptive thresholding and morphology, but those
> operations are not currently active and must not be claimed as implemented.

### 3. Canny edge detection

Canny converts the blurred image into a binary edge representation. From image
gradients `Gx` and `Gy`, its basic quantities are:

```math
|G| = \sqrt{G_x^2 + G_y^2}
```

```math
\theta = \operatorname{atan2}(G_y, G_x)
```

Non-maximum suppression and hysteresis thresholding retain meaningful edges.
Pixels above `CANNY_HIGH` are strong edges; pixels between `CANNY_LOW` and
`CANNY_HIGH` remain only if connected to strong edges.

This binary image is deliberately used for symbol contours. Contours extracted
from grayscale would treat almost every non-zero pixel as foreground and would
not isolate the marker strokes.

### 4. Circle proposals and quality scoring

The Hough Circle Transform proposes outer-boundary candidates. A circle with
centre `(a,b)` and radius `r` obeys:

```math
(x-a)^2 + (y-b)^2 = r^2
```

Edge pixels vote for plausible centres and radii. Configuration bounds the
search with a radius range and Hough parameters.

Candidates are rejected if they touch the image border, duplicate an accepted
candidate, or have insufficient edge evidence near their circumference. For
`N_e` edge pixels under a thin circular mask:

```math
D = \frac{N_e}{2\pi r}
```

`D` may exceed one because real boundaries may be thick. The circle heuristic
uses:

```math
S_d = \min(D,1)
```

```math
S_r = \max\left(0,1-\frac{|r-r_{expected}|}{r_{max}-r_{min}}\right)
```

```math
S_{circle} = 0.7S_d + 0.3S_r
```

This and all later confidence values are deterministic ranking heuristics, not
statistical probabilities.

### 5. Circular region of interest

For every candidate, a `2r x 2r` region is extracted from the binary edge
image. A filled circular mask removes pixels outside the proposed boundary. In
this local ROI, the marker centre is `(r,r)`.

Filled masks and thin circumference masks are cached by radius. The configured
radius range is bounded, so the cache stays small while avoiding repeated array
allocation on the Raspberry Pi.

### 6. Contours and contour objects

`findContours` extracts external contours using `CHAIN_APPROX_NONE`, preserving
their points. Contours smaller than `MIN_CONTOUR_AREA` are rejected as noise.

Each retained contour is enriched once and carried through later stages:

```python
{
    "contour": contour_points,
    "area": area,
    "perimeter": perimeter,
    "bbox": (x, y, width, height),
}
```

This candidate-enrichment pattern prevents repeated computation and keeps the
vision pipeline inspectable.

### 7. Fitted contour lines

The project intentionally does not classify symbols with Hough Lines. Thick,
worn, and hollow strokes often generate multiple edge lines for a single
physical stroke.

Instead, `cv2.fitLine` fits a line directly to contour points. The result is a
unit direction vector `(v_x,v_y)` and a point `(x_0,y_0)` on the line:

```math
\mathbf{p}(t) =
\begin{bmatrix}x_0\\y_0\end{bmatrix}
+ t\begin{bmatrix}v_x\\v_y\end{bmatrix}
```

The L2 fit uses the contour geometry directly and is more tolerant of imperfect
paint than a pixel-space line vote.

### 8. Orientation, support length, and midpoint

Line orientation is calculated as:

```math
\alpha = \operatorname{atan2}(v_y,v_x)
```

It is normalised to `[0,180)` degrees because a physical line is axial: 0 and
180 degrees are the same orientation.

For every contour point `q`, the projection on the fitted line is:

```math
t_q = (\mathbf{q}-\mathbf{p_0})\cdot\mathbf{v}
```

The smallest and largest projections define finite support endpoints:

```math
\mathbf{s}=\mathbf{p_0}+t_{min}\mathbf{v}, \qquad
\mathbf{e}=\mathbf{p_0}+t_{max}\mathbf{v}
```

Support length is `||e-s||`; midpoint is `(s+e)/2`. These quantities stay
aligned with a diagonal stroke, unlike an axis-aligned bounding box.

### 9. Orientation clustering

Multiple contour edges can describe one physical stroke. Similar axial
orientations are clustered using:

```math
d(\alpha,\beta)=|((\alpha-\beta+90)\bmod180)-90|
```

Longer contours establish and influence clusters more strongly. The mean angle
uses doubled angles to handle the 0/180-degree wrap-around:

```math
\bar{\alpha}=\frac{1}{2}\operatorname{atan2}
\left(\sum_i w_i\sin(2\alpha_i),\sum_i w_i\cos(2\alpha_i)\right)
```

where `w_i` is the contour support length.

### 10. X, plus, and crossing-line geometry

The two strongest clusters are compared with ideal labels:

```text
X: 45 degrees and 135 degrees
+: 0 degrees and 90 degrees
```

These labels are useful but not mandatory. A rotated landing marker is accepted
as a generic `cross` if two non-parallel, contour-supported lines physically
intersect inside the candidate circle.

For lines `p1 + td1` and `p2 + ud2`, define:

```math
\operatorname{cross}(a,b)=a_xb_y-a_yb_x
```

The intersection parameter is:

```math
t=\frac{\operatorname{cross}(p_2-p_1,d_2)}
        {\operatorname{cross}(d_1,d_2)}
```

and intersection point is `p1 + td1`. The implementation verifies that the
intersection is on both finite support segments, permits only a small extension
for broken edges, requires enough angular separation, and requires the point
to lie inside the circle.

### 11. Final marker validation

Independent circle, crossing, centring, and support evidence are combined:

```math
S_{marker}=0.30S_{circle}+0.35S_{cross}
          +0.20S_{center}+0.15S_{support}
```

The crossing must be near the ROI centre, both directions must have meaningful
support relative to the circle diameter, and the combined score must meet
`MIN_MARKER_CONFIDENCE`.

The final per-frame object includes:

```python
{
    "center": (x, y),
    "radius": r,
    "symbol": "X" | "+" | "cross" | "unknown",
    "crossing_point": (roi_x, roi_y) | None,
    "marker_confidence": score,
    "is_marker": True | False,
}
```

## Temporal tracking and image guidance

### Tracking

Single-frame detection is too unstable for flight. `MarkerTracker` associates
nearby validated candidates and smooths measurements with an exponential moving
average:

```math
\hat{x}_t=(1-\lambda)\hat{x}_{t-1}+\lambda z_t
```

where `z_t` is the current measurement and `lambda` is
`TRACK_SMOOTHING_ALPHA`.

A target becomes stable after `TRACK_CONFIRMATION_FRAMES` detections. It is
held for a limited number of missed frames, marked `visible = False` while
held, then expires. Held targets are never treated as fresh guidance.

### Uncalibrated guidance

For a stable and visible target at pixel `(u,v)` in an image of width `W` and
height `H`, `guidance.py` emits:

```math
e_x=\frac{u-W/2}{W/2}, \qquad e_y=\frac{v-H/2}{H/2}
```

These errors lie in `[-1,1]`. Positive `e_x` means right of image centre;
positive `e_y` means below it. Apparent marker radius is also reported.

These outputs are unitless, not metre offsets, and are intended for simulation
or a future autonomy interface rather than direct flight commands.

### Autonomy-facing perception message

`AutonomyTargetAdapter` converts fresh guidance into a transport-neutral,
JSON-compatible message:

```json
{
  "schema_version": 1,
  "target_available": true,
  "track_id": 4,
  "target_point": [321.5, 119.0],
  "horizontal_error": 0.10,
  "vertical_error": -0.20,
  "normalized_radius": 0.25,
  "marker_confidence": 0.92
}
```

When no fresh, stable visual target exists, `target_available` is `false` and
all measurement fields are `null`. This is an intentional safety contract:
the CV system never emits a held/lost track as if it were a live measurement.
The adapter does not choose a network, serial, ROS, or MAVLink transport and
does not issue any vehicle command.

## Calibration and relative position

`calibrate_camera.py` estimates the deployment camera's intrinsic matrix and
lens distortion from checkerboard photographs:

```math
K=\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}
```

Images must be taken with the exact USB/onboard camera, resolution, and lens
configuration planned for use. The entire checkerboard must be sharp and
visible in at least ten varied images.

After calibration, an image point becomes an undistorted camera ray
`(x_n,y_n,1)`. Bearings are:

```math
\phi_x=\arctan(x_n), \qquad \phi_y=\arctan(y_n)
```

With trusted altitude `h`, a level-ground approximation is:

```math
(X,Y,Z)=(x_nh,y_nh,h)
```

Known physical marker diameter also enables a fronto-parallel range
approximation. It is not a replacement for full pose estimation under tilt.
Calibration remains disabled by default until real calibration data exists.

## Debugging, testing, and performance

## Configuration reference

Every `config.py` setting, including active, optional, and currently inactive
settings, is explained in plain language in the
[configuration guide](docs/configuration-guide.md). It explains what each
setting controls, what happens when it is raised or lowered, and a safe order
for tuning with recorded video.

The following is the complete at-a-glance reference. **Active** means the
current program reads the value. **Inactive** means it is retained for planned
or retired functionality and changing it currently does nothing. **Optional**
means it matters only when the related feature is enabled.

| Configuration setting | Status | Plain-language effect |
| --- | --- | --- |
| `DEBUG` | Active | Shows or hides debug windows. `True` helps development but costs CPU; it does not change the detection decision. |
| `CAMERA_INDEX` | Active | Chooses the camera number, such as laptop camera `0` or a USB camera often at `1+`. |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | Active requests | Ask the camera for image size. More pixels can see smaller markers but cost CPU; the camera can ignore unsupported sizes. |
| `FPS` | Active request | Asks for camera frames per second. Higher values can improve responsiveness but cost more processing. |
| `CAMERA_BUFFER_SIZE` | Active where supported | Requests a short camera queue to reduce delayed, stale frames. |
| `PERFORMANCE_LOGGING` | Active | Enables periodic pipeline timing output. |
| `PERFORMANCE_LOG_INTERVAL` | Active | Number of frames used for each printed timing average; lower reports sooner, higher is steadier. |
| `VERBOSE_PIPELINE_LOGGING` | Active | Prints the autonomy-facing target message every frame; useful briefly, but expensive/noisy if left on. |
| `GAUSSIAN_KERNEL` | Active | Blur size before edge detection. Larger removes more noise but can erase thin marker edges; values normally need to be odd. |
| `ADAPTIVE_BLOCK_SIZE` | Inactive | Reserved for planned adaptive thresholding; has no effect today. |
| `ADAPTIVE_C` | Inactive | Reserved for planned adaptive thresholding; has no effect today. |
| `MORPH_KERNEL_SIZE` | Inactive | Reserved for planned morphology; has no effect today. |
| `CANNY_LOW` | Active | Weak-edge threshold. Lower keeps faint edges and more noise; higher removes weak edges. |
| `CANNY_HIGH` | Active | Strong-edge threshold. Higher requires clearer edges and reduces clutter; it must stay above `CANNY_LOW`. |
| `HOUGH_DP` | Active | Circle-search resolution scale. Lower is more detailed/costly; higher is coarser/faster. |
| `HOUGH_MIN_DIST` | Active | Minimum separation between circle centres. Higher removes nearby duplicates; lower allows more candidates. |
| `HOUGH_PARAM2` | Active | Circle-vote strictness. Higher finds fewer/cleaner circles; lower finds more circles and more false positives. |
| `MIN_RADIUS`, `MAX_RADIUS` | Active | Smallest/largest acceptable circle radius in pixels. Widening the range detects more sizes but increases work and false candidates. |
| `MIN_EDGE_DENSITY` | Active | Required edge evidence around a circle. Higher rejects weak circles; lower accepts worn circles and more false ones. |
| `DUPLICATE_DISTANCE` | Active | How close two circle centres can be before one is treated as a duplicate. |
| `SEARCH_WINDOW` | Inactive | Reserved for future tracker-only region searching; has no effect today. |
| `FULL_SCAN_INTERVAL` | Inactive | Reserved for future periodic full-image scans; has no effect today. |
| `TRACK_SMOOTHING_ALPHA` | Active | Balance between smoothness and responsiveness. Near `0` is stable but slow; near `1` follows raw measurements and jitters. |
| `TRACK_ASSOCIATION_DISTANCE` | Active | Largest allowed motion between frames for a detection to remain the same track. Higher tolerates motion but risks switching targets. |
| `TRACK_CONFIRMATION_FRAMES` | Active | Consecutive detections required before a track is stable. Higher is safer but slower. |
| `TRACK_MAX_MISSED_FRAMES` | Active | Brief loss tolerated before deleting a track. Held tracks never create active guidance. |
| `POSE_ENABLED` | Optional | Enables calibrated bearing/range features. Leave `False` for the current uncalibrated workflow. |
| `CAMERA_CALIBRATION_FILE` | Optional | Path to camera calibration data; used only when pose is enabled. |
| `MARKER_DIAMETER_METERS` | Optional | Physical marker size for optional range approximation; unused while pose is disabled. |
| `HOUGH_LINE_THRESHOLD`, `MIN_LINE_LENGTH`, `MAX_LINE_GAP` | Inactive | Old Hough-line settings. The project now uses contour fitting, so these do nothing. |
| `MIN_CONTOUR_AREA` | Active | Rejects tiny contour blobs. Higher removes more noise but can lose thin/distant strokes. |
| `ORIENTATION_CLUSTER_TOLERANCE_DEGREES` | Active | Angle allowance for contour edges to count as one stroke. Higher merges more fragments; lower separates directions more strictly. |
| `SYMBOL_ANGLE_TOLERANCE_DEGREES` | Active | Allowance for optional X/+ labels. Higher accepts more rotated symbols as X/+; generic `cross` remains available. |
| `MIN_SYMBOL_CONFIDENCE` | Active | Score needed for the specific X/+ label. Higher is stricter; generic crossing uses separate rules. |
| `MIN_CROSSING_ANGLE_DEGREES` | Active | Rejects two directions that are nearly parallel. Higher requires a clearer cross. |
| `MAX_CROSS_CENTER_OFFSET_RATIO` | Active | Broad crossing check: intersection must remain within the circle. |
| `CROSS_SEGMENT_EXTENSION_RATIO` | Active | Small extra line extension allowed for broken edges. Higher is more forgiving but can accept unrelated lines. |
| `MIN_CROSS_CONFIDENCE` | Active | Minimum geometric crossing score before a generic cross is accepted. |
| `MAX_INTERSECTION_CENTER_OFFSET_RATIO` | Active | Final strict centring limit for the crossing. Lower requires a more centred marker. |
| `MIN_LINE_SUPPORT_RATIO` | Active | Minimum visible length for the weaker stroke relative to circle diameter. Higher requires clearer strokes. |
| `MIN_MARKER_CONFIDENCE` | Active | Final acceptance threshold. Higher is safer/stricter; lower finds more targets and can add false positives. |

For recommended tuning order, current values, examples, and fuller
explanations of each trade-off, use the linked configuration guide before
changing a value.

Run the live system:

```bash
venv/bin/python main.py
```

Set `CAMERA_INDEX` to the USB-camera index. On Linux, use
`v4l2-ctl --list-devices` to identify devices and
`v4l2-ctl --list-formats-ext -d /dev/videoN` to inspect supported modes.

With `DEBUG = True`, windows show Canny edges, ROI, contours, fitted lines,
cluster directions, crossing intersection, per-frame validation, tracker state,
and image-centre error. Green circles are accepted markers; orange circles are
rejected candidates.

Run automated tests:

```bash
venv/bin/python -m unittest discover -s tests -v
```

Evaluate a recorded USB-camera video with the same non-UI pipeline used by the
application:

```bash
venv/bin/python evaluate_video.py videos/marker_test.mp4 --csv evaluation.csv
```

The CSV has one row per frame, including candidate count, validated-marker
count, track status, image guidance errors, and processing time. The command
also reports marker/stable-track coverage plus mean and 95th-percentile frame
processing time. This makes threshold tuning reproducible rather than relying
only on visual inspection.

Enable timing on the Raspberry Pi:

```python
PERFORMANCE_LOGGING = True
PERFORMANCE_LOG_INTERVAL = 60
```

The reported average excludes debug drawing and OpenCV windows. Measure with
the actual deployment camera because laptop performance is not representative.

## Safety boundary and remaining work

DroneCV is not an autopilot. Any autonomy system consuming its output must
enforce manual override, explicit state transitions, velocity and altitude
limits, target-loss handling, timeouts, and an abort path.

Before real autonomous landing, complete:

1. USB-camera video validation and threshold tuning.
2. Checkerboard camera calibration and physical marker measurement.
3. Camera-to-drone-body mounting/extrinsic calibration.
4. Autonomy simulation and constrained ground/tethered testing.
5. Incremental flight testing under appropriate safety procedures.
