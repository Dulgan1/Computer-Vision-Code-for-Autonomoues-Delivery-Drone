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
camera frame -> validated marker -> stable temporal track -> image alignment
             -> body-frame angle target -> Pixhawk / payload decision
```

The reusable vision pipeline is a perception subsystem: it neither drives
motors nor bypasses the flight controller. The current `main.py` integration
does, however, publish MAVLink landing-target angles to a connected Pixhawk and
contains a configurable payload-servo release gate. Those are integration
features, not proof that an autonomous flight or release is safe. Flight-mode,
arming, geofence, altitude, operator-override, and abort decisions remain the
responsibility of the flight-controller/autonomy system and operating crew.

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
  -> normalized image guidance
  -> pixel-to-body-frame angle conversion -> MAVLink LANDING_TARGET
  -> consecutive-alignment gate -> optional payload-servo command
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
| `mav_adapter.py` | Converts live marker pixels to approximate body-frame angular offsets and sends MAVLink messages; also contains the servo command helper. |
| `calibrate_camera.py` | Calibrates the deployment camera from checkerboard photographs. |
| `pose.py` | Creates calibrated camera rays and optional relative-position estimates. |
| `main.py` | Opens camera and Pixhawk connections, runs the pipeline, transmits fresh landing targets, applies the current release gate, and optionally shows debug views. |

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

### Image coordinates and uncalibrated guidance

OpenCV uses an image coordinate system whose origin is the **top-left pixel**:
`x`/`u` increases to the right and `y`/`v` increases downward. The image centre
is `(W/2,H/2)`. `target_point` is the preferred landing point in these full-frame
pixel coordinates: the measured crossing if one exists, otherwise the detected
circle centre. It is not an offset and it is not a metre position.

For a stable and visible target at pixel `(u,v)` in an image of width `W` and
height `H`, `guidance.py` emits normalized displacement:

```math
e_x=\frac{u-W/2}{W/2}, \qquad e_y=\frac{v-H/2}{H/2}
```

These errors lie in `[-1,1]`. Positive `e_x` means right of image centre;
positive `e_y` means below it. Apparent marker radius is also reported. For
example, at 640 x 240, a point 16 pixels right of centre has
`e_x = 16 / 320 = 0.05`.

These outputs are unitless, not metre offsets. They express where the marker
appears in the camera image and are valid only while the target is both stable
and visible in the current frame. A target retained by the tracker after a
missed frame is deliberately not emitted as guidance.

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

### Live runtime workflow (`main.py`)

The live program combines the pipeline with the current Pixhawk integration in
the following order on every frame:

1. Open the configured camera, request its width, height, FPS, and a short
   capture buffer. The driver may choose a different mode, so the real mode
   must match the assumptions below.
2. Connect to `/dev/ttyAMA0` at 921600 baud and wait up to five seconds for a
   MAVLink heartbeat. If no heartbeat arrives, the program releases the camera
   and exits rather than running without its expected flight-controller link.
3. Capture one frame and run `LandingMarkerPipeline.process(frame)`: grayscale,
   blur, Canny edges, circle proposals, ROI/cross validation, tracking, and
   image guidance.
4. Build an `AutonomyTargetAdapter` message for logs or a future independent
   autonomy transport. This message itself does not control a vehicle.
5. Only when guidance is fresh (stable **and** visible) convert its
   `target_point` to body-frame angular offsets and send `LANDING_TARGET`.
6. If payload release has not already happened, count only consecutive fresh
   frames whose horizontal and vertical image errors are both within the
   configured alignment threshold. A missing, unstable, held, or off-centre
   target resets that count immediately.
7. Once the count reaches its threshold, issue one `MAV_CMD_DO_SET_SERVO`
   command. `payload_dropped` latches true so this process cannot send a second
   release command during the same program run.

The normal loop ends when the camera returns no frame or the user presses `q`.
It then releases the camera and closes OpenCV windows. Debug windows are only
created when `DEBUG = True`.

### Pixel-to-body angle conversion and MAVLink output

`mav_adapter.py` currently assumes a 640 x 240 image, horizontal FOV of 67.5
degrees, vertical FOV of 42.5 degrees, and an installation where the top of the
camera image points toward the drone nose. Its image-centre constants are:

```math
C_x = 640/2, \qquad C_y = 240/2
```

For the target pixel, first form image offsets:

```math
\Delta x=u-C_x, \qquad \Delta y=v-C_y
```

The camera-to-body convention used by the code is then:

```math
p_{forward}=-\Delta y, \qquad p_{right}=\Delta x
```

Thus, a marker above the image centre becomes a positive forward direction;
one to the right becomes a positive right direction. With horizontal and
vertical fields of view `F_h` and `F_v`, the angular approximation is:

```math
\theta_{forward}=\arctan\left(\frac{p_{forward}}{C_y}
                         \tan\frac{F_v}{2}\right)
```

```math
\theta_{right}=\arctan\left(\frac{p_{right}}{C_x}
                       \tan\frac{F_h}{2}\right)
```

The runtime sends these as `angle_x` and `angle_y` in a MAVLink
`LANDING_TARGET` message using `MAV_FRAME_BODY_FRD`. It sends `distance = 0.0`;
therefore this message does **not** provide a measured altitude or marker range.
How a particular autopilot firmware and flight mode consume an angle-only
message must be verified in a simulator and on the intended hardware.

This conversion is a field-of-view model, not full camera calibration. Its
quality depends on the actual FOV, the real capture resolution, negligible lens
distortion, and correct camera mounting. The constants in `mav_adapter.py` are
not automatically derived from `config.py`. Change and validate them together
whenever the camera mode, lens, crop, or installation changes. If the camera is
rotated, mirrored, or mounted differently, the forward/right signs can be
wrong; validate directions without propellers before flight.

### Current payload-alignment gate

The present values are defined near the top of `main.py`:

| Value | Current setting | Meaning |
| --- | ---: | --- |
| `ALIGNMENT_THRESHOLD` | `0.05` | Both normalized image errors must be smaller than 0.05 in magnitude. At 640 x 240, this is within roughly 16 pixels horizontally and 6 pixels vertically of the image centre. |
| `STABLE_FRAMES_REQUIRED` | `10` | Number of consecutive fresh, in-tolerance frames required. At a true 15 FPS this is about 0.67 seconds, but the actual time depends on achieved frame rate. |
| Servo channel | `9` | Channel supplied to `MAV_CMD_DO_SET_SERVO`. It must be the correctly configured, safe payload channel on the flight controller. |
| Open PWM | `2000` | PWM value sent to that channel when the release condition is met. It must be verified against the physical mechanism. |

This gate is strictly image alignment; it does not check flight mode, altitude,
ground clearance, vehicle velocity, arming state, payload status, or whether a
human has authorized release. It must be treated as a development integration
until those independent safety interlocks exist and have been tested.

## Optional calibration and relative position

The deployed workflow can detect, centre, track, and transmit **angular**
marker direction without a checkerboard calibration. That is the current mode:
`POSE_ENABLED = False`, and no trusted metre altitude, range, or lateral metre
offset is claimed by the vision pipeline.

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
Calibration remains optional and disabled by default. If reliable metre-scale
position is needed later, use validated calibration with a known marker size,
or combine camera bearings with a separate trusted altitude/range sensor. Do
not present apparent pixel radius or the FOV-based MAVLink angle as a measured
distance in metres.

## Debugging, testing, and performance

For development, set `DEBUG = True` and present the marker at different sizes,
rotations, lighting levels, and positions. The live overlays expose the
evidence chain: candidate circle, circle density, contour edges, fitted-line
directions, crossing point, final confidence, track state, and normalized
image error. Set `DEBUG = False` for normal operation to avoid UI overhead.

Before connecting a live payload, test the complete runtime in stages:

1. Run the perception pipeline with recorded video and confirm a live marker
   produces a stable track and errors with the expected sign.
2. With motors disarmed and the payload disconnected, confirm that moving the
   marker up/down/left/right yields the expected forward/right body angles.
3. Confirm that a lost marker, an unstable marker, or a marker outside the
   threshold resets the alignment count.
4. Use a harmless test load or no load to validate the intended servo channel
   and PWM. The current code can command the servo after its release gate.

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

### Running and validating the system

The runtime imports OpenCV (`cv2`), NumPy, and `pymavlink`. It also expects a
working camera and, when `main.py` is used unchanged, a Pixhawk MAVLink link on
`/dev/ttyAMA0` at 921600 baud. Confirm the virtual environment contains these
packages before deployment.

**Important:** `main.py` is not a vision-only preview in its current form. It
waits for a Pixhawk heartbeat, sends `LANDING_TARGET` messages when a live
target is available, and can send the configured servo command after ten
aligned frames. For perception-only validation, use the automated tests or
`evaluate_video.py` below rather than attaching an active payload.

Run the integrated live system only after the connection, servo configuration,
and safety conditions above have been checked:

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

DroneCV is not an autopilot. The vision code supplies marker evidence and the
current integration sends target angles plus a single servo command; neither
substitutes for a flight-safety system. Any vehicle using this repository must
enforce manual override, explicit state transitions, velocity and altitude
limits, target-loss handling, timeouts, geofencing, a tested abort path, and
payload-release authorization outside the image-only gate.

Before real autonomous operation, complete and document:

1. USB-camera video validation and threshold tuning in conditions that match
   the flight environment.
2. Verification that the camera's actual resolution, FOV, orientation, and
   `mav_adapter.py` constants agree; validate the forward/right sign convention
   with motors disarmed.
3. Simulator, bench, and constrained ground/tethered tests of the specific
   flight-controller handling of `LANDING_TARGET`.
4. A separate release state machine with flight-mode, altitude, velocity,
   operator-authorization, and mechanical payload interlocks.
5. Camera-to-drone-body mounting validation; optional calibrated camera and
   physical marker measurement only if metre-level estimates are required.
6. Incremental flight testing under appropriate safety procedures.
