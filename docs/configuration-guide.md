# DroneCV Configuration Guide

This guide explains every setting in [`config.py`](../config.py) in plain
language. Think of `config.py` as the control panel for the vision program.
Most settings are numbers that tell the program how strict or forgiving it
should be.

## Read this before changing values

1. Change **one setting at a time**.
2. Test the change using the same USB camera, resolution, and type of lighting
   that will be used on the drone.
3. Use the debug windows and `evaluate_video.py` report to decide whether the
   change helped.
4. Do not assume that a bigger number is always better. Most settings trade
   false detections against missed detections.
5. Some settings exist for future work but are not used by the current code.
   They are marked **Reserved / inactive** below. Changing them currently has
   no effect.

## Quick status guide

| Status | Meaning |
| --- | --- |
| **Active** | The current program reads and uses this value. |
| **Optional** | Used only when its associated feature is enabled. |
| **Reserved / inactive** | Kept for planned or retired code; changing it currently does nothing. |

## 1. Program mode and camera settings

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `DEBUG` | `True` | Active | Turns on the visual debugging windows. `True` shows the camera, edges, ROI, contours, fitted lines, crossing point, tracker, and scores. It is helpful while developing but costs CPU and memory. Set it to `False` on the Raspberry Pi for normal non-visual operation. It does **not** change the detection decision itself. |
| `CAMERA_INDEX` | `1` | Active | Selects which camera OpenCV opens. `0` is often the laptop camera; a USB camera is often `1` or higher. This is not guaranteed—use `v4l2-ctl --list-devices` on Linux to confirm. |
| `FRAME_WIDTH` | `640` | Active request | Asks the camera for this many horizontal pixels. More pixels can make a small far-away marker easier to see, but require more CPU and memory. Cameras may ignore an unsupported request. Always use a resolution that the USB camera actually supports. |
| `FRAME_HEIGHT` | `240` | Active request | Asks the camera for this many vertical pixels. It must make sense together with width. A 16:9 camera commonly supports pairs such as `640 x 360`; `640 x 240` is unusually wide and may be ignored or cropped by a camera driver. The actual received frame size is what matters. |
| `FPS` | `15` | Active request | Asks the camera to provide this many frames per second. Higher FPS gives quicker response but increases USB, CPU, and memory pressure. The driver can ignore unsupported values. |
| `CAMERA_BUFFER_SIZE` | `1` | Active where supported | Requests that OpenCV keep only about one captured frame waiting in a queue. A small buffer reduces lag: the detector sees what is happening now rather than old frames. Some camera backends ignore this request. |

## 2. Performance and logging settings

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `PERFORMANCE_LOGGING` | `False` | Active | When `True`, the program measures processing time and periodically prints the average milliseconds per frame. It measures the vision pipeline, not debug drawing or OpenCV windows. Use it on the Raspberry Pi to measure actual performance. |
| `PERFORMANCE_LOG_INTERVAL` | `60` | Active when performance logging is on | Number of frames collected before printing one average timing result. A smaller number prints feedback sooner but makes the average noisier. A larger number is steadier but slower to report changes. |
| `VERBOSE_PIPELINE_LOGGING` | `False` | Active | When `True`, prints the autonomy-facing perception message every frame. This helps inspect the data contract, but printing continuously can slow the Pi and flood the terminal. Leave `False` during normal operation. |

## 3. Preprocessing settings

Preprocessing prepares the camera image before searching for a marker.

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `GAUSSIAN_KERNEL` | `(5, 5)` | Active | Controls the blur size used before edge detection. It must normally be odd numbers, such as `(3,3)` or `(5,5)`. A larger kernel removes more grain/noise but can blur thin marker edges. A smaller kernel preserves detail but may create noisy false edges. `(5,5)` is a moderate starting point. |
| `ADAPTIVE_BLOCK_SIZE` | `11` | Reserved / inactive | Intended for adaptive thresholding, where local image areas choose their own brightness threshold. The active pipeline currently uses Canny edges instead, so changing this has no effect. If activated later, this must be an odd number greater than 1; larger areas react more slowly to local lighting changes. |
| `ADAPTIVE_C` | `2` | Reserved / inactive | Intended to adjust the threshold used by adaptive thresholding. It currently has no effect because adaptive thresholding is not enabled. If used later, increasing it generally makes fewer pixels pass the local threshold. |
| `MORPH_KERNEL_SIZE` | `3` | Reserved / inactive | Intended for morphological opening/closing, small pixel operations used to remove specks or join tiny gaps. The morphology code is currently disabled, so this value has no effect. If enabled later, larger kernels repair larger gaps but can merge shapes that should stay separate. |

## 4. Edge detection settings

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `CANNY_LOW` | `50` | Active | Lower edge-strength threshold for Canny. Lowering it keeps more weak edges, which may recover faint marker paint but also adds noise and clutter. Raising it removes more weak edges. |
| `CANNY_HIGH` | `120` | Active | Strong edge-strength threshold for Canny. Raising it requires sharper brightness changes before an edge is considered strong, reducing noise but possibly losing faint marker boundaries. It should be greater than `CANNY_LOW`. |

## 5. Circle proposal settings

The outer circle is found first. These settings determine which circles are
considered plausible.

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `HOUGH_DP` | `1.2` | Active | Hough-circle accumulator resolution scale. Roughly, lower values inspect circle centres more finely and can be more precise but use more CPU. Higher values are faster/coarser and can miss or shift circles. Do not change this casually; `1.2` is a practical middle value. |
| `HOUGH_MIN_DIST` | `40` | Active | Minimum pixel distance between proposed circle centres. Increasing it prevents nearby duplicate circles but can miss two truly close circles. Decreasing it allows nearby proposals and may create duplicates. Since only one landing marker is expected, this is mainly duplicate control. |
| `HOUGH_PARAM2` | `20` | Active | Circle-vote threshold. Higher values are stricter: fewer, more confident circle proposals but more missed circles. Lower values are more forgiving: more proposals, including false circles, and more downstream CPU work. This is one of the most important values to tune with recorded videos. |
| `MIN_RADIUS` | `40` | Active | Smallest accepted marker radius in pixels. Increasing it ignores small/far-away markers but reduces false positives and CPU work. Decreasing it permits more distant/smaller markers but makes false circular patterns more likely. |
| `MAX_RADIUS` | `100` | Active | Largest accepted marker radius in pixels. Increasing it permits close/large markers but expands the Hough search and can add false positives. Decreasing it ignores very close or large markers. It must be greater than `MIN_RADIUS`. |

## 6. Circle validation settings

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `MIN_EDGE_DENSITY` | `0.50` | Active | Minimum amount of edge evidence around the proposed circle. The program counts edge pixels close to the circumference and divides by circumference length. Raising this rejects weak/broken circles more aggressively; lowering it accepts worn or poorly lit circles but increases false circles. |
| `DUPLICATE_DISTANCE` | `10` | Active | If two proposed circle centres are closer than this many pixels, only the first accepted proposal is kept. Increasing it removes more duplicates but could merge different nearby circles. Lowering it permits more duplicate work. |

## 7. Tracking settings

The tracker follows one accepted marker over several frames so the drone does
not react to a single noisy image.

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `SEARCH_WINDOW` | `80` | Reserved / inactive | A planned search-area size for future optimisation. The current tracker does not crop to a search window, so changing it has no effect. |
| `FULL_SCAN_INTERVAL` | `15` | Reserved / inactive | A planned number of frames between full-image scans when a tracker is active. The current detector scans every frame, so changing it has no effect. |
| `TRACK_SMOOTHING_ALPHA` | `0.35` | Active | Controls how quickly the smoothed tracked position follows a new measurement. `0` would never move; `1` would use the raw new measurement with no smoothing. A smaller value is steadier but lags behind motion. A larger value reacts quickly but jitters more. `0.35` gives 35% weight to the new frame and 65% to the previous tracked value. |
| `TRACK_ASSOCIATION_DISTANCE` | `80` | Active | Maximum pixel distance from the previous tracked centre for a new marker to be considered the same target. Increasing it tolerates faster target movement but can jump to a false marker. Decreasing it avoids target swaps but can lose a rapidly moving target. |
| `TRACK_CONFIRMATION_FRAMES` | `3` | Active | Number of consecutive matching detections required before a track becomes `stable`. Increasing it makes the system safer against one-frame false positives but delays guidance. Decreasing it reacts faster but is less trustworthy. |
| `TRACK_MAX_MISSED_FRAMES` | `5` | Active | Number of consecutive missed detections tolerated before deleting a track. Increasing it tolerates short occlusion but retains old information longer. Decreasing it abandons the target sooner. During these held frames, `visible` is false and guidance is deliberately unavailable. |

## 8. Optional calibration and relative-position settings

These are not needed for the current uncalibrated image-guidance workflow.

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `POSE_ENABLED` | `False` | Optional | Turns calibrated camera-ray and bearing output on/off. `False` keeps the system in image-space mode. Do not set it to `True` unless valid camera calibration data and marker size are available; otherwise the program stops with a clear configuration error. |
| `CAMERA_CALIBRATION_FILE` | `camera_calibration.npz` | Optional | File path where camera intrinsic and distortion values are stored. It is read only when `POSE_ENABLED` is `True`. Changing it while pose is disabled has no effect. |
| `MARKER_DIAMETER_METERS` | `None` | Optional | Real outer-circle diameter in metres. It is used only for an approximate range estimate when pose is enabled. `None` means no metric marker-size estimate is attempted. |

## 9. Retired Hough-line settings

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `HOUGH_LINE_THRESHOLD` | `20` | Reserved / inactive | An old Hough Line Transform vote threshold. The project intentionally moved to contour fitting with `cv2.fitLine`, so this value currently does nothing. |
| `MIN_LINE_LENGTH` | `20` | Reserved / inactive | An old Hough Line Transform minimum line length in pixels. It is not used by the contour-based symbol detector. |
| `MAX_LINE_GAP` | `5` | Reserved / inactive | An old Hough Line Transform maximum gap between line fragments. It is not used by the contour-based symbol detector. |

## 10. Contour filtering and orientation settings

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `MIN_CONTOUR_AREA` | `40` | Active | Smallest accepted contour area in square pixels. Increasing it throws away more small blobs/noise but can remove thin or distant marker strokes. Lowering it keeps more possible evidence but also more noise and CPU work. |
| `ORIENTATION_CLUSTER_TOLERANCE_DEGREES` | `15.0` | Active | Maximum angle difference for contours to be treated as part of the same physical stroke direction. Increasing it combines noisier/slightly bent stroke edges but risks merging different directions. Lowering it separates directions more strictly but can split one worn stroke into several clusters. |

## 11. Ideal-symbol label settings

The detector accepts a generic geometric crossing even when it is neither a
perfect `X` nor a perfect `+`. These settings mainly affect the optional label
and score for near-ideal symbols.

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `SYMBOL_ANGLE_TOLERANCE_DEGREES` | `20.0` | Active | How far observed line directions may differ from the ideal X (`45/135` degrees) or plus (`0/90` degrees) templates. Increasing it labels more rotated/distorted patterns as X/+; lowering it is stricter and leaves more valid rotated targets labelled as `cross`. It does not remove the generic crossing test. |
| `MIN_SYMBOL_CONFIDENCE` | `0.60` | Active | Minimum template-based score needed to label a candidate specifically as X or +. Raising it means only clearer ideal symbols receive those labels. Lowering it assigns X/+ labels more freely. A valid generic `cross` can still be accepted through its separate crossing rules. |

## 12. Crossing-line validation settings

These values decide whether two fitted directions truly form a marker-like
crossing within the proposed circle.

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `MIN_CROSSING_ANGLE_DEGREES` | `20.0` | Active | Smallest allowed angle between the two crossing directions. Raising it rejects lines that are too close to parallel, reducing false crossings; lowering it allows shallow X-like crossings but makes nearly parallel clutter easier to accept. |
| `MAX_CROSS_CENTER_OFFSET_RATIO` | `0.95` | Active | Broad early crossing limit: the intersection must be inside the circle and no farther than 95% of one radius from its centre. This prevents two lines from crossing outside the marker. The stricter final-centre rule below is the main final decision. |
| `CROSS_SEGMENT_EXTENSION_RATIO` | `0.15` | Active | Extra allowance at each fitted contour segment end when deciding whether the lines meet. `0.15` means 15% of the segment length. It helps tolerate broken paint/edge gaps. Increasing it is more forgiving but can accept lines that only meet when extended too far. |
| `MIN_CROSS_CONFIDENCE` | `0.45` | Active | Minimum geometric crossing score required before a candidate can be considered a generic `cross`. Raising it reduces false crosses but can reject imperfect paint or difficult lighting. Lowering it is more forgiving but increases false positives. |

## 13. Final marker decision settings

The final decision combines circle quality, crossing quality, centrality, and
line support. It is the main gate before tracking and autonomy guidance.

| Setting | Current value | Status | Simple meaning and effect |
| --- | ---: | --- | --- |
| `MAX_INTERSECTION_CENTER_OFFSET_RATIO` | `0.45` | Active | Final strict limit for how far the line intersection may be from the circle centre, expressed as a fraction of radius. `0.45` means less than half a radius away. Lowering it demands a more centred cross and reduces false positives. Raising it accepts off-centre/worn markers but is less selective. |
| `MIN_LINE_SUPPORT_RATIO` | `0.30` | Active | Minimum support length for the weaker of the two stroke directions, divided by the circle diameter. Raising it requires longer, clearer strokes. Lowering it accepts shorter/partly hidden strokes but increases false crossings. |
| `MIN_MARKER_CONFIDENCE` | `0.65` | Active | Final combined confidence threshold required to set `is_marker = True`. Raising it is safer and stricter but can miss difficult valid markers. Lowering it detects more markers but lets more false candidates reach the tracker. This is the most important final acceptance threshold. |

## Safe tuning order

When a real marker is missed, inspect the debug windows first and tune in this
order:

1. Confirm camera resolution and lighting.
2. Tune `CANNY_LOW` and `CANNY_HIGH` until marker edges are visible but not
   covered in noise.
3. Tune `MIN_RADIUS`, `MAX_RADIUS`, and `HOUGH_PARAM2` until the outer circle
   is proposed reliably.
4. Tune `MIN_EDGE_DENSITY` if valid circles are rejected.
5. Tune `MIN_CONTOUR_AREA` and orientation tolerance if stroke contours are
   missing or fragmented.
6. Tune the final crossing/marker thresholds only after earlier stages look
   correct.

Do not compensate for a bad edge image by immediately lowering every final
confidence threshold. That usually creates false positives instead of fixing
the root cause.

## Settings that should not be used for flight control

None of these values command motors or define flight safety. The confidence
scores are deterministic image-quality heuristics, not probabilities. A
separate autonomy system must still enforce manual override, altitude limits,
velocity limits, timeouts, target-loss behaviour, and abort logic.
