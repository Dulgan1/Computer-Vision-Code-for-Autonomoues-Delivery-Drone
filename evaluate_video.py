"""Evaluate a recorded video with the production DroneCV pipeline.

Example:
    venv/bin/python evaluate_video.py videos/marker_test.mp4 --csv evaluation.csv
"""

import argparse
import csv
from statistics import mean
from time import perf_counter

import cv2

from pipeline import LandingMarkerPipeline


CSV_FIELDS = (
    "frame",
    "timestamp_ms",
    "candidate_count",
    "validated_count",
    "track_id",
    "track_stable",
    "track_visible",
    "marker_confidence",
    "horizontal_error",
    "vertical_error",
    "processing_ms",
)


def evaluation_row(frame_number, capture, result, processing_ms):
    """Flatten one pipeline result into a CSV-friendly evaluation record."""

    target = result["tracked_target"]
    guidance = result["guidance"]
    validated_count = sum(item["is_marker"] for item in result["candidates"])

    return {
        "frame": frame_number,
        "timestamp_ms": round(capture.get(cv2.CAP_PROP_POS_MSEC), 3),
        "candidate_count": len(result["candidates"]),
        "validated_count": validated_count,
        "track_id": target["track_id"] if target else "",
        "track_stable": target["is_stable"] if target else False,
        "track_visible": target["visible"] if target else False,
        "marker_confidence": (
            round(target["marker_confidence"], 4) if target else ""
        ),
        "horizontal_error": (
            round(guidance["horizontal_error"], 4) if guidance else ""
        ),
        "vertical_error": round(guidance["vertical_error"], 4) if guidance else "",
        "processing_ms": round(processing_ms, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to an input video file")
    parser.add_argument("--csv", default="video_evaluation.csv", help="CSV output path")
    parser.add_argument("--max-frames", type=int, default=None)
    arguments = parser.parse_args()

    capture = cv2.VideoCapture(arguments.video)
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {arguments.video}")

    pipeline = LandingMarkerPipeline()
    processing_times = []
    frame_number = 0
    validated_frames = 0
    stable_frames = 0

    with open(arguments.csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while arguments.max_frames is None or frame_number < arguments.max_frames:
            success, frame = capture.read()
            if not success:
                break

            start = perf_counter()
            result = pipeline.process(frame)
            processing_ms = 1000.0 * (perf_counter() - start)
            processing_times.append(processing_ms)

            row = evaluation_row(frame_number, capture, result, processing_ms)
            writer.writerow(row)
            validated_frames += row["validated_count"] > 0
            stable_frames += row["track_stable"] and row["track_visible"]
            frame_number += 1

    capture.release()

    if not processing_times:
        raise SystemExit("Video contained no readable frames.")

    ordered_times = sorted(processing_times)
    percentile_index = int(0.95 * (len(ordered_times) - 1))
    print(f"Frames processed: {frame_number}")
    print(f"Frames with validated marker: {validated_frames}")
    print(f"Frames with visible stable track: {stable_frames}")
    print(f"Mean processing time: {mean(processing_times):.2f} ms/frame")
    print(f"95th percentile processing time: {ordered_times[percentile_index]:.2f} ms/frame")
    print(f"CSV report: {arguments.csv}")


if __name__ == "__main__":
    main()
