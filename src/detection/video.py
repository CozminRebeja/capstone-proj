"""
Panorama video analysis script.

Usage:
    $ python3 video_detection.py --input pano.mp4 --output result.mp4

This script reuses the existing panorama-to-stereo projection (`pano2stereo`)
and YOLO detector (`Yolo`) to annotate each frame of a panoramic video.
"""
import argparse
import cv2

from ..projection.stereo import pano2stereo
from .yolo import (
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    OBJECTNESS_THRESHOLD,
    Yolo,
)


def analyze_video(
    input_path,
    output_path,
    distance,
    conf_threshold,
    objectness_threshold,
    nms_threshold,
):
    """
    Run panorama detection on every frame of the video.

    Args:
        input_path (str): Path to the panoramic source video.
        output_path (str): Path for the annotated video.
        distance (float): Projection parameter passed to `pano2stereo`.
        conf_threshold (float): Minimum objectness*class confidence to keep.
        objectness_threshold (float): Minimum YOLO objectness score to keep.
        nms_threshold (float): Panorama-space NMS threshold.
    """
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    detector = Yolo(
        conf_threshold=conf_threshold,
        objectness_threshold=objectness_threshold,
        nms_threshold=nms_threshold,
    )
    frame_idx = 0

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            projections = pano2stereo(frame, distance)
            annotated = detector.process_output(frame.copy(), projections)
            writer.write(annotated)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx} frames")
    finally:
        capture.release()
        writer.release()

    print(f"Done. Wrote {frame_idx} frames to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects frame-by-frame on panorama videos.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input panorama video.")
    parser.add_argument("--output", "-o", required=True, help="Path to the output video with boxes.")
    parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=1.0,
        help="Projection parameter d passed to pano2stereo (default 1.0).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Minimum objectness*class confidence to keep a box (default: %(default)s).",
    )
    parser.add_argument(
        "--objectness-threshold",
        type=float,
        default=OBJECTNESS_THRESHOLD,
        help="Minimum YOLO objectness score to keep a box (default: %(default)s).",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=NMS_THRESHOLD,
        help="Panorama-space NMS threshold (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_video(
        args.input,
        args.output,
        args.distance,
        args.conf_threshold,
        args.objectness_threshold,
        args.nms_threshold,
    )


if __name__ == "__main__":
    main()
