"""
Render saved equirectangular panorama videos into a rectilinear view.

For Insta360 Pro folders, this script uses preview.mp4 by default. The
origin_*.mp4 files are raw lens videos and must be stitched before this viewer
can render them as a panorama.
"""
import argparse
from pathlib import Path

import cv2

from ..paths import ensure_output_dir, resolve_output_dir
from .pano_viewer import PanoViewer


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}


def is_equirectangular_video(path):
    """Return True when the first video frame has an approximate 2:1 ratio."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return False

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()

    if width <= 0 or height <= 0:
        return False

    ratio = width / height
    return 1.9 <= ratio <= 2.1


def discover_inputs(input_path):
    """Find panorama videos from a file or Insta360 export folder."""
    path = Path(input_path)
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise RuntimeError(f"Input path does not exist: {path}")

    preview = path / "preview.mp4"
    if preview.exists():
        return [preview]

    videos = []
    for candidate in sorted(path.iterdir()):
        if candidate.suffix.lower() in VIDEO_EXTENSIONS and is_equirectangular_video(candidate):
            videos.append(candidate)

    if not videos:
        raise RuntimeError(
            "No equirectangular panorama videos found. Stitch origin_*.mp4 first."
        )

    return videos


def output_path_for(input_video, output_dir):
    stem = input_video.stem
    return output_dir / f"{stem}_crystal_ball.mp4"


def render_video(input_video, output_video, view_size, yaw, pitch, roll, fov, fps_override):
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps_override or source_fps or 30.0
    fps = fps if fps > 0 else 30.0

    view_h, view_w = view_size
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (view_w, view_h),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {output_video}")

    viewer = PanoViewer(out_h=view_h, out_w=view_w, fov=fov, roll=roll)
    viewer.set_view(yaw=yaw, pitch=pitch, fov=fov, roll=roll)

    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            writer.write(viewer.render(frame))
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames from {input_video.name}")
    finally:
        capture.release()
        writer.release()

    print(f"Wrote {frame_count} frames to {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render saved 360 panorama video into a crystal-ball view."
    )
    parser.add_argument(
        "input",
        help="Path to a panorama video or an Insta360 folder containing preview.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for rendered videos (default: output/crystal_ball).",
    )
    parser.add_argument(
        "--view-size",
        type=int,
        nargs=2,
        default=(720, 1280),
        metavar=("H", "W"),
        help="Output view size as height width (default: 720 1280).",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="View yaw in degrees.")
    parser.add_argument("--pitch", type=float, default=0.0, help="View pitch in degrees.")
    parser.add_argument("--roll", type=float, default=0.0, help="View roll in degrees.")
    parser.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV in degrees.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Defaults to the source video FPS.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = discover_inputs(args.input)

    output_dir = resolve_output_dir(args.output_dir) if args.output_dir else ensure_output_dir("crystal_ball")
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_video in inputs:
        render_video(
            input_video=input_video,
            output_video=output_path_for(input_video, output_dir),
            view_size=tuple(args.view_size),
            yaw=args.yaw,
            pitch=args.pitch,
            roll=args.roll,
            fov=args.fov,
            fps_override=args.fps,
        )


if __name__ == "__main__":
    main()
