"""Unified CLI for live and offline 360 detection workflows."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from . import detection as _detection  # noqa: F401  (ensures subpackage import)
from .detection import live as live_detection
from .detection.video import analyze_video
from .detection.yolo import (
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    OBJECTNESS_THRESHOLD,
)
from .insta360.project import validate_origin_videos
from .insta360.stitch_cli import stitch_folder
from .paths import REPO_ROOT, ensure_output_dir, resolve_output_dir, resolve_output_path
from .viewer.crystal_ball import preview_video


def _absolute(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _has_origin_videos(path):
    if not path.is_dir():
        return False
    try:
        validate_origin_videos(path)
    except RuntimeError:
        return False
    return (path / "pro.prj").exists()


def _default_output_dir(video_path):
    if video_path.is_dir():
        return ensure_output_dir(video_path.name)
    return ensure_output_dir(video_path.stem)


def _stitch_input_folder(input_path, output_dir, args):
    stitched_path = output_dir / "stitched_equirect.mp4"
    stitch_args = SimpleNamespace(
        folder=str(input_path),
        output=str(stitched_path),
        width=args.stitch_width,
        fps=args.fps,
        max_frames=args.max_frames,
        calibration_block=args.calibration_block,
        feather_degrees=args.feather_degrees,
        edge_degrees=args.edge_degrees,
        blend_mode=args.blend_mode,
        interpolation=args.interpolation,
        rotation_order=args.rotation_order,
        no_inverse_rotation=args.no_inverse_rotation,
        crystal_output=None,
        crystal_yaw=args.preview_yaw,
        crystal_pitch=args.preview_pitch,
        crystal_roll=args.preview_roll,
        crystal_fov=args.preview_fov,
        crystal_view_size=args.preview_size,
    )
    stitch_folder(stitch_args)
    return stitched_path


def run_video(args):
    input_path = _absolute(args.video)
    if not input_path.exists():
        raise RuntimeError(f"Video path does not exist: {input_path}")

    output_dir = resolve_output_dir(args.output_dir) if args.output_dir else _default_output_dir(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _has_origin_videos(input_path):
        print(f"Stitching six-lens folder: {input_path}")
        yolo_input = _stitch_input_folder(input_path, output_dir, args)
    elif input_path.is_file():
        yolo_input = input_path
    else:
        preview = input_path / "preview.mp4"
        if preview.exists():
            yolo_input = preview
        else:
            raise RuntimeError(
                "Folder must contain origin_0.mp4..origin_5.mp4 plus pro.prj, "
                "or a preview.mp4 file."
            )

    annotated_path = output_dir / "annotated_equirect.mp4"
    print(f"Running YOLO scan: {yolo_input}")
    analyze_video(
        input_path=str(yolo_input),
        output_path=str(annotated_path),
        distance=args.distance,
        conf_threshold=args.conf_threshold,
        objectness_threshold=args.objectness_threshold,
        nms_threshold=args.nms_threshold,
    )

    if not args.no_preview:
        print(f"Opening crystal-ball preview: {annotated_path}")
        preview_video(
            path=str(annotated_path),
            view_size=tuple(args.preview_size),
            fov=args.preview_fov,
            yaw=args.preview_yaw,
            pitch=args.preview_pitch,
            roll=args.preview_roll,
            seek_seconds=args.seek_seconds,
        )

    print(f"Done. Outputs are in {output_dir}")


def run_live(args):
    live_args = [
        "live_detection",
        "--camera",
        args.live,
        "--distance",
        str(args.distance),
        "--width",
        str(args.live_width),
        "--view-size",
        str(args.preview_size[0]),
        str(args.preview_size[1]),
        "--fov",
        str(args.preview_fov),
        "--conf-threshold",
        str(args.conf_threshold),
        "--objectness-threshold",
        str(args.objectness_threshold),
        "--nms-threshold",
        str(args.nms_threshold),
    ]
    if args.output:
        live_output = resolve_output_path(args.output)
        live_output.parent.mkdir(parents=True, exist_ok=True)
        live_args.extend(["--output", str(live_output)])
    if args.no_preview:
        live_args.append("--no-display")

    old_argv = sys.argv
    try:
        sys.argv = live_args
        live_detection.main()
    finally:
        sys.argv = old_argv


def run_preview(args):
    preview_path = _absolute(args.preview)
    if not preview_path.exists():
        raise RuntimeError(f"Preview video does not exist: {preview_path}")

    preview_video(
        path=str(preview_path),
        view_size=tuple(args.preview_size),
        fov=args.preview_fov,
        yaw=args.preview_yaw,
        pitch=args.preview_pitch,
        roll=args.preview_roll,
        seek_seconds=args.seek_seconds,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline or live 360 YOLO detection.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Video file or Insta360 folder to process.")
    source.add_argument("--live", help="Live RTSP URL, video source, or camera index.")
    source.add_argument("--preview", help="Open a video directly in crystal-ball preview.")

    parser.add_argument("--output-dir", default=None, help="Offline output directory under output/.")
    parser.add_argument("--output", default=None, help="Live output video path under output/.")
    parser.add_argument("--no-preview", action="store_true", help="Do not open a preview window.")
    parser.add_argument("--max-frames", type=int, default=None, help="Offline smoke-test frame limit.")
    parser.add_argument("--fps", type=float, default=None, help="Override stitched output FPS.")
    parser.add_argument("--distance", type=float, default=1.0, help="Projection distance.")

    parser.add_argument("--stitch-width", type=int, default=1920, help="Offline stitched width.")
    parser.add_argument("--live-width", type=int, default=1280, help="Live processing width.")
    parser.add_argument("--calibration-block", default="pano_16_9", help="pro.prj calibration block.")
    parser.add_argument("--blend-mode", choices=("nearest", "feather"), default="nearest")
    parser.add_argument("--interpolation", choices=("linear", "cubic", "lanczos"), default="lanczos")
    parser.add_argument("--rotation-order", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"), default="zxy")
    parser.add_argument("--no-inverse-rotation", action="store_true")
    parser.add_argument("--edge-degrees", type=float, default=96.0)
    parser.add_argument("--feather-degrees", type=float, default=6.0)

    parser.add_argument("--preview-size", type=int, nargs=2, default=(720, 1280), metavar=("H", "W"))
    parser.add_argument("--preview-yaw", type=float, default=0.0)
    parser.add_argument("--preview-pitch", type=float, default=0.0)
    parser.add_argument("--preview-roll", type=float, default=-90.0)
    parser.add_argument("--preview-fov", type=float, default=90.0)
    parser.add_argument("--seek-seconds", type=float, default=2.0)

    parser.add_argument("--conf-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--objectness-threshold", type=float, default=OBJECTNESS_THRESHOLD)
    parser.add_argument("--nms-threshold", type=float, default=NMS_THRESHOLD)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.video:
        run_video(args)
    elif args.live:
        run_live(args)
    else:
        run_preview(args)
