"""
Stitch six Insta360 Pro origin videos into an approximate equirectangular MP4.

Usage:
    python stitch_insta360.py vids/VID_2026_04_20_14_11_39 --width 1920
"""
import argparse
from pathlib import Path

import cv2

from ..paths import ensure_output_dir, resolve_output_path
from ..viewer.pano_viewer import PanoViewer
from .stitcher import (
    Insta360Stitcher,
    open_origin_captures,
    read_frame_set,
    source_video_info,
)


def _make_writer(path, fps, size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {path}")
    return writer


def _default_output(folder):
    return ensure_output_dir(Path(folder).name) / "stitched_equirect.mp4"


def stitch_folder(args):
    stitcher = Insta360Stitcher(
        args.folder,
        output_width=args.width,
        calibration_block=args.calibration_block,
        edge_degrees=args.edge_degrees,
        blend_mode=args.blend_mode,
        feather_degrees=args.feather_degrees,
        interpolation=args.interpolation,
        rotation_order=args.rotation_order,
        inverse_rotation=not args.no_inverse_rotation,
    )
    captures = open_origin_captures(stitcher.origin_paths)
    source_fps, shortest_count, frame_counts = source_video_info(captures)
    fps = args.fps or source_fps
    output_path = resolve_output_path(args.output) if args.output else _default_output(args.folder)
    frame_limit = shortest_count
    if args.max_frames is not None:
        frame_limit = min(frame_limit, args.max_frames)

    print(f"Origin frame counts: {frame_counts}")
    print(f"Processing {frame_limit} frames at {fps:.3f} fps")
    print(f"Writing stitched panorama: {output_path}")

    pano_writer = _make_writer(
        output_path,
        fps,
        (stitcher.output_width, stitcher.output_height),
    )

    crystal_writer = None
    crystal_viewer = None
    if args.crystal_output:
        view_h, view_w = args.crystal_view_size
        crystal_output = resolve_output_path(args.crystal_output)
        crystal_writer = _make_writer(crystal_output, fps, (view_w, view_h))
        crystal_viewer = PanoViewer(
            out_h=view_h,
            out_w=view_w,
            fov=args.crystal_fov,
            roll=args.crystal_roll,
        )
        crystal_viewer.set_view(
            yaw=args.crystal_yaw,
            pitch=args.crystal_pitch,
            fov=args.crystal_fov,
            roll=args.crystal_roll,
        )
        print(f"Writing crystal-ball view: {crystal_output}")

    processed = 0
    try:
        while processed < frame_limit:
            frames = read_frame_set(captures)
            if frames is None:
                break

            panorama = stitcher.stitch(frames)
            pano_writer.write(panorama)
            if crystal_writer is not None:
                crystal_writer.write(crystal_viewer.render(panorama))

            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{frame_limit} frames")
    finally:
        for capture in captures:
            capture.release()
        pano_writer.release()
        if crystal_writer is not None:
            crystal_writer.release()

    print(f"Done. Wrote {processed} stitched frames to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approximate-stitch six Insta360 Pro origin videos."
    )
    parser.add_argument("folder", help="Folder containing origin_0.mp4..origin_5.mp4 and pro.prj.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output equirectangular MP4 path under output/ (default: output/folder/stitched_equirect.mp4).",
    )
    parser.add_argument("--width", type=int, default=3840, help="Output panorama width.")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for smoke tests.")
    parser.add_argument(
        "--calibration-block",
        default="pano_16_9",
        help="pro.prj offset block to use (default: pano_16_9).",
    )
    parser.add_argument(
        "--feather-degrees",
        type=float,
        default=6.0,
        help="Angular blend width for --blend-mode feather.",
    )
    parser.add_argument(
        "--edge-degrees",
        type=float,
        default=96.0,
        help="Maximum lens angle to sample before a pixel is considered uncovered.",
    )
    parser.add_argument(
        "--blend-mode",
        choices=("nearest", "feather"),
        default="nearest",
        help="nearest avoids ghosting; feather softens seams but can ghost.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("linear", "cubic", "lanczos"),
        default="lanczos",
        help="Sampling filter for lens remaps. Lanczos is sharpest but slower.",
    )
    parser.add_argument(
        "--rotation-order",
        choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"),
        default="zxy",
        help="Euler rotation order for pro.prj lens metadata.",
    )
    parser.add_argument(
        "--no-inverse-rotation",
        action="store_true",
        help="Use direct lens rotations instead of inverse world-to-lens rotations.",
    )
    parser.add_argument("--crystal-output", default=None, help="Optional crystal-ball MP4 path under output/.")
    parser.add_argument("--crystal-yaw", type=float, default=0.0, help="Crystal view yaw.")
    parser.add_argument("--crystal-pitch", type=float, default=0.0, help="Crystal view pitch.")
    parser.add_argument("--crystal-roll", type=float, default=-90.0, help="Crystal view roll.")
    parser.add_argument("--crystal-fov", type=float, default=90.0, help="Crystal view FOV.")
    parser.add_argument(
        "--crystal-view-size",
        type=int,
        nargs=2,
        default=(720, 1280),
        metavar=("H", "W"),
        help="Crystal output size as height width.",
    )
    return parser.parse_args()


def main():
    stitch_folder(parse_args())


if __name__ == "__main__":
    main()
