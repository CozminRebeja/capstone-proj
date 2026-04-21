"""
Interactive OpenCV previewer for stitched/equirectangular panorama videos.

Controls:
    mouse drag: rotate view
    mouse wheel or +/-: zoom
    space: pause/play
    left/right arrows: seek
    s: save current view frame
    r: reset view
    q: quit
"""
import argparse
from pathlib import Path

import cv2

from .pano_viewer import PanoViewer


LEFT_KEYS = {81, 2, 2424832}
RIGHT_KEYS = {83, 3, 2555904}


def _source_fps(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 30.0


def _seek(capture, frame_index):
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        frame_index = max(0, min(frame_count - 1, frame_index))
    else:
        frame_index = max(0, frame_index)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    return frame_index


def _save_frame(frame, source_path, frame_index):
    output = source_path.with_name(f"{source_path.stem}_view_{frame_index:06d}.jpg")
    cv2.imwrite(str(output), frame)
    print(f"Saved {output}")


def preview_video(path, view_size, fov, yaw, pitch, roll, seek_seconds):
    source_path = Path(path)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    view_h, view_w = view_size
    viewer = PanoViewer(out_h=view_h, out_w=view_w, fov=fov, roll=roll)
    viewer.set_view(yaw=yaw, pitch=pitch, fov=fov, roll=roll)

    fps = _source_fps(capture)
    delay_ms = max(1, int(1000 / fps))
    seek_frames = max(1, int(fps * seek_seconds))
    paused = False
    frame = None
    frame_index = 0
    rendered = None
    window_name = "Crystal Ball Previewer"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, viewer.on_mouse)

    try:
        while True:
            if not paused or frame is None:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                rendered = viewer.render(frame)
            elif rendered is not None:
                rendered = viewer.render(frame)

            cv2.imshow(window_name, rendered)
            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFFFFFFFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                viewer.set_view(yaw=0, pitch=0, fov=90, roll=roll)
            elif key == ord("s") and rendered is not None:
                _save_frame(rendered, source_path, frame_index)
            elif key in LEFT_KEYS:
                frame_index = _seek(capture, frame_index - seek_frames)
                frame = None
            elif key in RIGHT_KEYS:
                frame_index = _seek(capture, frame_index + seek_frames)
                frame = None
            else:
                viewer.handle_key(key & 0xFF)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview an equirectangular video as a crystal-ball view."
    )
    parser.add_argument("video", help="Stitched/equirectangular video path.")
    parser.add_argument(
        "--view-size",
        type=int,
        nargs=2,
        default=(720, 1280),
        metavar=("H", "W"),
        help="Preview size as height width.",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw.")
    parser.add_argument("--pitch", type=float, default=0.0, help="Initial pitch.")
    parser.add_argument("--roll", type=float, default=-90.0, help="Initial roll.")
    parser.add_argument("--fov", type=float, default=90.0, help="Initial FOV.")
    parser.add_argument("--seek-seconds", type=float, default=2.0, help="Arrow seek step.")
    return parser.parse_args()


def main():
    args = parse_args()
    preview_video(
        path=args.video,
        view_size=tuple(args.view_size),
        fov=args.fov,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        seek_seconds=args.seek_seconds,
    )


if __name__ == "__main__":
    main()
