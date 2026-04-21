"""
Live 360 video detection pipeline.

Captures an RTSP stream (or local video) from an Insta360 Pro camera,
projects each frame into 4 stereographic faces using precomputed remap maps,
runs YOLOv3 detection, and displays annotated results in real time.

Insta360 Pro connection:
    1. Connect to the camera's Wi-Fi hotspot (SSID: Insta360 Pro_XXXXXX)
    2. Camera IP: 192.168.1.188
    3. RTSP URLs to try (in order):
       - rtsp://192.168.1.188:8554/stitched
       - rtsp://192.168.1.188:8554/live_stream
       - rtsp://192.168.1.188:8554/origin_0  (single fisheye lens)

Usage:
    python live_detection.py --camera rtsp://192.168.1.188:8554/stitched
    python live_detection.py --camera example/test1.mp4
    python live_detection.py --camera 0  (USB webcam)
"""
import argparse
import os
import signal
import threading
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

import cv2

from ..projection.fast_stereo import FastPano2Stereo
from ..viewer.pano_viewer import PanoViewer
from .yolo import CONFIDENCE_THRESHOLD, NMS_THRESHOLD, OBJECTNESS_THRESHOLD, Yolo

DEFAULT_RTSP = "rtsp://192.168.1.188:8554/stitched"


class FrameReader:
    """Continuously reads frames from a VideoCapture in a background thread."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.lock:
                # Keep only the newest frame so slow detection does not build latency.
                self.latest_frame = frame

    def read(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


class DetectionWorker:
    """Runs projection + YOLO detection in a background thread."""

    def __init__(self, detector, projector):
        self.detector = detector
        self.projector = projector

        self.lock = threading.Lock()
        self.input_frame = None
        self.result_frame = None
        self.new_input = False
        self.running = True

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            # Wait for a new frame
            with self.lock:
                if not self.new_input:
                    frame = None
                else:
                    frame = self.input_frame
                    self.new_input = False

            if frame is None:
                time.sleep(0.001)
                continue

            # Projection is fast; YOLO is the slow step, so it runs off the UI thread.
            faces = self.projector.project(frame)
            annotated = self.detector.process_output(frame.copy(), faces)

            with self.lock:
                self.result_frame = annotated

    def submit(self, frame):
        with self.lock:
            self.input_frame = frame
            self.new_input = True

    def get_result(self):
        with self.lock:
            return self.result_frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=5.0)


def parse_source(camera_arg):
    """Convert CLI camera arg to a cv2.VideoCapture-compatible source."""
    try:
        return int(camera_arg)
    except ValueError:
        return camera_arg


def main():
    parser = argparse.ArgumentParser(
        description="Live 360 video detection with Insta360 Pro + YOLOv3")
    parser.add_argument("--camera", default=DEFAULT_RTSP,
                        help="RTSP URL, video file path, or device index (default: %(default)s)")
    parser.add_argument("--distance", "-d", type=float, default=1.0,
                        help="Stereographic projection parameter (default: 1.0)")
    parser.add_argument("--width", "-w", type=int, default=1280,
                        help="Resize frame width before processing (default: 1280)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional path to save annotated video (.mp4)")
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress OpenCV display window")
    parser.add_argument("--view-size", type=int, nargs=2, default=(720, 1280),
                        metavar=("H", "W"),
                        help="Rectilinear viewer output size (default: 720 1280)")
    parser.add_argument("--fov", type=float, default=90.0,
                        help="Initial horizontal FOV in degrees (default: 90)")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Minimum objectness*class confidence to keep a box (default: %(default)s)",
    )
    parser.add_argument(
        "--objectness-threshold",
        type=float,
        default=OBJECTNESS_THRESHOLD,
        help="Minimum YOLO objectness score to keep a box (default: %(default)s)",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=NMS_THRESHOLD,
        help="Panorama-space NMS threshold (default: %(default)s)",
    )
    args = parser.parse_args()

    source = parse_source(args.camera)
    print(f"Opening source: {source}")

    reader = FrameReader(source)

    # Wait for the first frame to determine dimensions
    first_frame = None
    for _ in range(100):
        first_frame = reader.read()
        if first_frame is not None:
            break
        time.sleep(0.05)

    if first_frame is None:
        print("Error: could not read any frames from the source.")
        reader.stop()
        return

    # Compute target dimensions
    orig_h, orig_w = first_frame.shape[:2]
    scale = args.width / orig_w
    target_w = args.width
    target_h = int(orig_h * scale)
    print(f"Frame size: {orig_w}x{orig_h} -> resize to {target_w}x{target_h}")

    # Projection maps are tied to the resized frame size, so build them after sizing.
    projector = FastPano2Stereo(target_h, target_w, distance=args.distance)
    detector = Yolo(
        conf_threshold=args.conf_threshold,
        objectness_threshold=args.objectness_threshold,
        nms_threshold=args.nms_threshold,
    )
    worker = DetectionWorker(detector, projector)

    # Interactive rectilinear viewer (crystal-ball navigation)
    view_h, view_w = args.view_size
    viewer = PanoViewer(out_h=view_h, out_w=view_w, fov=args.fov)
    window_name = "Live 360 Detection"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, viewer.on_mouse)

    # Optional video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 15.0, (view_w, view_h))
        if not writer.isOpened():
            print(f"Warning: could not open output video writer for {args.output}")
            writer = None

    # Graceful shutdown on Ctrl+C
    shutdown = threading.Event()

    def handle_signal(sig, frame):
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)

    frame_count = 0
    t_start = time.time()

    print("Running live detection. Press 'q' to quit.")

    while not shutdown.is_set():
        raw = reader.read()
        if raw is None:
            if not reader.running:
                print("End of video stream.")
                break
            time.sleep(0.01)
            continue

        # Resize for faster processing
        frame = cv2.resize(raw, (target_w, target_h))

        # Submit to detection worker
        worker.submit(frame)

        # Get latest detection result (annotated equirect) and render viewer
        result = worker.get_result()
        display_source = result if result is not None else frame
        rect_view = viewer.render(display_source)

        if result is not None:
            if writer:
                writer.write(rect_view)
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  [{frame_count} detections] {fps:.1f} det/s")

        if not args.no_display:
            cv2.imshow(window_name, rect_view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            viewer.handle_key(key)

    # Cleanup
    worker.stop()
    reader.stop()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Done. {frame_count} detections in {elapsed:.1f}s ({fps:.1f} det/s)")


if __name__ == "__main__":
    main()
