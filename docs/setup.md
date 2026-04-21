# Setup and Assets

## Python Environment

Run commands from the repository root unless noted.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required YOLO Files

`src.detection.yolo.Yolo` loads the following from [assets/](../assets/) via `src.paths.ASSETS_DIR`, so commands work from any CWD:

- `assets/yolov3.cfg`
- `assets/yolov3.weights` (symlink to a local weights file — verify `readlink assets/yolov3.weights` resolves)
- `assets/coco.names`

## Insta360 Pro Live Input

For the live camera pipeline:

1. Power on the camera and wait until it finishes booting.
2. Connect the machine to the camera Wi-Fi hotspot.
3. Confirm the camera is reachable, usually at `192.168.1.188`.
4. Put the camera into live-streaming mode so the stitched RTSP stream is available.
5. Try these URLs in order:

```text
rtsp://192.168.1.188:8554/stitched
rtsp://192.168.1.188:8554/live_stream
rtsp://192.168.1.188:8554/origin_0
```

`origin_0` is a single raw lens stream, not a stitched panorama, so it is mainly useful for connectivity checks.

## Dependencies

The runtime dependencies are:

- OpenCV (`opencv-python`)
- NumPy
- SciPy

OpenCV DNN runs YOLOv3 on CPU in the current implementation.

