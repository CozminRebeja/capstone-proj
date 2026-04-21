# Usage

## Still Image Detection

```bash
cd src
python detection.py input.jpg output.jpg
```

Optional thresholds:

```bash
python detection.py input.jpg output.jpg \
  --conf-threshold 0.45 \
  --objectness-threshold 0.55 \
  --nms-threshold 0.25
```

## Offline Video Detection

```bash
cd src
python video_detection.py \
  --input vids/preview.mp4 \
  --output result.mp4
```

Useful options:

- `--distance`: stereographic projection parameter, default `1.0`.
- `--conf-threshold`: minimum `objectness * class_score`.
- `--objectness-threshold`: minimum YOLO objectness.
- `--nms-threshold`: panorama-space NMS IoU threshold.

## Live Detection

```bash
cd src
python live_detection.py --camera rtsp://192.168.1.188:8554/stitched
```

Local file or webcam examples:

```bash
python live_detection.py --camera vids/preview.mp4
python live_detection.py --camera 0
```

Headless or output recording:

```bash
python live_detection.py \
  --camera rtsp://192.168.1.188:8554/stitched \
  --no-display \
  --output result.mp4
```

## Interactive Viewer Controls

When `live_detection.py` opens the `Live 360 Detection` window:

- Left-drag: pan the rectilinear view.
- Scroll wheel: zoom by changing field of view.
- `W/A/S/D` or arrow keys: pan.
- `+` / `-`: zoom.
- `0`: reset view to yaw `0`, pitch `0`, FOV `90`.
- `q`: quit.

## Projection Tests

Reference conversion:

```bash
cd src
python stereo.py --p2s input.jpg --d 1.0
```

Reverse projection:

```bash
python stereo.py --s2p face.jpg --output output.jpg
```

