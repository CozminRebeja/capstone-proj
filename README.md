# Capstone 360 Object Detection

This project detects objects in 360-degree equirectangular panoramas and video.
It projects a panorama into stereographic views, runs YOLOv3 with OpenCV DNN,
maps detections back into panorama coordinates, applies non-maximum suppression,
and renders either an annotated equirectangular output or an interactive
rectilinear "crystal ball" preview.

## Project Layout

```text
src/
  __main__.py              # `python -m src` entrypoint
  cli.py                   # unified offline/live/preview CLI
  paths.py                 # repo-root anchored assets/data paths
  detection/
    yolo.py                # YOLOv3 wrapper, box filtering, NMS, drawing
    video.py               # offline equirectangular video detection
    live.py                # threaded live RTSP/video detection pipeline
  projection/
    stereo.py              # reference equirectangular <-> stereographic math
    fast_stereo.py         # precomputed cv2.remap projection for live/video use
  viewer/
    pano_viewer.py         # interactive rectilinear panorama renderer
    crystal_ball.py        # interactive video previewer
    render.py              # batch crystal-ball renderer
  insta360/
    project.py             # pro.prj parser and validation helpers
    stitcher.py            # six-lens Insta360 Pro stitcher
    stitch_cli.py          # stitcher CLI
assets/
  yolov3.cfg
  yolov3.weights           # local weights file or symlink
  coco.names
```

The code resolves assets relative to the repository root through `src.paths`, so
commands can be run from any working directory. The examples below assume you
are in the repository root.

# Usage

## 1. Create the Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the YOLO assets before running detection:

```bash
ls -l assets/yolov3.cfg assets/yolov3.weights assets/coco.names
```

## 2. Unified CLI

Process an equirectangular video file:

```bash
python -m src --video data/videos/preview.mp4 --output-dir data/processed
```

Process an Insta360 Pro folder containing `origin_0.mp4` through
`origin_5.mp4` plus `pro.prj`. The CLI stitches the raw lens videos first, then
runs YOLO on the stitched panorama:

```bash
python -m src --video data/insta360/VID_2026_04_20_14_11_39 --output-dir data/processed
```

Limit work for a smoke test:

```bash
python -m src --video data/insta360/VID_2026_04_20_14_11_39 --output-dir data/processed --max-frames 30 --no-preview
```

Run live detection from an Insta360 Pro stitched RTSP stream:

```bash
python -m src --live rtsp://192.168.1.188:8554/stitched
```

Run live detection from a local video or webcam:

```bash
python -m src --live data/videos/preview.mp4
python -m src --live 0
```

Record the live rectilinear output and suppress the display window:

```bash
python -m src --live rtsp://192.168.1.188:8554/stitched --output data/result_live.mp4 --no-preview
```

Open an existing equirectangular video in the interactive crystal-ball previewer:

```bash
python -m src --preview data/processed/annotated_equirect.mp4
```

Useful shared options:

```bash
python -m src --video data/videos/preview.mp4 \
  --output-dir data/processed \
  --distance 1.0 \
  --conf-threshold 0.45 \
  --objectness-threshold 0.55 \
  --nms-threshold 0.25 \
  --preview-size 720 1280 \
  --preview-fov 90
```

## 3. Direct Module Commands

Run YOLO on a single panorama image:

```bash
python -m src.detection.yolo data/images/input.jpg data/images/output.jpg
```

Run YOLO on every frame of an equirectangular video:

```bash
python -m src.detection.video --input data/videos/preview.mp4 --output data/result.mp4
```

Run the live detection pipeline directly:

```bash
python -m src.detection.live --camera rtsp://192.168.1.188:8554/stitched
```

Stitch a six-lens Insta360 Pro folder into an equirectangular MP4:

```bash
python -m src.insta360.stitch_cli data/insta360/VID_2026_04_20_14_11_39 \
  --output data/processed/stitched_equirect.mp4 \
  --width 3840
```

Stitch only a short sample and also write a crystal-ball render:

```bash
python -m src.insta360.stitch_cli data/insta360/VID_2026_04_20_14_11_39 \
  --output data/processed/stitched_equirect.mp4 \
  --crystal-output data/processed/stitched_crystal_ball.mp4 \
  --max-frames 30
```

Preview an equirectangular video interactively:

```bash
python -m src.viewer.crystal_ball data/processed/annotated_equirect.mp4
```

Batch-render panorama videos into rectilinear crystal-ball videos:

```bash
python -m src.viewer.render data/processed/annotated_equirect.mp4 --output-dir data/processed/crystal_ball
```

Run projection smoke tests:

```bash
python -m src.projection.stereo --p2s data/images/input.jpg --d 1.0
python -m src.projection.stereo --s2p data/images/face.jpg --output data/images/pano_output.jpg
```

## 4. Viewer Controls

The live and preview windows use the same interaction model:

- Drag with the mouse to rotate the rectilinear view.
- Use the mouse wheel or `+` / `-` to zoom.
- Use `w`, `a`, `s`, `d` or arrow keys to pan.
- Press `r` to reset the view.
- Press `q` to quit.

The crystal-ball previewer also supports:

- Space: pause or play.
- Left/right arrows: seek.
- `s`: save the current rendered view as a JPEG.

## Output Files

The unified offline CLI writes to the chosen `--output-dir`:

- `stitched_equirect.mp4` when the input is a six-lens Insta360 folder.
- `annotated_equirect.mp4` after YOLO detection.

Live mode writes a rectilinear preview video only when `--output` is provided.
