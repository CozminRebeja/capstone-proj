
# Capstone Codebase Documentation

This project detects objects in 360-degree equirectangular panorama images and video. The core pipeline splits a panorama into four stereographic views, runs YOLOv3 detection on those views, maps detections back into panorama coordinates, suppresses duplicate boxes, and renders the annotated panorama or an interactive rectilinear view.

## Documentation Map

- [Architecture](architecture.md): how the modules fit together and how data moves through the system.
- [Setup and Assets](setup.md): Python environment, model files, sample assets, and camera setup notes.
- [Usage](usage.md): commands for still images, videos, live streams, and the panorama viewer.
- [Module Reference](modules.md): file-by-file responsibilities and important functions/classes.
- [Troubleshooting](troubleshooting.md): common runtime issues and practical fixes.

## Project Layout

```text
src/
  detection.py        # YOLOv3 wrapper, detection filtering, NMS, box drawing
  stereo.py           # reference panorama <-> stereographic projection math
  fast_stereo.py      # faster cv2.remap-based projection for live/video use
  live_detection.py   # threaded live RTSP/video detection pipeline
  video_detection.py  # offline frame-by-frame video processing
  pano_viewer.py      # interactive rectilinear panorama viewer
  requirements.txt    # Python dependencies
```

Generated media and sample inputs live under `src/vids/`, `src/image/`, and `src/result.mp4` when present.

## High-Level Flow

```text
Panorama image/video frame
        |
        v
Stereographic projection into 4 square faces
        |
        v
YOLOv3 detection on each face
        |
        v
Realign face detections back into panorama coordinates
        |
        v
Panorama-space confidence filtering + NMS
        |
        v
Annotated panorama or rectilinear viewer output
```
