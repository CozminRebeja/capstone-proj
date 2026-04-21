# Architecture

## Pipeline Overview

The project works around a key limitation of standard object detectors: YOLOv3 expects perspective-like images, while an equirectangular panorama has heavy distortion near the poles and wraps horizontally. To make detections more stable, the panorama is projected into four stereographic faces before inference.

1. **Capture or load a panorama**
   - `detection.py` reads one still image.
   - `video_detection.py` reads a video file frame by frame.
   - `live_detection.py` reads an RTSP stream, local video, or webcam.

2. **Project the panorama**
   - `stereo.pano2stereo()` is the reference implementation. It rebuilds interpolation splines every call, so it is useful for correctness and offline tests.
   - `fast_stereo.FastPano2Stereo` precomputes OpenCV remap tables once per frame size and projection distance. This is the live path.

3. **Detect objects**
   - `detection.Yolo.detect()` prepares an OpenCV DNN blob, runs YOLOv3, and returns raw detections.
   - `Yolo.process_output()` runs detections on all four faces.

4. **Map boxes back to panorama space**
   - `stereo.realign_bbox()` converts normalized face-space bounding boxes back into normalized panorama-space center/width/height values.
   - This step accounts for the 90-degree face offsets used when producing the four stereographic projections.

5. **Filter and merge detections**
   - `Yolo.nms_selection()` removes low-objectness detections, combines objectness with class confidence, clamps boxes to valid image bounds, drops tiny boxes, and runs OpenCV NMS.

6. **Render results**
   - `Yolo.draw_bbox()` draws labels and boxes on the panorama frame.
   - `pano_viewer.PanoViewer` can render an interactive rectilinear view from the annotated equirectangular frame.

## Runtime Paths

### Still Image

```text
detection.main()
  -> cv2.imread()
  -> stereo.pano2stereo()
  -> Yolo.process_output()
  -> cv2.imwrite()
```

### Offline Video

```text
video_detection.analyze_video()
  -> cv2.VideoCapture()
  -> for each frame:
       stereo.pano2stereo()
       Yolo.process_output()
       VideoWriter.write()
```

### Live Stream

```text
FrameReader thread:
  VideoCapture.read() -> latest_frame

DetectionWorker thread:
  latest_frame -> FastPano2Stereo.project() -> Yolo.process_output() -> result_frame

Main thread:
  result_frame or latest raw frame -> PanoViewer.render() -> imshow / writer
```

The live pipeline intentionally keeps only the latest frame. That reduces latency when YOLO inference is slower than the camera frame rate.

## Coordinate Systems

- **Equirectangular panorama:** full input frame. Horizontal pixels map to longitude, vertical pixels map to latitude.
- **Stereographic face:** square projection used for detection. The project produces four faces, each covering a 90-degree horizontal sector.
- **YOLO output:** normalized face-space values in `(center_x, center_y, width, height)` format.
- **Final boxes:** pixel-space boxes on the original panorama after realignment and NMS.

