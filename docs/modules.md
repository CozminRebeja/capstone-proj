# Module Reference

## `src/detection.py`

Wraps YOLOv3 through OpenCV DNN.

Important pieces:

- `Yolo.__init__()`: loads `yolov3.cfg`, `yolov3.weights`, and `coco.names`.
- `Yolo.detect(frame)`: converts an image to a DNN blob and runs the YOLO output layers.
- `Yolo.nms_selection(frame, output)`: converts raw detections into filtered panorama-space boxes and runs NMS.
- `Yolo.process_output(input_img, frames)`: detects across the four stereographic faces, realigns boxes, and draws final boxes.

The class currently uses CPU inference via `cv2.dnn.DNN_TARGET_CPU`.

## `src/stereo.py`

Contains the reference projection math.

Important pieces:

- `projection_angle(x, d)`: converts stereographic plane coordinates into angular coordinates.
- `pano2stereo(pic, distance=1.0)`: converts one equirectangular panorama into four square stereographic faces.
- `stereo2pano(in_pic)`: converts one stereographic image back into panorama form.
- `realign_bbox(center_x, center_y, width, height, face)`: maps normalized YOLO boxes from face space back into normalized panorama space.
- `merge_stereo(stereos)`: experimental merge path for recombining four stereographic faces.

`pano2stereo()` uses SciPy spline interpolation. It is readable, but slower than `FastPano2Stereo` for repeated frames.

## `src/fast_stereo.py`

Optimized projection for video and live use.

`FastPano2Stereo` precomputes four `cv2.remap` lookup table pairs at initialization. The maps depend only on frame dimensions and `distance`, so each later frame can be projected without rebuilding interpolation data.

## `src/live_detection.py`

Live capture, detection, and display entrypoint.

Important pieces:

- `FrameReader`: continuously reads frames into `latest_frame` on a background thread.
- `DetectionWorker`: runs projection and YOLO inference on the latest submitted frame.
- `parse_source()`: converts numeric camera CLI arguments into webcam indices.
- `main()`: wires capture, resizing, projection, detection, viewer rendering, optional recording, and shutdown.

## `src/video_detection.py`

Offline video entrypoint. It processes every frame from an input video and writes a full annotated output video.

This path currently uses `stereo.pano2stereo()`, which prioritizes the reference implementation over live speed.

## `src/pano_viewer.py`

Interactive viewer for equirectangular frames.

`PanoViewer` builds OpenCV remap tables that turn an equirectangular frame into a rectilinear camera view. It rebuilds maps only when the source frame shape or view parameters change.

