**Ready for review**Select text to add comments on the plan

# Plan: Live 360° Video Analyzer — Insta360 Pro + YOLO

## Context

The project already has a working offline pipeline in `360yolo/`:

* `stereo.py` — equirectangular → 4 stereographic faces (uses `scipy.RectBivariateSpline`)
* `detection.py` — `Yolo` class wrapping YOLOv3 via OpenCV DNN
* `video_detection.py` — processes pre-recorded MP4 files frame-by-frame

**Problem:** This pipeline is far too slow for live use:

1. `pano2stereo` rebuilds `RectBivariateSpline` objects **every frame** — costs several seconds per frame.
2. There is no camera capture layer — only file-based `cv2.VideoCapture`.
3. `pano2stereo` has a side-effect bug: writes `face_N_d.jpg` files to disk on every call.

The Insta360 Pro records 6 fisheye lenses (`origin_0..5.mp4`, 1920×1440 each) and produces a stitched equirectangular output. In live mode the camera hosts a Wi-Fi hotspot (`192.168.1.188`) and exposes an RTSP stream of the stitched equirectangular video.

**Goal:** Add a live RTSP capture + fast projection + YOLOv3 detection pipeline with real-time display.

---

## Critical Files

| File                         | Role                                                                   |
| ---------------------------- | ---------------------------------------------------------------------- |
| `360yolo/stereo.py:41-87`  | `pano2stereo` — has disk side-effect, slow per-frame spline rebuild |
| `360yolo/detection.py`     | `Yolo` class — reuse as-is                                          |
| `360yolo/requirements.txt` | Add no new deps (cv2.remap already in OpenCV)                          |

---

## Implementation Steps

### Step 1 — Fix `stereo.py` side-effect bug

**File:** `360yolo/stereo.py`

Remove the `cv2.imwrite(...)` call at line 82 inside `pano2stereo`. It writes `face_N_d.jpg` to disk on every frame — unusable in live mode and pollutes the working directory.

```python
# DELETE this line inside pano2stereo loop:
cv2.imwrite('face_'+str(face)+'_'+str(d)+'.jpg', output_img)
```

---

### Step 2 — Create `fast_stereo.py` (precomputed remap maps)

**New file:** `360yolo/fast_stereo.py`

`RectBivariateSpline` reconstruction per frame is the main bottleneck. The mapping is purely geometric — it depends only on `(height, width, distance)`, not on pixel values. Replace with `cv2.remap()` using maps precomputed once at init.

```
class FastPano2Stereo:
    __init__(height, width, distance=1.0)
        → compute self.maps: list of 4 × (map_x: float32 H×H, map_y: float32 H×H)
      
        For face k (0-3):
          • pano_x[j] = width/2 + projection_angle(xp_domain[j], d) / delta_rad
          • pano_x shifted by k*(width/4) mod width  (replaces the image-rotation trick)
          • pano_y[i] = height/2 + projection_angle(yp_domain[i], d) / delta_rad
          • map_x[i, j] = pano_x[j]   (broadcast to 2D H×H)
          • map_y[i, j] = pano_y[i]   (broadcast to 2D H×H)
          → store (map_x.astype(float32), map_y.astype(float32))
  
    project(frame) → list of 4 faces
        for each (map_x, map_y): cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
```

Reuse `projection_angle` from `stereo.py` (import it).

Expected speedup: from ~3-5 seconds/frame to ~30-100 ms/frame for map application.

---

### Step 3 — Create `live_detection.py`

**New file:** `360yolo/live_detection.py`

Architecture uses two threads to decouple slow YOLO inference from fast camera capture:

```
Thread A (reader):   cap.read() in tight loop → shared latest_frame (lock)
Thread B (detector): reads latest_frame → project → detect → shared latest_result (lock)
Main thread:         reads latest_result → cv2.imshow / VideoWriter → waitKey(1)
```

This ensures the camera buffer never overflows and the display always shows the most recent annotated frame, even if YOLO is slower than the stream's FPS.

**CLI:**

```
python live_detection.py [OPTIONS]
  --camera URL   RTSP URL or device index (default: rtsp://192.168.1.188:8554/stitched)
  --distance D   Projection parameter (default: 1.0)
  --width W      Resize frame to W pixels wide before processing (default: 1280)
  --output PATH  Optional path to save annotated video (.mp4)
  --no-display   Suppress OpenCV window (for headless/server use)
```

Key implementation details:

* `cv2.VideoCapture(url, cv2.CAP_FFMPEG)` for RTSP, or integer for USB webcam
* Set `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` to minimize latency
* Frame resize before projection to reduce computational load
* FPS stats printed every 30 annotated frames (detection throughput, not capture FPS)
* Graceful shutdown on `q` key or SIGINT

**Insta360 Pro connection note** (in docstring):

* Connect host to camera Wi-Fi hotspot (default SSID: `Insta360 Pro_XXXXXX`, IP: `192.168.1.188`)
* Try these RTSP URLs in order:
  * `rtsp://192.168.1.188:8554/stitched`
  * `rtsp://192.168.1.188:8554/live_stream`
  * For raw fisheye feed: `rtsp://192.168.1.188:8554/origin_0` (one lens)

---

### Step 4 — Update `requirements.txt`

No new runtime deps needed. Add optional comment about YOLOv8:

```
# Optional: pip install ultralytics  (for YOLOv8 upgrade)
```

---

## File Change Summary

| File                          | Action                                                                 |
| ----------------------------- | ---------------------------------------------------------------------- |
| `360yolo/stereo.py`         | Remove `cv2.imwrite` side-effect at line 82                          |
| `360yolo/fast_stereo.py`    | **Create** — precomputed remap maps + `FastPano2Stereo` class |
| `360yolo/live_detection.py` | **Create** — threaded RTSP capture + live display               |
| `360yolo/requirements.txt`  | Minor comment addition                                                 |

---

## Verification

1. **Offline test first** — run `FastPano2Stereo` on a still image and compare output with original `pano2stereo` output to confirm projections match.
2. **Mock live test** — run `live_detection.py --camera example/test1.mp4` to verify detection works against a recorded file before touching the physical camera.
3. **RTSP test** — connect to the Insta360 Pro hotspot and run `live_detection.py` with the camera URL; confirm bounding boxes appear and FPS stats print.
4. **Output test** — add `--output out.mp4` and verify annotated video is saved correctly.
