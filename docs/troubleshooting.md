# Troubleshooting

## Model Files Are Not Found

`detection.py` uses relative paths for `yolov3.cfg`, `yolov3.weights`, and `coco.names`. Run from `src/` or place/link those files in the current working directory.

## No RTSP Frames Arrive

Check these points:

- The machine is connected to the Insta360 Pro Wi-Fi hotspot.
- The camera is in live-streaming mode.
- `ping 192.168.1.188` succeeds.
- Another app is not already consuming the stream.
- Try both `8554` and camera-specific alternate ports if the firmware differs.

## Live Display Lags

Lower the processing width:

```bash
python live_detection.py --camera <source> --width 960
```

The live pipeline keeps the newest frame, but YOLO inference can still limit how often annotations update.

## Output Video Is Empty

Confirm that `cv2.VideoWriter` can create the selected output path and codec on the machine. Try an `.mp4` extension first because the scripts use the `mp4v` codec.

## Boxes Look Duplicated or Noisy

Tune the detection thresholds:

```bash
python detection.py input.jpg output.jpg \
  --conf-threshold 0.55 \
  --objectness-threshold 0.60 \
  --nms-threshold 0.20
```

Higher confidence thresholds reduce false positives. Lower NMS thresholds merge overlapping boxes more aggressively.

