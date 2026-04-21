Insta360 Pro Live Setup

1. Power on the camera and wait for it to boot (status LED
   stabilizes).
2. Connect to the camera's Wi-Fi hotspot:

- SSID: Insta360 Pro_XXXXXX (check the camera label for the
  exact name)
- The camera acts as a router — your machine gets an IP on the
  192.168.1.x subnet
- Camera IP: 192.168.1.188

3. Verify connectivity:
   ping 192.168.1.188
4. Start live stitching on the camera. You need to put the
   camera into live-streaming mode — either:

- Use the Insta360 Pro companion app (iOS/Android) to start a
  live preview, or
- Use the physical button on the camera to switch to live mode

  The camera must be actively stitching for the RTSP stream to be
  available.

5. Run the pipeline — try these RTSP URLs in order:
   cd /Users/min/Desktop/01-projects/@imc/capstone/src

# Try stitched stream first

  python live_detection.py --camera
  rtsp://192.168.1.188:8554/stitched

# If that fails, try this alternate path

  python live_detection.py --camera rtsp://192.168.1.188:8554/live_stream

# For a single raw fisheye lens (no stitching needed on camera)

  python live_detection.py --camera
  rtsp://192.168.1.188:8554/origin_0

  Quick test — you can verify the stream works before running
  detection:
  ffplay rtsp://192.168.1.188:8554/stitched
  (requires ffmpeg installed — brew install ffmpeg)

  Troubleshooting:

- If no stream connects, make sure the camera is in live mode,
  not recording mode
- Some firmware versions use different ports — try 1935 if 8554
  doesn't work
- The stitched stream requires the camera to be actively
  stitching, which uses more battery — keep it plugged in
