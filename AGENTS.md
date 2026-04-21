# Repository Guidelines

## Project Structure & Module Organization
`src/` is a Python package with `__main__.py` as the unified entrypoint. Subpackages group related modules:
- [src/detection/](src/detection/) — YOLO wrapper ([yolo.py](src/detection/yolo.py)), offline video pipeline ([video.py](src/detection/video.py)), live RTSP pipeline ([live.py](src/detection/live.py)).
- [src/projection/](src/projection/) — equirectangular↔stereographic math ([stereo.py](src/projection/stereo.py)) and its precomputed remap variant ([fast_stereo.py](src/projection/fast_stereo.py)).
- [src/viewer/](src/viewer/) — interactive pinhole navigator ([pano_viewer.py](src/viewer/pano_viewer.py)), crystal-ball previewer ([crystal_ball.py](src/viewer/crystal_ball.py)), batch renderer ([render.py](src/viewer/render.py)).
- [src/insta360/](src/insta360/) — pro.prj parser ([project.py](src/insta360/project.py)), six-lens stitcher ([stitcher.py](src/insta360/stitcher.py)), stitch CLI ([stitch_cli.py](src/insta360/stitch_cli.py)).
- [src/cli.py](src/cli.py) and [src/__main__.py](src/__main__.py) dispatch the top-level `--video / --live / --preview` flags.
- [src/paths.py](src/paths.py) resolves `assets/` (YOLO model, labels) and `data/` (sample media) relative to the repo root, so commands work from any CWD.
Model files live in [assets/](assets/); sample media in [data/](data/).

## Build, Test, and Development Commands
Set up Python locally from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run from the repo root. Paths resolve via `src/paths.py`, so CWD no longer matters:

```bash
python -m src --video data/videos/preview.mp4 --output-dir data/processed
python -m src --live rtsp://192.168.1.188:8554/stitched
python -m src --preview data/result.mp4
python -m src.detection.yolo input.jpg output.jpg
python -m src.detection.video --input data/videos/preview.mp4 --output data/result.mp4
python -m src.projection.stereo --p2s input.jpg --d 1.0
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and module-level constants in `UPPER_SNAKE_CASE`. Keep scripts importable by guarding CLIs with `if __name__ == "__main__":`. Prefer small, focused helpers over adding more logic to the existing entrypoint scripts. Use docstrings for public functions and keep OpenCV/NumPy code explicit rather than overly compact.

## Testing Guidelines
No automated test suite is checked in yet. Until one exists, every change should include a reproducible smoke test using files in `data/videos/` or `data/images/`, plus a brief note on expected output. For new reusable logic, add `pytest`-style tests under `tests/` named `test_<module>.py`.

## Commit & Pull Request Guidelines
The `src/` Git history mostly uses short conventional prefixes (`feat:`, `chore:`), though some older commits are generic. Prefer clear Conventional Commits such as `fix: clamp invalid RTSP reconnect loop`. PRs should include: purpose, local run steps, sample input/output paths, and screenshots or rendered frames for any detection or viewer change. Call out large binary additions explicitly.

## Configuration & Assets
Keep `assets/yolov3.cfg`, `assets/coco.names`, and the `assets/yolov3.weights` symlink valid before running detection. Do not commit camera credentials or private RTSP endpoints; use local overrides when testing live streams.
