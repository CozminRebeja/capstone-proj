"""
Read Insta360 Pro project metadata needed by the approximate stitcher.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class LensCalibration:
    index: int
    center_x: float
    center_y: float
    focal: float
    rot_x: float
    rot_y: float
    rot_z: float
    width: int
    height: int
    model: int


def _parse_float_block(text):
    if not text:
        raise RuntimeError("Project calibration block is empty.")
    return [float(part) for part in text.strip().split("_") if part]


def read_lens_calibrations(folder, block_name="pano_16_9"):
    """Parse six per-lens calibration entries from an Insta360 pro.prj file."""
    folder = Path(folder)
    project_path = folder / "pro.prj"
    if not project_path.exists():
        raise RuntimeError(f"Missing Insta360 project file: {project_path}")

    root = ET.parse(project_path).getroot()
    block = root.find(f"./offset/{block_name}")
    if block is None:
        raise RuntimeError(f"Missing calibration block <{block_name}> in {project_path}")

    values = _parse_float_block(block.text)
    count = int(values[0])
    if count != 6:
        raise RuntimeError(f"Expected 6 lenses, found {count} in {project_path}")

    stride = 9
    expected = 1 + count * stride
    if len(values) < expected:
        raise RuntimeError(
            f"Calibration block has {len(values)} values; expected at least {expected}."
        )

    lenses = []
    for index in range(count):
        start = 1 + index * stride
        (
            center_x,
            center_y,
            focal,
            rot_x,
            rot_y,
            rot_z,
            width,
            height,
            model,
        ) = values[start:start + stride]
        lenses.append(
            LensCalibration(
                index=index,
                center_x=center_x,
                center_y=center_y,
                focal=focal,
                rot_x=rot_x,
                rot_y=rot_y,
                rot_z=rot_z,
                width=int(width),
                height=int(height),
                model=int(model),
            )
        )

    return lenses


def validate_origin_videos(folder):
    """Return origin_0.mp4..origin_5.mp4 paths or raise a clear error."""
    folder = Path(folder)
    paths = [folder / f"origin_{index}.mp4" for index in range(6)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Missing origin videos: " + ", ".join(missing))
    return paths
