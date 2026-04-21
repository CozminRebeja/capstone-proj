"""
Approximate six-lens Insta360 Pro stitcher.

This is a practical visualization stitcher. It uses pro.prj lens centers,
focal lengths, and rotations with a simple fisheye model. The default "nearest"
blend chooses one lens per panorama pixel to avoid ghosting from misaligned
overlaps; "feather" remains available for smoother but softer seams.
"""
from pathlib import Path

import cv2
import numpy as np

from .project import read_lens_calibrations, validate_origin_videos


def _axis_rotation(axis, degrees):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _rotation_matrix(rx_deg, ry_deg, rz_deg, order):
    values = {"x": rx_deg, "y": ry_deg, "z": rz_deg}
    rotation = np.eye(3, dtype=np.float32)
    for axis in order:
        rotation = _axis_rotation(axis, values[axis]) @ rotation
    return rotation


def _world_directions(width, height):
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    lon = (x * 2.0 - 1.0) * np.pi
    lat = (0.5 - y) * np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    return np.stack(
        [
            cos_lat * np.sin(lon_grid),
            np.sin(lat_grid),
            cos_lat * np.cos(lon_grid),
        ],
        axis=-1,
    ).astype(np.float32)


INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _build_lens_map(lens, directions, edge_degrees, rotation_order, inverse_rotation):
    rotation = _rotation_matrix(lens.rot_x, lens.rot_y, lens.rot_z, rotation_order)
    if inverse_rotation:
        rotation = rotation.T
    local = directions @ rotation

    z = np.clip(local[:, :, 2], -1.0, 1.0)
    theta = np.arccos(z)
    radial = lens.focal * theta
    radius_xy = np.sqrt(local[:, :, 0] ** 2 + local[:, :, 1] ** 2)
    radius_xy = np.maximum(radius_xy, 1e-6)

    map_x = lens.center_x + radial * (local[:, :, 0] / radius_xy)
    map_y = lens.center_y - radial * (local[:, :, 1] / radius_xy)

    valid = (
        (z > 0)
        & (map_x >= 0)
        & (map_x < lens.width - 1)
        & (map_y >= 0)
        & (map_y < lens.height - 1)
    )

    max_angle = np.radians(edge_degrees)
    weight = np.clip(max_angle - theta, 0.0, None)
    weight = np.where(valid, weight, 0.0).astype(np.float32)

    return map_x.astype(np.float32), map_y.astype(np.float32), weight


class Insta360Stitcher:
    """Precomputes per-lens maps and blends six fisheye frames into a panorama."""

    def __init__(self, folder, output_width=3840, calibration_block="pano_16_9",
                 edge_degrees=96.0, blend_mode="nearest", feather_degrees=6.0,
                 interpolation="lanczos", rotation_order="zxy",
                 inverse_rotation=True):
        self.folder = Path(folder)
        self.output_width = int(output_width)
        self.output_height = self.output_width // 2
        self.origin_paths = validate_origin_videos(self.folder)
        self.lenses = read_lens_calibrations(self.folder, calibration_block)
        if blend_mode not in ("nearest", "feather"):
            raise RuntimeError(f"Unsupported blend mode: {blend_mode}")
        if interpolation not in INTERPOLATION:
            raise RuntimeError(f"Unsupported interpolation: {interpolation}")
        if sorted(rotation_order) != ["x", "y", "z"]:
            raise RuntimeError(f"Unsupported rotation order: {rotation_order}")
        self.blend_mode = blend_mode
        self.feather_degrees = float(feather_degrees)
        self.interpolation = INTERPOLATION[interpolation]
        self.rotation_order = rotation_order
        self.inverse_rotation = bool(inverse_rotation)
        self._maps = self._build_maps(edge_degrees)

    def _build_maps(self, edge_degrees):
        directions = _world_directions(self.output_width, self.output_height)
        maps = [
            _build_lens_map(
                lens,
                directions,
                edge_degrees,
                self.rotation_order,
                self.inverse_rotation,
            )
            for lens in self.lenses
        ]
        if self.blend_mode == "nearest":
            maps = self._winner_take_all_maps(maps)
        return maps

    def _winner_take_all_maps(self, maps):
        weights = np.stack([weight for _, _, weight in maps], axis=0)
        owners = np.argmax(weights, axis=0)
        covered = np.max(weights, axis=0) > 0.0

        crisp_maps = []
        for lens_index, (map_x, map_y, _weight) in enumerate(maps):
            mask = (owners == lens_index) & covered
            crisp_maps.append((map_x, map_y, mask.astype(np.float32)))
        return crisp_maps

    def _feather_weight(self, weight):
        if self.blend_mode != "feather":
            return weight
        feather = np.radians(max(self.feather_degrees, 0.1))
        return np.clip(weight / feather, 0.0, 1.0).astype(np.float32)

    def stitch(self, frames):
        accum = np.zeros((self.output_height, self.output_width, 3), dtype=np.float32)
        total_weight = np.zeros((self.output_height, self.output_width), dtype=np.float32)

        for frame, (map_x, map_y, weight) in zip(frames, self._maps):
            sampled = cv2.remap(
                frame,
                map_x,
                map_y,
                self.interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.float32)
            weight = self._feather_weight(weight)
            accum += sampled * weight[:, :, None]
            total_weight += weight

        valid = total_weight > 1e-6
        output = np.zeros_like(accum, dtype=np.uint8)
        output[valid] = np.clip(
            accum[valid] / total_weight[valid, None],
            0,
            255,
        ).astype(np.uint8)
        return output


def open_origin_captures(paths):
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    unopened = [str(path) for path, cap in zip(paths, captures) if not cap.isOpened()]
    if unopened:
        for cap in captures:
            cap.release()
        raise RuntimeError("Unable to open origin videos: " + ", ".join(unopened))
    return captures


def read_frame_set(captures):
    frames = []
    for capture in captures:
        ok, frame = capture.read()
        if not ok:
            return None
        frames.append(frame)
    return frames


def source_video_info(captures):
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
    fps = next((value for value in fps_values if value and value > 0), 30.0)
    shortest = min(count for count in frame_counts if count > 0)
    return fps, shortest, frame_counts
