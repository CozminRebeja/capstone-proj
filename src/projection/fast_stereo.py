"""
Fast panorama-to-stereographic projection using precomputed cv2.remap maps.

Usage:
    projector = FastPano2Stereo(height, width, distance=1.0)
    faces = projector.project(frame)  # returns list of 4 face images

The remap maps are computed once at init, making per-frame projection ~100x
faster than rebuilding scipy splines every frame.
"""
import numpy as np
import cv2
from math import pi

from .stereo import projection_angle


class FastPano2Stereo:
    """Precomputes remap lookup tables for equirectangular -> 4 stereographic faces."""

    def __init__(self, height, width, distance=1.0):
        """
        Args:
            height: frame height (equirectangular image height).
            width:  frame width  (equirectangular image width).
            distance: projection parameter d in (0, 1]. Default 1.0.
        """
        self.height = height
        self.width = width
        self.distance = distance
        self.maps = self._build_maps()

    def _build_maps(self):
        """Build 4 pairs of (map_x, map_y) float32 arrays for cv2.remap."""
        h = self.height
        w = self.width
        d = self.distance

        xp_max = (1 + d) / d
        yp_max = (1 + d) / d
        xp_domain = xp_max * (np.arange(-1.0, 1.0, 2.0 / h) + 1.0 / h)
        yp_domain = yp_max * (np.arange(-1.0, 1.0, 2.0 / h) + 1.0 / h)
        delta_rad = 2 * pi / w

        # Compute 1-D source panorama coordinates; broadcasting expands them to maps.
        pano_x = np.empty(h, dtype=np.float64)
        pano_y = np.empty(h, dtype=np.float64)

        for j, xp in enumerate(xp_domain):
            phi = projection_angle(xp, d)
            pano_x[j] = w / 2.0 + phi / delta_rad

        for i, yp in enumerate(yp_domain):
            theta = projection_angle(yp, d)
            pano_y[i] = h / 2.0 + theta / delta_rad

        maps = []
        for face in range(4):
            # Shift pano_x by face * (width/4) to select each 90-degree sector
            shifted_x = (pano_x + face * (w / 4.0)) % w

            # OpenCV remap needs an explicit source x/y coordinate for every output pixel.
            map_x = np.broadcast_to(shifted_x[np.newaxis, :], (h, h)).astype(np.float32).copy()
            map_y = np.broadcast_to(pano_y[:, np.newaxis], (h, h)).astype(np.float32).copy()

            maps.append((map_x, map_y))

        return maps

    def project(self, frame):
        """
        Project an equirectangular frame into 4 stereographic face images.

        Args:
            frame: BGR image, shape (height, width, 3).

        Returns:
            List of 4 face images, each shape (height, height, 3).
        """
        faces = []
        for map_x, map_y in self.maps:
            face = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_WRAP)
            faces.append(face)
        return faces
