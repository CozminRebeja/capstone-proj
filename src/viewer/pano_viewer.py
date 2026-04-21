"""
Interactive rectilinear viewer for equirectangular panoramas.

Projects a 360x180 equirect frame into a perspective view with mouse controls:
    - Left-drag:  inverted pan (drag right -> view left, drag down -> view up)
    - Scroll:     zoom (fov)
    - Arrow keys: pan (fallback for touchpads without scroll)

Usage:
    viewer = PanoViewer(out_h=720, out_w=1280, fov=90.0)
    cv2.namedWindow("view")
    cv2.setMouseCallback("view", viewer.on_mouse)
    while True:
        rect = viewer.render(equirect_frame)
        cv2.imshow("view", rect)
        viewer.handle_key(cv2.waitKey(1) & 0xFF)
"""
from typing import Any, Optional, Tuple, cast
import numpy as np
import cv2


class PanoViewer:
    def __init__(self, out_h=720, out_w=1280, fov=90.0, roll=0.0):
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.fov = float(fov)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = float(roll)

        u = np.arange(self.out_w) - self.out_w / 2.0 + 0.5
        v = np.arange(self.out_h) - self.out_h / 2.0 + 0.5
        # Pixel-centered camera-plane coordinates are reused for every source frame.
        self._u_grid, self._v_grid = np.meshgrid(u, v)

        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._last_src_shape: Optional[Tuple[int, int]] = None
        self._dirty = True

        self._dragging = False
        self._last_xy = (0, 0)

    def set_view(self, yaw=None, pitch=None, fov=None, roll=None):
        if yaw is not None:
            self.yaw = ((yaw + 180.0) % 360.0) - 180.0
        if pitch is not None:
            self.pitch = max(-89.0, min(89.0, pitch))
        if fov is not None:
            self.fov = max(20.0, min(140.0, fov))
        if roll is not None:
            self.roll = ((roll + 180.0) % 360.0) - 180.0
        self._dirty = True

    def _build_maps(self, src_h, src_w):
        # Convert output pixels into unit rays from a pinhole camera model.
        f = (self.out_w / 2.0) / np.tan(np.radians(self.fov) / 2.0)
        x = self._u_grid / f
        y = -self._v_grid / f
        z = np.ones_like(x)

        norm = np.sqrt(x * x + y * y + z * z)
        x /= norm
        y /= norm
        z /= norm

        r = np.radians(self.roll)
        cr, sr = np.cos(r), np.sin(r)
        x0 = x * cr - y * sr
        y0 = x * sr + y * cr
        z0 = z

        p = np.radians(self.pitch)
        yw = np.radians(self.yaw)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(yw), np.sin(yw)

        # Apply roll, pitch, then yaw, so controls stay camera-like.
        y1 = y0 * cp - z0 * sp
        z1 = y0 * sp + z0 * cp
        x1 = x0

        x2 = x1 * cy + z1 * sy
        y2 = y1
        z2 = -x1 * sy + z1 * cy

        lon = np.arctan2(x2, z2)
        lat = np.arcsin(np.clip(y2, -1.0, 1.0))

        # cv2.remap expects source pixel coordinates for each output pixel.
        self._map_x = ((lon / (2.0 * np.pi) + 0.5) * src_w).astype(np.float32)
        # Equirectangular images place +latitude at the top of the frame.
        self._map_y = ((0.5 - lat / np.pi) * src_h).astype(np.float32)
        self._last_src_shape = (src_h, src_w)
        self._dirty = False

    def render(self, equirect):
        h, w = equirect.shape[:2]
        if self._dirty or self._last_src_shape != (h, w):
            self._build_maps(h, w)

        assert self._map_x is not None
        assert self._map_y is not None

        return cv2.remap(
            cast(Any, equirect), cast(Any, self._map_x), cast(Any, self._map_y),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
        )

    def on_mouse(self, event, x, y, flags, _param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._last_xy = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            dx = x - self._last_xy[0]
            dy = y - self._last_xy[1]
            self._last_xy = (x, y)
            deg_per_px = self.fov / self.out_w
            self.set_view(
                yaw=self.yaw - dx * deg_per_px,
                pitch=self.pitch - dy * deg_per_px,
            )
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = np.int16(flags >> 16)
            self.set_view(fov=self.fov - delta * 0.05)

    def handle_key(self, key):
        # Arrow keys (platform-dependent codes); also WASD fallback.
        step = max(2.0, self.fov * 0.04)
        if key in (ord('a'), 81):       # left
            self.set_view(yaw=self.yaw - step)
        elif key in (ord('d'), 83):     # right
            self.set_view(yaw=self.yaw + step)
        elif key in (ord('w'), 82):     # up
            self.set_view(pitch=self.pitch + step)
        elif key in (ord('s'), 84):     # down
            self.set_view(pitch=self.pitch - step)
        elif key in (ord('='), ord('+')):
            self.set_view(fov=self.fov - 5)
        elif key in (ord('-'), ord('_')):
            self.set_view(fov=self.fov + 5)
        elif key == ord('e'):
            self.set_view(roll=self.roll + 5)
        elif key == ord('q'):
            self.set_view(roll=self.roll - 5)
        elif key == ord('0'):
            self.set_view(yaw=0, pitch=0, fov=90, roll=0)
