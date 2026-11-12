"""Earth Rover Mini+ front camera geometry for GeNIE's ground-plane projection.

GeNIE projects SAM-TP traversability onto the ground with a pinhole model, a camera height prior and the
camera pose. The Mini+ has no published calibration, so these defaults are estimates:

- horizontal field of view ~92.6 degrees (the value used by the Sana Rover policy for the 1024x576 stream);
  distortion is ignored;
- lens ~0.14 m above the ground and ~0.11 m ahead of the base origin;
- pitch ~0 degrees: on the Wuhan plaza the far edge of the ground sits on the image's middle row.

The frame size comes from each observation, so the same numbers serve the 1024x576 platform stream and the
512x288 frames the Colosseum Client sends.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MiniPlusCamera:
    hfov_deg: float = 92.6
    height_m: float = 0.14
    forward_m: float = 0.11
    pitch_down_deg: float = 0.0

    def intrinsics(self, width: int, height: int) -> np.ndarray:
        """Pinhole K for a ``width`` x ``height`` frame (square pixels, principal point at the centre)."""
        focal = (width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])

    def pose(self) -> np.ndarray:
        """T_world_camera for the optical frame (x right, y down, z forward) with the robot at the origin.

        The world frame is the robot base frame: x forward, y left, z up, ground at z = 0.
        """
        pitch = math.radians(self.pitch_down_deg)
        right = np.array([0.0, -1.0, 0.0])
        down = np.array([-math.sin(pitch), 0.0, -math.cos(pitch)])
        forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, down, forward], axis=1)
        pose[:3, 3] = [self.forward_m, 0.0, self.height_m]
        return pose

    def half_fov_deg(self) -> float:
        return self.hfov_deg / 2.0
