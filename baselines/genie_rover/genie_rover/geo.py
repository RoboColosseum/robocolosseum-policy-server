"""GPS goal geometry and the heading estimate.

The rover's ``orientation`` is a magnetometer heading in degrees clockwise from north (a left turn lowers it).
It can carry a site-dependent bias, so the estimate adds an offset learned from the GPS course over ground
whenever the rover has driven a few metres in a straight line.
"""
from __future__ import annotations

import math
from collections import deque

EARTH_RADIUS_M = 6_371_000.0


def wrap_deg(angle: float) -> float:
    """Wrap to (-180, 180]."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def bearing_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """Bearing in degrees clockwise from north and distance in metres from point 1 to point 2."""
    north = math.radians(lat2 - lat1) * EARTH_RADIUS_M
    east = math.radians(lon2 - lon1) * EARTH_RADIUS_M * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.degrees(math.atan2(east, north)) % 360.0, math.hypot(north, east)


def goal_in_robot_frame(distance_m: float, relative_deg: float) -> tuple[float, float]:
    """The goal as GeNIE's planner wants it: (x right, y forward) in metres."""
    angle = math.radians(relative_deg)
    return distance_m * math.sin(angle), distance_m * math.cos(angle)


class HeadingEstimator:
    """Compass heading plus an offset learned from straight GPS segments."""

    def __init__(self, *, segment_m: float = 5.0, max_turn_deg: float = 12.0, gain: float = 0.5,
                 window_s: float = 10.0) -> None:
        self.segment_m = segment_m
        self.max_turn_deg = max_turn_deg
        self.gain = gain
        self.window_s = window_s
        self.offset_deg = 0.0
        self.calibrated = False
        self._track: deque[tuple[float, float, float, float]] = deque()

    def update(self, now: float, latitude: float, longitude: float, compass_deg: float, *, forward: bool) -> float:
        """Record a fix and return the current heading estimate in degrees clockwise from north."""
        if not forward:
            self._track.clear()
        else:
            self._track.append((now, latitude, longitude, compass_deg))
            while self._track and now - self._track[0][0] > self.window_s:
                self._track.popleft()
            self._learn()
        return self.heading(compass_deg)

    def heading(self, compass_deg: float) -> float:
        return (compass_deg + self.offset_deg) % 360.0

    def _learn(self) -> None:
        first, last = self._track[0], self._track[-1]
        course, moved = bearing_distance(first[1], first[2], last[1], last[2])
        if moved < self.segment_m:
            return
        reference = self._track[0][3]
        turns = [abs(wrap_deg(sample[3] - reference)) for sample in self._track]
        if max(turns) <= self.max_turn_deg:
            mean_compass = reference + sum(wrap_deg(sample[3] - reference) for sample in self._track) / len(self._track)
            error = wrap_deg(course - (mean_compass + self.offset_deg))
            self.offset_deg = wrap_deg(self.offset_deg + self.gain * error)
            self.calibrated = True
        self._track.clear()
