"""GeNIE's waypoint controller for the Mini+: align with the next waypoint, then advance about a metre.

The paper: "the robot aligns its heading with the next waypoint, and then advances by 1 m", unless the
near-front region is not traversable; when the planner finds no valid path, a VLM proposes how to get back to
a navigable route. The VLM recovery is not released, so this controller backs up and turns toward the goal.

Mini+ facts behind the defaults (Jun's 107 manual drives on the Wuhan route, 23-24 Oct 2026, and the 27 Oct
trials): an in-place turn at angular 0.4-1.0 runs at 15-20 deg/s (the pivot deadband is ~0.35); at forward
0.3, angular 0.15-0.3 already turns the rover at ~7-15 deg/s, so steering while driving stays gentle; and a
command takes 1-2 s to reach the rover through Colosseum, so every in-place turn is followed by a wait for a
frame taken after the rover stopped.

Through ``colosseum-rover-sdk-bridge`` a new observation arrives every ~2-3 s and is answered at once with the
command current at that moment; the Client then holds that command until the next answer. A command the
policy sets between two observations is never seen by the rover. So an advance holds its command until the
following plan replaces it (``advance_s`` > the observation interval), and an in-place turn lasts until at
least one observation has been answered with it (each answered turn command turns the rover ~45 degrees).

Commands are Earth Rovers SDK values: linear and angular in [-1, 1], +angular turns left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .geo import bearing_distance, goal_in_robot_frame, wrap_deg


@dataclass(frozen=True)
class ControlConfig:
    forward: float = 0.30          # linear command while advancing (~1.1 m/s on the Wuhan Mini+)
    pivot: float = 0.45            # angular command for in-place turns (the pivot deadband is ~0.35)
    pivot_rate_dps: float = 18.0   # turn rate at that command
    steer_gain: float = 0.005      # angular per degree of waypoint offset while advancing
    max_steer: float = 0.15
    align_deg: float = 35.0        # turn in place when the waypoint is further off than this
    goal_cone_deg: float = 45.0    # goal outside this cone: turn toward it before planning
    max_turn_deg: float = 60.0     # largest single in-place turn
    settle_s: float = 3.0          # after a turn: frames must arrive this long after the stop to be used
    advance_s: float = 4.0         # an advance holds until the next plan (observations arrive every ~2-3 s)
    reverse: float = -0.30
    reverse_s: float = 1.5
    front_clear_min: float = 0.45  # mean traversability straight ahead needed to advance
    reach_m: float = 8.0           # call /checkpoint-reached from this GPS distance


@dataclass
class Telemetry:
    latitude: float
    longitude: float
    heading_deg: float


@dataclass
class Decision:
    linear: float
    angular: float
    state: str
    note: str = ""
    plan: object | None = None


@dataclass
class _Maneuver:
    name: str
    linear: float
    angular: float
    started: float
    until: float
    then: str = "settle"
    sampled: bool = False  # an observation reached the SDK bridge while this command was current


@dataclass
class Controller:
    goal: tuple[float, float]
    plan_fn: Callable[[np.ndarray, float, float], object]
    config: ControlConfig = field(default_factory=ControlConfig)
    maneuver: _Maneuver | None = None
    settle_until_frame: float = float("-inf")
    advancing_until: float = float("-inf")
    advance_command: tuple[float, float] = (0.0, 0.0)
    recoveries: int = 0
    last_frame_ts: float = float("-inf")

    def goal_relative(self, telemetry: Telemetry) -> tuple[float, float]:
        """(distance m, goal direction in degrees relative to the heading, positive = to the right)."""
        bearing, distance = bearing_distance(telemetry.latitude, telemetry.longitude, *self.goal)
        return distance, wrap_deg(bearing - telemetry.heading_deg)

    def step(self, now: float, telemetry: Telemetry, frame: np.ndarray | None, frame_ts: float | None) -> Decision:
        """One control tick. ``frame_ts`` is the time the frame reached the SDK bridge (same clock as ``now``)."""
        cfg = self.config
        if self.maneuver is not None:
            if frame_ts is not None and frame_ts > self.maneuver.started + 0.2:
                self.maneuver.sampled = True
            if now < self.maneuver.until or not self.maneuver.sampled:
                return Decision(self.maneuver.linear, self.maneuver.angular, self.maneuver.name)
            then, self.maneuver = self.maneuver.then, None
            if then == "settle":
                self.settle_until_frame = now + cfg.settle_s
            else:
                self.maneuver = self._turn_maneuver(now, then)
                return Decision(self.maneuver.linear, self.maneuver.angular, self.maneuver.name)
        fresh = frame is not None and frame_ts is not None and frame_ts > self.last_frame_ts
        if not fresh or frame_ts < self.settle_until_frame:
            if now < self.advancing_until:
                return Decision(*self.advance_command, "advance")
            return Decision(0.0, 0.0, "settle" if now < self.settle_until_frame + 1.0 else "wait")
        self.last_frame_ts = frame_ts
        distance, relative = self.goal_relative(telemetry)
        if abs(relative) > cfg.goal_cone_deg:
            return self._turn(now, relative, "face_goal", f"goal {relative:+.0f} deg, {distance:.0f} m")
        plan = self.plan_fn(frame, *goal_in_robot_frame(distance, relative))
        waypoint = getattr(plan, "waypoint_deg", None)
        if waypoint is None or getattr(plan, "front_clear", 1.0) < cfg.front_clear_min:
            return self._recover(now, relative, plan)
        self.recoveries = 0
        if abs(waypoint) > cfg.align_deg:
            decision = self._turn(now, waypoint, "align", f"waypoint {waypoint:+.0f} deg")
            decision.plan = plan
            return decision
        steer = float(np.clip(-cfg.steer_gain * waypoint, -cfg.max_steer, cfg.max_steer))
        self.advance_command = (cfg.forward, steer)
        self.advancing_until = now + cfg.advance_s
        return Decision(cfg.forward, steer, "advance", f"waypoint {waypoint:+.0f} deg, goal {distance:.0f} m", plan)

    def _turn(self, now: float, angle_deg: float, name: str, note: str) -> Decision:
        """Turn in place by ``angle_deg`` (positive = right), then settle."""
        cfg = self.config
        angle = max(-cfg.max_turn_deg, min(cfg.max_turn_deg, angle_deg))
        duration = abs(angle) / cfg.pivot_rate_dps
        self.advancing_until = float("-inf")
        self.maneuver = _Maneuver(name, 0.0, -math.copysign(cfg.pivot, angle), now, now + duration)
        return Decision(self.maneuver.linear, self.maneuver.angular, name, f"{note}; turning {angle:+.0f} deg")

    def _turn_maneuver(self, now: float, spec: str) -> _Maneuver:
        angle = float(spec.split(":", 1)[1])
        cfg = self.config
        return _Maneuver("recover_turn", 0.0, -math.copysign(cfg.pivot, angle), now, now + abs(angle) / cfg.pivot_rate_dps)

    def _recover(self, now: float, goal_relative: float, plan: object) -> Decision:
        """No valid path (or the way ahead is blocked): back up, then turn toward the goal, alternating sides."""
        cfg = self.config
        self.recoveries += 1
        side = 1.0 if goal_relative >= 0 else -1.0
        if self.recoveries % 2 == 0:
            side = -side
        angle = side * 45.0
        self.advancing_until = float("-inf")
        self.maneuver = _Maneuver("recover_back", cfg.reverse, 0.0, now, now + cfg.reverse_s, then=f"turn:{angle}")
        reason = "no valid path" if getattr(plan, "waypoint_deg", None) is None else f"front {plan.front_clear:.2f}"
        return Decision(cfg.reverse, 0.0, "recover_back", f"{reason}; backing up, then turning {angle:+.0f} deg", plan)
