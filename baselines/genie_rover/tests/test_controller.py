"""Controller and geometry tests; no torch, checkpoint or network needed."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from genie_rover.camera import MiniPlusCamera
from genie_rover.controller import ControlConfig, Controller, Telemetry
from genie_rover.geo import HeadingEstimator, bearing_distance, goal_in_robot_frame, wrap_deg

START = (30.48243713, 114.3026428)  # Wuhan mission-1 checkpoint 1
GOAL = (30.48268318, 114.3026047)  # checkpoint 2, ~28 m north-north-west
FRAME = np.zeros((288, 512, 3), np.uint8)


def plan(waypoint_deg=0.0, front_clear=0.9):
    return SimpleNamespace(waypoint_deg=waypoint_deg, front_clear=front_clear, status="ok")


def controller(plan_result=None, **overrides):
    calls = []

    def plan_fn(rgb, x, y):
        calls.append((x, y))
        return plan_result if plan_result is not None else plan()

    return Controller(goal=GOAL, plan_fn=plan_fn, config=ControlConfig(**overrides)), calls


def test_bearing_and_distance_to_checkpoint_2():
    bearing, distance = bearing_distance(*START, *GOAL)
    assert distance == pytest.approx(27.5, abs=0.5)
    assert wrap_deg(bearing) == pytest.approx(-7.6, abs=1.0)


def test_goal_in_robot_frame_is_right_forward():
    x, y = goal_in_robot_frame(10.0, 90.0)
    assert x == pytest.approx(10.0) and y == pytest.approx(0.0, abs=1e-9)


def test_camera_pose_is_a_rotation_looking_forward():
    pose = MiniPlusCamera(pitch_down_deg=8.0).pose()
    rotation = pose[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation[:, 2][0] > 0.9 and rotation[:, 2][2] < 0  # optical axis forward and slightly down
    assert rotation[:, 0][1] == pytest.approx(-1.0)  # image right = robot right (-y)


def test_goal_behind_turns_in_place_toward_it_then_waits_for_a_fresh_frame():
    ctrl, calls = controller()
    facing_south = Telemetry(*START, heading_deg=180.0)
    decision = ctrl.step(0.0, facing_south, FRAME, 0.0)
    assert decision.state == "face_goal" and decision.linear == 0.0 and abs(decision.angular) == 0.45
    assert calls == []
    after = ctrl.step(60.0 / 18.0 + 0.05, facing_south, FRAME, 1.0)
    assert after.state == "settle" and (after.linear, after.angular) == (0.0, 0.0)
    # A frame that reached the bridge before the settle time is not used.
    assert ctrl.step(4.0, Telemetry(*START, heading_deg=-8.0), FRAME, 4.0).state == "settle"
    assert ctrl.step(7.0, Telemetry(*START, heading_deg=-8.0), FRAME, 6.6).state == "advance"


def test_turn_direction_follows_sdk_sign_convention():
    ctrl, _ = controller()
    goal_to_the_right = Telemetry(*START, heading_deg=-80.0)  # goal bearing ~ -8 deg, so it is ~72 deg right
    assert ctrl.step(0.0, goal_to_the_right, FRAME, 0.0).angular < 0  # +angular is a left turn
    ctrl, _ = controller()
    goal_to_the_left = Telemetry(*START, heading_deg=60.0)
    assert ctrl.step(0.0, goal_to_the_left, FRAME, 0.0).angular > 0


def test_advances_toward_a_near_straight_waypoint_and_keeps_moving_between_frames():
    ctrl, calls = controller(plan(waypoint_deg=10.0))
    facing_goal = Telemetry(*START, heading_deg=-8.0)
    decision = ctrl.step(0.0, facing_goal, FRAME, 0.0)
    assert decision.state == "advance" and decision.linear == 0.30
    assert decision.angular == pytest.approx(-0.05)  # waypoint to the right: steer right
    x, y = calls[0]
    assert y > 25 and abs(x) < 2
    assert ctrl.step(3.0, facing_goal, FRAME, 0.0).state == "advance"  # same frame: keep the command
    assert ctrl.step(4.5, facing_goal, FRAME, 0.0).state == "wait"  # no new plan for advance_s: stop


def test_large_waypoint_offset_aligns_in_place():
    ctrl, _ = controller(plan(waypoint_deg=-40.0))
    decision = ctrl.step(0.0, Telemetry(*START, heading_deg=-8.0), FRAME, 0.0)
    assert decision.state == "align" and decision.linear == 0.0 and decision.angular > 0


def test_no_path_backs_up_then_turns_toward_the_goal():
    ctrl, _ = controller(plan(waypoint_deg=None))
    facing = Telemetry(*START, heading_deg=-30.0)  # goal ~22 deg to the right
    decision = ctrl.step(0.0, facing, FRAME, 0.0)
    assert decision.state == "recover_back" and decision.linear < 0
    turn = ctrl.step(1.6, facing, FRAME, 1.0)
    assert turn.state == "recover_turn" and turn.angular < 0
    assert ctrl.step(1.6 + 45.0 / 18.0 + 0.1, facing, FRAME, 3.0).state == "settle"


def test_blocked_front_counts_as_no_path():
    ctrl, _ = controller(plan(waypoint_deg=0.0, front_clear=0.2))
    assert ctrl.step(0.0, Telemetry(*START, heading_deg=-8.0), FRAME, 0.0).state == "recover_back"


def test_heading_offset_learned_from_a_straight_gps_segment():
    estimator = HeadingEstimator(gain=1.0)
    north_m = 1.0 / 111_195.0
    for step in range(8):  # 7 m due north while the compass reads 20 degrees
        estimator.update(float(step), START[0] + step * north_m, START[1], 20.0, forward=True)
    assert estimator.calibrated
    assert estimator.heading(20.0) == pytest.approx(0.0, abs=1.0) or estimator.heading(20.0) == pytest.approx(360.0, abs=1.0)


def test_heading_offset_ignores_turning_segments():
    estimator = HeadingEstimator()
    north_m = 1.0 / 111_195.0
    for step in range(8):
        estimator.update(float(step), START[0] + step * north_m, START[1], 20.0 + 10 * step, forward=True)
    assert not estimator.calibrated and estimator.offset_deg == 0.0


def test_planning_helpers():
    planning = pytest.importorskip("genie_rover.planning")
    bev = planning.BevSettings()
    x, y = planning.clip_goal_to_map(-20.0, 20.0, bev)
    assert math.atan2(x, y) == pytest.approx(math.atan2(-20.0, 20.0))
    assert abs(x) <= bev.side_m and y <= bev.forward_m
    path = np.stack([np.zeros(11), np.linspace(0, 5, 11)], axis=1)
    assert planning.waypoint_along(path, 1.0) == pytest.approx((0.0, 1.0))


def test_a_turn_lasts_until_an_observation_has_been_answered_with_it():
    ctrl, _ = controller(plan(waypoint_deg=-50.0))
    facing = Telemetry(*START, heading_deg=-8.0)
    assert ctrl.step(0.0, facing, FRAME, 0.0).state == "align"
    # The turn's duration (50/18 s) has passed, but no observation reached the bridge since it started.
    assert ctrl.step(3.0, facing, FRAME, 0.0).state == "align"
    assert ctrl.step(3.5, facing, FRAME, 0.1).state == "align"  # arrived too soon after the command
    assert ctrl.step(4.0, facing, FRAME, 3.9).state == "settle"
