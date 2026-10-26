from __future__ import annotations

import base64
import json
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from colosseum_policy_server import ImageFrame, Observation
from colosseum_policy_server.rover_sdk_bridge import Bridge, PolicyProcess, distance_m, handler_for, png

GOAL = (30.48268318, 114.3026047)
START = (30.48243713, 114.3026428)  # Wuhan mission-1 checkpoints 1 and 2, about 28 m apart


def observation(session="ses-1", position=START, goal=GOAL, rgb=None):
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) if rgb is None else rgb
    navigation = np.array([*position, 271.5, 0.4, 87.0, 10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    state = {"navigation_state": navigation}
    if goal is not None:
        state["goal_gps"] = np.array(goal, dtype=np.float64)
    frame = ImageFrame(image_id="front_image", encoding="RAW_RGB", capture_time_ns=1,
                       width=rgb.shape[1], height=rgb.shape[0], data=rgb.tobytes())
    return Observation(session_id=session, sequence=1, deadline_ms=5000, robot_time_ns=1, control_step=0,
                       instruction="Drive to checkpoint 2", raw_state=state, images=(frame,), _received_monotonic_ns=1)


@pytest.fixture
def served():
    bridge = Bridge(max_linear=0.5, max_angular=1.0, horizon=3)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(bridge))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    yield bridge, call
    server.shutdown()


def decode_png(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    idat_length = struct.unpack(">I", data[33:37])[0]
    rows = np.frombuffer(zlib.decompress(data[41:41 + idat_length]), np.uint8).reshape(height, width * 3 + 1)
    return rows[:, 1:].reshape(height, width, 3)


def test_png_round_trip():
    rgb = np.random.default_rng(0).integers(0, 255, (4, 5, 3), dtype=np.uint8)
    np.testing.assert_array_equal(decode_png(png(rgb)), rgb)


def test_endpoints_wait_for_a_trial(served):
    _, call = served
    for method, path in (("GET", "/v2/screenshot"), ("GET", "/data"), ("GET", "/checkpoints-list"),
                         ("POST", "/start-mission"), ("POST", "/checkpoint-reached")):
        assert call(method, path, {} if method == "POST" else None)[0] == 503
    assert call("POST", "/control", {"command": {"linear": 0.2, "angular": 0}})[0] == 503
    assert call("GET", "/nope")[0] == 404


def test_trial_is_served_in_the_sdk_formats(served):
    bridge, call = served
    rgb = np.random.default_rng(1).integers(0, 255, (2, 3, 3), dtype=np.uint8)
    assert bridge.observe(observation(rgb=rgb))
    assert not bridge.observe(observation(rgb=rgb))
    status, shot = call("GET", "/v2/screenshot")
    assert status == 200 and set(shot) >= {"front_frame", "timestamp"}
    np.testing.assert_array_equal(decode_png(base64.b64decode(shot["front_frame"])), rgb)
    status, data = call("GET", "/data")
    assert status == 200
    assert (data["latitude"], data["longitude"], data["orientation"], data["battery"]) == (*START, 271.5, 87.0)
    assert data["rpms"][-1][:4] == [10.0, 11.0, 12.0, 13.0]
    assert call("POST", "/start-mission", {})[0] == 200
    status, checkpoints = call("GET", "/checkpoints-list")
    assert status == 200 and checkpoints["latest_scanned_checkpoint"] == 0
    assert [(c["sequence"], float(c["latitude"]), float(c["longitude"])) for c in checkpoints["checkpoints_list"]] == [(1, *GOAL)]
    assert isinstance(checkpoints["checkpoints_list"][0]["latitude"], str)  # the SDK sends strings here


def test_control_becomes_normalized_chunks_and_expires(served):
    bridge, call = served
    bridge.observe(observation())
    chunk, done = bridge.actions()
    assert not done and np.array_equal(chunk, np.zeros((3, 2), np.float32))
    assert call("POST", "/control", {"command": {"linear": 0.25, "angular": -0.4, "lamp": 1}})[0] == 200
    chunk, done = bridge.actions()
    assert not done and chunk.shape == (3, 2)
    np.testing.assert_allclose(chunk[0], [0.5, -0.4])  # 0.25 / max_linear 0.5
    bridge.trial.commanded -= 10
    assert np.array_equal(bridge.actions()[0], np.zeros((3, 2), np.float32))
    assert call("POST", "/control", {"command": {"linear": "fast"}})[0] == 400


def test_checkpoint_reached_needs_the_radius_then_declares_done(served):
    bridge, call = served
    bridge.observe(observation())
    status, body = call("POST", "/checkpoint-reached", {})
    assert status == 400 and body["detail"]["proximate_distance_to_checkpoint"] == pytest.approx(distance_m(START, GOAL), abs=0.1)
    assert 25 < distance_m(START, GOAL) < 30
    bridge.observe(observation(position=(GOAL[0] - 0.00005, GOAL[1])))  # about 5.6 m short
    assert call("POST", "/checkpoint-reached", {})[0] == 200
    assert call("GET", "/checkpoints-list")[1]["latest_scanned_checkpoint"] == 1
    chunk, done = bridge.actions()
    assert done and not chunk.any()


def test_new_session_resets_the_trial(served):
    bridge, call = served
    bridge.observe(observation())
    call("POST", "/control", {"command": {"linear": 0.3, "angular": 0}})
    assert bridge.observe(observation(session="ses-2"))
    assert not bridge.actions()[0].any()


def test_trial_without_goal_has_no_checkpoints(served):
    bridge, call = served
    bridge.observe(observation(goal=None))
    assert call("GET", "/checkpoints-list")[1] == {"checkpoints_list": [], "latest_scanned_checkpoint": 0}
    assert call("POST", "/checkpoint-reached", {})[0] == 400


def test_policy_exit_declares_done():
    bridge = Bridge()
    bridge.observe(observation())
    chunk, done = bridge.actions(policy_exited=True)
    assert done and not chunk.any()


def test_policy_process_stops_on_sigint():
    policy = PolicyProcess(f"{sys.executable} -c \"import time; time.sleep(30)\"")
    policy.start()
    assert policy.running and not policy.exited
    started = time.monotonic()
    policy.stop("test")
    assert policy.exited and time.monotonic() - started < 10
