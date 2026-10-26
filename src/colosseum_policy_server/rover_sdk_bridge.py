"""Run an Earth Rovers SDK navigation policy in Colosseum without changing it.

Earth Rover policies usually drive the rover through the Earth Rovers SDK's local HTTP API. This bridge
serves the endpoints they use on localhost, filled from the current Colosseum trial, and answers each
observation with the policy's latest ``POST /control``:

- ``GET /v2/screenshot`` (and ``/v2/front``): the trial's cameras as base64 PNG (``front_frame``, ``rear_frame``).
- ``GET /data``: telemetry from ``navigation_state``; ``orientation`` is the heading in degrees and
  ``rpms`` is ``[[fl, fr, rl, rr, timestamp]]``.
- ``POST /start-mission`` and ``GET /checkpoints-list``: the trial's GPS goal as the mission's only checkpoint.
- ``POST /checkpoint-reached``: accepted within ``--reach-m`` of the goal (15 m, the FrodoBots backend's
  radius); the next observation is then answered with a zero chunk and ``done=True``.
- ``POST /control``: SDK commands in [-1, 1], divided by the operator's adapter limits (``--max-linear``,
  ``--max-angular``; 1, the default, sends them unchanged). A command older than ``--command-ttl`` becomes a stop.

With ``--run`` the policy command starts when a trial's first observation arrives and gets SIGINT when the
trial ends, so it stops the rover and writes its own report; if it exits during a trial, the trial is
declared done.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import shlex
import signal
import struct
import subprocess
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import yaml

from .sync_sdk import ColosseumPolicySDK

REACH_M = 15.0  # frodobots_web ReachMissionCheckpoint::BOT_PROXIMITY_REACH_IN_M
EARTH_RADIUS_M = 6_371_000.0
CONTROL_HZ = 10


def png(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 frame as PNG (the SDK's default screenshot format) with the standard library."""
    height, width, _ = rgb.shape
    rows = np.concatenate([np.zeros((height, 1), np.uint8), np.ascontiguousarray(rgb, np.uint8).reshape(height, width * 3)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows.tobytes(), 1)) + chunk(b"IEND", b""))


def frame_bytes(frame) -> bytes:
    """JPEG and PNG frames pass through; RAW_RGB (what the Client sends) is encoded as PNG."""
    return frame.data if frame.encoding in {"JPEG", "PNG"} else png(frame.as_rgb())


def distance_m(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


class Trial:
    """What the SDK endpoints serve for one Colosseum session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.frames: dict[str, str] = {}
        self.navigation: np.ndarray | None = None
        self.goal: tuple[float, float] | None = None
        self.seen = time.monotonic()
        self.captured = time.time()
        self.command = (0.0, 0.0, 0)
        self.commanded = float("-inf")
        self.started = False
        self.reached = False


class Bridge:
    """Shared between the HTTP threads (the policy's calls) and the SDK loop (observations and actions)."""

    def __init__(self, *, max_linear: float = 1.0, max_angular: float = 1.0, reach_m: float = REACH_M,
                 command_ttl_s: float = 3.0, idle_s: float = 5.0, horizon: int = 5) -> None:
        if not (0 < max_linear <= 1 and 0 < max_angular <= 1):
            raise ValueError("max_linear and max_angular must be in (0, 1]")
        self.limits = (max_linear, max_angular)
        self.reach_m = reach_m
        self.command_ttl_s = command_ttl_s
        self.idle_s = idle_s
        self.horizon = horizon
        self.lock = threading.Lock()
        self.trial: Trial | None = None

    # Observations and actions (SDK loop)

    def observe(self, observation) -> bool:
        """Store the observation; True when it opens a new trial."""
        frames = {}
        for view in ("front", "rear"):
            frame = observation.image(f"{view}_image")
            if frame is not None:
                frames[view] = base64.b64encode(frame_bytes(frame)).decode("ascii")
        navigation = observation.raw_state.get("navigation_state")
        goal = observation.raw_state.get("goal_gps")
        with self.lock:
            new = self.trial is None or self.trial.session_id != observation.session_id
            if new:
                self.trial = Trial(observation.session_id)
            trial = self.trial
            trial.frames.update(frames)
            trial.navigation = None if navigation is None else np.asarray(navigation, np.float64).reshape(-1)
            trial.goal = None if goal is None else (float(goal[0]), float(goal[1]))
            trial.seen, trial.captured = time.monotonic(), time.time()
        return new

    def actions(self, *, policy_exited: bool = False) -> tuple[np.ndarray, bool]:
        """The chunk answering the current observation and whether it declares the trial done."""
        with self.lock:
            trial = self.trial
            if trial is None or trial.reached or policy_exited:
                return np.zeros((1, 2), np.float32), trial is not None
            linear, angular, _ = trial.command
            if time.monotonic() - trial.commanded > self.command_ttl_s:
                linear = angular = 0.0
        action = np.clip([linear / self.limits[0], angular / self.limits[1]], -1.0, 1.0).astype(np.float32)
        return np.tile(action, (self.horizon, 1)), False

    def idle(self) -> bool:
        with self.lock:
            return self.trial is None or time.monotonic() - self.trial.seen > self.idle_s

    def summary(self) -> str:
        with self.lock:
            trial = self.trial
            if trial is None or trial.navigation is None or trial.goal is None:
                return "no GPS goal"
            return f"goal {trial.goal[0]:.6f},{trial.goal[1]:.6f}, {distance_m(trial.navigation[:2], trial.goal):.0f} m away"

    # SDK endpoints (HTTP threads); each returns (status, JSON body)

    def _active(self) -> Trial | None:
        trial = self.trial
        return trial if trial is not None and time.monotonic() - trial.seen <= self.idle_s else None

    def screenshot(self, body=None):
        with self.lock:
            trial = self._active()
            if trial is None or "front" not in trial.frames:
                return 503, {"detail": "No Colosseum trial is running"}
            payload = {f"{view}_frame": image for view, image in trial.frames.items()}
            payload.update({f"{view}_timestamp": trial.captured for view in trial.frames}, timestamp=trial.captured)
            return 200, payload

    def data(self, body=None):
        with self.lock:
            trial = self._active()
            if trial is None or trial.navigation is None:
                return 503, {"detail": "No Colosseum trial is running"}
            latitude, longitude, heading, speed, battery, *rpms = trial.navigation.tolist()
            return 200, {"latitude": latitude, "longitude": longitude, "orientation": heading, "speed": speed,
                         "battery": battery, "rpms": [[*rpms[:4], trial.captured]], "lamp": trial.command[2],
                         "timestamp": trial.captured}

    def control(self, body=None):
        command = (body or {}).get("command") or {}
        try:
            linear, angular, lamp = float(command.get("linear", 0)), float(command.get("angular", 0)), int(command.get("lamp", 0))
        except (TypeError, ValueError):
            return 400, {"detail": "command needs numeric linear, angular and lamp"}
        if not (math.isfinite(linear) and math.isfinite(angular)):
            return 400, {"detail": "command values must be finite"}
        with self.lock:
            trial = self._active()
            if trial is None:
                return 503, {"detail": "No Colosseum trial is running"}
            trial.command = (min(1.0, max(-1.0, linear)), min(1.0, max(-1.0, angular)), lamp)
            trial.commanded = time.monotonic()
        return 200, {"message": "Command sent successfully"}

    def start_mission(self, body=None):
        with self.lock:
            trial = self._active()
            if trial is None:
                return 503, {"detail": "No Colosseum trial is running"}
            trial.started = True
        return 200, {"message": "Mission started successfully"}

    def end_mission(self, body=None):
        with self.lock:
            trial = self._active()
            if trial is not None:
                trial.command, trial.commanded = (0.0, 0.0, 0), time.monotonic()
        return 200, {"message": "Mission ended"}

    def checkpoints(self, body=None):
        with self.lock:
            trial = self._active()
            if trial is None:
                return 503, {"detail": "No Colosseum trial is running"}
            if trial.goal is None:
                return 200, {"checkpoints_list": [], "latest_scanned_checkpoint": 0}
            checkpoint = {"id": 1, "sequence": 1, "latitude": f"{trial.goal[0]:.8f}", "longitude": f"{trial.goal[1]:.8f}"}
            return 200, {"checkpoints_list": [checkpoint], "latest_scanned_checkpoint": 1 if trial.reached else 0}

    def checkpoint_reached(self, body=None):
        with self.lock:
            trial = self._active()
            if trial is None:
                return 503, {"detail": "No Colosseum trial is running"}
            if trial.goal is None or trial.navigation is None:
                return 400, {"detail": {"error": "This trial has no GPS goal"}}
            distance = distance_m(trial.navigation[:2], trial.goal)
            if not trial.reached and distance > self.reach_m:
                return 400, {"detail": {"error": "Bot is not within range of the checkpoint",
                                        "proximate_distance_to_checkpoint": round(distance, 1)}}
            trial.reached = True
        return 200, {"message": "Checkpoint reached", "next_checkpoint_sequence": None, "mission_completed": True}


def handler_for(bridge: Bridge):
    routes = {("GET", "/v2/screenshot"): bridge.screenshot, ("GET", "/v2/front"): bridge.screenshot,
              ("GET", "/data"): bridge.data, ("GET", "/checkpoints-list"): bridge.checkpoints,
              ("POST", "/start-mission"): bridge.start_mission, ("POST", "/end-mission"): bridge.end_mission,
              ("POST", "/control"): bridge.control, ("POST", "/checkpoint-reached"): bridge.checkpoint_reached}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self, method):
            route = routes.get((method, urlparse(self.path).path))
            if route is None:
                return self._send(404, {"detail": "Not Found"})
            body = None
            if method == "POST":
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                except ValueError:
                    return self._send(400, {"detail": "Body must be JSON"})
            self._send(*route(body))

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def log_message(self, format, *args):  # the policy polls several times a second
            return

    return Handler


class PolicyProcess:
    """The policy command, started for each trial; SIGINT lets it stop the rover and write its report."""

    def __init__(self, command: str, cwd: str | None = None) -> None:
        self.argv = shlex.split(command)
        self.cwd = cwd
        self.process: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def exited(self) -> bool:
        return self.process is not None and self.process.poll() is not None

    def start(self) -> None:
        self.stop("a new trial started")
        self.process = subprocess.Popen(self.argv, cwd=self.cwd)

    def stop(self, reason: str) -> None:
        if not self.running:
            return
        print(f"Stopping the policy ({reason}).", flush=True)
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


def run(settings: dict, bridge: Bridge, policy: PolicyProcess | None) -> None:
    """Answer the Router's observations until interrupted, reconnecting after connection failures."""
    while True:
        try:
            with ColosseumPolicySDK(router_url=settings["url"], token=settings["token"], policy_id=settings.get("policy_id", ""),
                                    action_spaces=("base_velocity",), control_hz=CONTROL_HZ, max_horizon=bridge.horizon) as sdk:
                print(f"Connected to Router at {sdk.router_url}. Waiting for an Earth Rover trial...", flush=True)
                announced = False
                while True:
                    try:
                        observation = sdk.get_obs(timeout=1.0)
                    except TimeoutError:
                        if policy is not None and policy.running and bridge.idle():
                            policy.stop("the trial ended")
                        continue
                    if bridge.observe(observation):
                        announced = False
                        print(f"Trial {observation.session_id}: {bridge.summary()}.", flush=True)
                        if policy is not None:
                            policy.start()
                    exited = policy is not None and policy.exited
                    if exited and not announced:
                        print(f"The policy exited (code {policy.process.returncode}); declaring the trial done.", flush=True)
                    chunk, done = bridge.actions(policy_exited=exited)
                    sdk.send_action(chunk, observation=observation, done=done)
                    if done and not announced:
                        announced = True
                        print(f"Declared done: {bridge.summary()}.", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # Router restarts and network drops: keep serving the next trial.
            print(f"Router connection lost ({type(exc).__name__}: {exc}); reconnecting in 5 s.", flush=True)
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Earth Rovers SDK API to a navigation policy from Colosseum trials")
    parser.add_argument("--config", default="configs/policy.yaml", help="YAML with the Router url and the policy token")
    parser.add_argument("--policy-id", default="", help="the Router deployment's policy_id (default: the token's name)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="the SDK's port; policies default to localhost:8000")
    parser.add_argument("--run", default="", help="policy command to start for each trial, e.g. \"python missions/level2.py ...\"")
    parser.add_argument("--cwd", default=None, help="working directory for --run")
    parser.add_argument("--horizon", type=int, default=5, help="actions per chunk at 10 Hz (5 = the Client asks again every 0.5 s)")
    parser.add_argument("--reach-m", type=float, default=REACH_M, help="checkpoint radius in metres")
    parser.add_argument("--command-ttl", type=float, default=3.0, help="seconds a /control command stays valid")
    parser.add_argument("--max-linear", type=float, default=1.0, help="the operator's adapter_config.max_linear")
    parser.add_argument("--max-angular", type=float, default=1.0, help="the operator's adapter_config.max_angular")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)
    if not isinstance(settings, dict) or not settings.get("url") or not settings.get("token"):
        raise SystemExit(f"{args.config} needs url and token")
    if args.policy_id:
        settings["policy_id"] = args.policy_id
    bridge = Bridge(max_linear=args.max_linear, max_angular=args.max_angular, reach_m=args.reach_m,
                    command_ttl_s=args.command_ttl, horizon=args.horizon)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(bridge))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="rover-sdk-bridge-http", daemon=True).start()
    print(f"Earth Rovers SDK API on http://{args.host}:{args.port}", flush=True)
    policy = PolicyProcess(args.run, args.cwd) if args.run else None
    try:
        run(settings, bridge, policy)
    except KeyboardInterrupt:
        pass
    finally:
        if policy is not None:
            policy.stop("the bridge is shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
