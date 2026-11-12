"""Drive an Earth Rover Mini+ to its mission checkpoint with GeNIE, through the Earth Rovers SDK HTTP API.

In Colosseum, run it behind ``colosseum-rover-sdk-bridge --run`` (the bridge serves the SDK endpoints from the
trial and starts this program for each trial). It also runs against a real Earth Rovers SDK.

Each run writes ``runs/<time>/log.jsonl`` (one line per control decision) and one picture per plan.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .camera import MiniPlusCamera
from .controller import ControlConfig, Controller, Telemetry
from .geo import HeadingEstimator, bearing_distance
from .weights import checkpoint_path

CONTROL_HZ = 10


class RoverAPI:
    def __init__(self, base_url: str, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def _get(self, path: str) -> dict:
        response = self.session.get(self.base_url + path, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict | None = None) -> requests.Response:
        return self.session.post(self.base_url + path, json=payload or {}, timeout=self.timeout_s)

    def data(self) -> dict:
        return self._get("/data")

    def front(self) -> tuple[np.ndarray, float]:
        payload = self._get("/v2/front")
        image = Image.open(io.BytesIO(base64.b64decode(payload["front_frame"]))).convert("RGB")
        return np.asarray(image), float(payload.get("front_timestamp", payload.get("timestamp", time.time())))

    def checkpoint(self) -> tuple[float, float]:
        self._post("/start-mission")
        checkpoints = self._get("/checkpoints-list").get("checkpoints_list") or []
        if not checkpoints:
            raise RuntimeError("The mission has no checkpoint")
        last = sorted(checkpoints, key=lambda item: int(item.get("sequence", 0)))[-1]
        return float(last["latitude"]), float(last["longitude"])

    def control(self, linear: float, angular: float) -> None:
        self._post("/control", {"command": {"linear": float(linear), "angular": float(angular), "lamp": 0}})

    def checkpoint_reached(self) -> tuple[bool, dict]:
        response = self._post("/checkpoint-reached")
        try:
            body = response.json()
        except ValueError:
            body = {}
        return response.status_code == 200, body


def wait_for_trial(api: RoverAPI, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            api.data()
            api.front()
            return
        except (requests.RequestException, KeyError, ValueError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="GeNIE navigation on the Earth Rover Mini+ (Earth Rovers SDK API)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--yes-i-want-the-rover-to-move", action="store_true", dest="move",
                        help="send /control commands (without it the program only plans and logs)")
    parser.add_argument("--max-seconds", type=float, default=170.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--no-images", action="store_true", help="do not save one picture per plan")
    for name, value in asdict(ControlConfig()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    parser.add_argument("--pitch-down", type=float, default=MiniPlusCamera.pitch_down_deg)
    parser.add_argument("--camera-height", type=float, default=MiniPlusCamera.height_m)
    parser.add_argument("--hfov", type=float, default=MiniPlusCamera.hfov_deg)
    args = parser.parse_args()

    from .planning import GeniePlanner, SamTP  # torch loads here
    from .viz import composite

    run_dir = args.runs_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "log.jsonl").open("a", encoding="utf-8")
    config = ControlConfig(**{name: getattr(args, name) for name in asdict(ControlConfig())})
    camera = MiniPlusCamera(hfov_deg=args.hfov, height_m=args.camera_height, pitch_down_deg=args.pitch_down)
    model = SamTP(checkpoint_path(args.checkpoint), args.device)
    planner = GeniePlanner(model, camera)
    print(f"GeNIE ready: SAM-TP on {model.device}, run log {run_dir}", flush=True)

    api = RoverAPI(args.base_url)
    wait_for_trial(api, timeout_s=60.0)
    goal = api.checkpoint()
    controller = Controller(goal=goal, plan_fn=lambda rgb, x, y: planner.plan(rgb, x, y, visualize=not args.no_images),
                            config=config)
    heading = HeadingEstimator()
    stop = {"requested": False}

    def request_stop(signum, frame):  # the bridge sends SIGINT when the trial ends
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    wall_offset = time.time() - started
    last_reach_try = float("-inf")
    outcome = "time limit"
    plans = 0
    linear = angular = 0.0
    print(f"Goal {goal[0]:.7f},{goal[1]:.7f}; moving: {args.move}", flush=True)
    try:
        while not stop["requested"]:
            tick = time.monotonic()
            if tick - started > args.max_seconds:
                break
            data = api.data()
            lat, lon = float(data["latitude"]), float(data["longitude"])
            forward = linear > 0.05 and abs(angular) < 0.2
            heading_deg = heading.update(tick, lat, lon, float(data["orientation"]), forward=forward)
            telemetry = Telemetry(lat, lon, heading_deg)
            _, distance = bearing_distance(lat, lon, *goal)
            if distance <= config.reach_m and tick - last_reach_try > 2.0:
                last_reach_try = tick
                reached, body = api.checkpoint_reached()
                print(f"[{tick - started:6.1f}s] checkpoint-reached at {distance:.1f} m: {reached} {body}", flush=True)
                if reached:
                    outcome = "reached"
                    break
            try:
                frame, frame_wall = api.front()
                frame_ts = frame_wall - wall_offset
            except (requests.RequestException, KeyError, ValueError):
                frame, frame_ts = None, None
            decision = controller.step(tick, telemetry, frame, frame_ts)
            linear, angular = decision.linear, decision.angular
            if args.move:
                api.control(linear, angular)
            plan = decision.plan
            record = {"t": round(tick - started, 2), "lat": lat, "lon": lon, "compass": data.get("orientation"),
                      "heading": round(heading_deg, 1), "offset": round(heading.offset_deg, 1), "goal_m": round(distance, 1),
                      "state": decision.state, "linear": round(linear, 3), "angular": round(angular, 3), "note": decision.note}
            if plan is not None:
                plans += 1
                record.update(plan=plans, status=plan.status, waypoint_deg=None if plan.waypoint_deg is None else round(plan.waypoint_deg, 1),
                              front_clear=round(plan.front_clear, 2), **plan.metadata)
                if plan.visualization is not None and frame is not None:
                    caption = f"{tick - started:5.1f}s {decision.state} goal {distance:.1f} m | {decision.note}"
                    composite(frame, plan, caption).save(run_dir / f"plan-{plans:04d}.jpg", quality=80)
            if decision.note or plan is not None:
                print(f"[{tick - started:6.1f}s] {decision.state:<12} lin {linear:+.2f} ang {angular:+.2f} | {decision.note}", flush=True)
            log.write(json.dumps(record) + "\n")
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.monotonic() - tick)))
        else:
            outcome = "stopped by the trial"
    finally:
        for _ in range(3):
            try:
                api.control(0.0, 0.0)
            except requests.RequestException:
                pass
        log.close()
        print(f"GeNIE finished: {outcome} after {time.monotonic() - started:.0f} s, {plans} plans; log {run_dir}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
