"""Run GeNIE's perception and planner on saved frames (no rover) and write one picture per frame."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import MiniPlusCamera
from .geo import goal_in_robot_frame
from .planning import GeniePlanner, SamTP
from .viz import composite
from .weights import checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan on saved front-camera frames with GeNIE")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-distance", type=float, default=20.0, help="metres")
    parser.add_argument("--goal-bearing", type=float, default=0.0, help="degrees, positive = to the right")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pitch-down", type=float, default=MiniPlusCamera.pitch_down_deg)
    parser.add_argument("--camera-height", type=float, default=MiniPlusCamera.height_m)
    parser.add_argument("--hfov", type=float, default=MiniPlusCamera.hfov_deg)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    camera = MiniPlusCamera(hfov_deg=args.hfov, height_m=args.camera_height, pitch_down_deg=args.pitch_down)
    planner = GeniePlanner(SamTP(checkpoint_path(args.checkpoint), args.device), camera)
    goal = goal_in_robot_frame(args.goal_distance, args.goal_bearing)
    for image_path in args.images:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        started = time.monotonic()
        plan = planner.plan(rgb, *goal, visualize=True)
        elapsed = time.monotonic() - started
        waypoint = "none" if plan.waypoint_deg is None else f"{plan.waypoint_deg:+.0f} deg"
        caption = f"{image_path.name}  {plan.status}  waypoint {waypoint}  front {plan.front_clear:.2f}  {elapsed * 1000:.0f} ms"
        composite(rgb, plan, caption).save(args.output / f"{image_path.stem}-plan.jpg", quality=88)
        print(caption, plan.metadata, flush=True)


if __name__ == "__main__":
    main()
