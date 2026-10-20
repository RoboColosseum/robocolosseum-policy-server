"""Minimal Earth Rover Mini+ policy: drive straight for a few observations, then declare done.

It shows the navigation observation contract (front camera, navigation_state,
goal_gps or goal_image) and the ``done`` signal. Replace the decision with a
real navigation model; keep the zero chunk with ``done=True`` for completion.
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml

from colosseum_policy_server import ColosseumPolicySDK


def describe(observation) -> str:
    navigation = observation.raw_state.get("navigation_state")
    goal = observation.raw_state.get("goal_gps")
    photo = observation.image("goal_image")
    front = observation.image("front_image")
    return (
        f"step={observation.control_step} front={None if front is None else (front.width, front.height)} "
        f"navigation_state={None if navigation is None else np.round(navigation, 5).tolist()} "
        f"goal_gps={None if goal is None else np.round(goal, 5).tolist()} goal_image={'yes' if photo else 'no'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive the Earth Rover Mini+ forward briefly, then declare done")
    parser.add_argument("--config", default="configs/policy.yaml", help="YAML with url and token")
    parser.add_argument("--steps", type=int, default=20,
                        help="observations to answer with a forward command before declaring done")
    parser.add_argument("--linear", type=float, default=0.3, help="normalized forward speed in [0, 1]")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)
    answered: dict[str, int] = {}
    with ColosseumPolicySDK(router_url=settings["url"], token=settings["token"],
                            action_spaces=("base_velocity",), control_hz=10, max_horizon=10) as sdk:
        print(f"Connected to Router at {sdk.router_url}. Waiting for an Earth Rover observation...", flush=True)
        while True:
            observation = sdk.get_obs()
            count = answered.get(observation.session_id, 0)
            print(describe(observation), flush=True)
            if count < args.steps:
                chunk = np.tile([[args.linear, 0.0]], (sdk.max_horizon, 1))
                sdk.send_action(chunk, observation=observation)
            else:
                sdk.send_action(np.zeros((1, 2)), observation=observation, done=True)
                print("Declared done; the Client stops the rover and ends the trial.", flush=True)
            answered[observation.session_id] = count + 1


if __name__ == "__main__":
    main()
