"""EEF target tool using Router chunk length without Inspect motion bounds."""
from __future__ import annotations

import math
import re
from types import MethodType

import numpy as np


def _schemas(self):
    schemas = self._router_original_schemas()
    move = next(item["function"] for item in schemas if item["function"]["name"] == "move_eef")
    move["description"] = (
        "Move to an absolute end-effector target in the robot base frame. "
        "Targets may give any of x, y, z in meters, a complete unit quaternion "
        "qw, qx, qy, qz, and/or an absolute gripper target in 0..100. "
        "Omitted coordinates and orientation hold their measured values. "
        "The target is divided evenly over the requested control steps, "
        "up to the Router-assigned chunk length. Observe the result after each motion. "
        "Unreachable poses may be rejected by the robot's IK solver."
    )
    move["parameters"]["properties"]["steps"]["description"] = "Number of control steps for this motion."
    return schemas


def _move_eef_target(self, arguments, observation):
    """Convert an absolute target to deltas without a per-step amplitude cap."""
    from inspect_robots.types import Action
    from inspect_robots_agent._tools import ToolResult

    state = np.asarray(observation.state.get("eef_state"), dtype=np.float64)
    current_quat = np.asarray(observation.state.get("eef_quat_wxyz"), dtype=np.float64)
    if (state.shape != (7,) or current_quat.shape != (4,)
            or not np.isfinite(state).all() or not np.isfinite(current_quat).all()
            or abs(float(np.linalg.norm(current_quat)) - 1) > 1e-3):
        raise ValueError("EEF target control needs finite measured pose and unit quaternion")
    note = arguments.get("note")
    if not isinstance(note, str) or not note.strip():
        return ToolResult(error="note is required: describe the observation and motion")
    steps = arguments.get("steps")
    if type(steps) is not int or not 1 <= steps <= self._max_steps:
        return ToolResult(error=f"steps must be an integer from 1 to {self._max_steps}")
    values = arguments.get("targets")
    if not isinstance(values, dict) or not values:
        return ToolResult(error="targets must contain EEF coordinates, quaternion, or gripper")
    allowed = {"x", "y", "z", "qw", "qx", "qy", "qz", "gripper"}
    if values.keys() - allowed:
        return ToolResult(error=f"targets contain unknown keys {sorted(values.keys() - allowed)}")
    quat_keys = ("qw", "qx", "qy", "qz")
    present = [key for key in quat_keys if key in values]
    if present and len(present) != 4:
        return ToolResult(error="specify all of qw,qx,qy,qz, or omit all four")
    targets = {}
    for key, raw in values.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return ToolResult(error=f"target {key} must be a finite number")
        try:
            targets[key] = float(raw)
        except OverflowError:
            return ToolResult(error=f"target {key} must be a finite number")
        if not math.isfinite(targets[key]):
            return ToolResult(error=f"target {key} must be a finite number")
    position = np.array([targets.get(key, state[i]) for i, key in enumerate(("x", "y", "z"))])
    target_quat = np.array([targets[key] for key in quat_keys]) if present else current_quat.copy()
    if present:
        norm = float(np.linalg.norm(target_quat))
        if abs(norm - 1) > 1e-3:
            return ToolResult(error="qw,qx,qy,qz must be a unit quaternion in wxyz order")
        target_quat /= norm
    gripper = targets.get("gripper", state[6])
    if not 0 <= gripper <= 100:
        return ToolResult(error="gripper target must be within 0..100")
    if present:
        target_w, target_xyz = target_quat[0], target_quat[1:]
        measured_w, measured_xyz = current_quat[0], -current_quat[1:]
        relative_w = target_w * measured_w - float(np.dot(target_xyz, measured_xyz))
        relative_xyz = (target_w * measured_xyz + measured_w * target_xyz
                        + np.cross(target_xyz, measured_xyz))
        if relative_w < 0:
            relative_w, relative_xyz = -relative_w, -relative_xyz
        sin_half = float(np.linalg.norm(relative_xyz))
        rotation = (2 * relative_xyz if sin_half < 1e-8
                    else 2 * math.atan2(sin_half, relative_w) * relative_xyz / sin_half)
    else:
        rotation = np.zeros(3)
    displacement = np.r_[position - state[:3], rotation, gripper - state[6]]
    if not np.isfinite(displacement).all():
        return ToolResult(error="target displacement must be finite")
    per_step = displacement / steps
    return self._success([Action(data=per_step.copy()) for _ in range(steps)], steps)


def install_router_eef_tool(agent, max_horizon):
    """Replace Inspect's bounded EEF conversion after binding its LLM policy."""
    toolset = getattr(agent, "_toolset", None)
    if toolset is None:
        # Test doubles supply their own agent; real Inspect policies have a toolset.
        return
    if not all(hasattr(toolset, key) for key in ("schemas", "_move_eef_target", "_success", "_max_steps")):
        raise RuntimeError("Installed Inspect EEF tool interface changed")
    toolset._max_steps = max_horizon
    toolset._router_original_schemas = toolset.schemas
    toolset.schemas = MethodType(_schemas, toolset)
    toolset._move_eef_target = MethodType(_move_eef_target, toolset)


def align_router_eef_prompt(agent):
    """Remove Inspect rollout claims that do not apply to this local bridge."""
    messages = getattr(agent, "_messages", None)
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return
    system = messages[0].get("content")
    if not isinstance(system, str):
        return
    system = system.replace(
        "Safety approvers clamp out-of-bounds and too-fast actions below you. ",
        "Robot commands must have a valid pose and gripper value. ")
    system = re.sub(r"You have a budget of\s+\d+\s+LLM calls for the whole trial\.",
                    "The Router controls the trial duration.", system)
    messages[0]["content"] = system
