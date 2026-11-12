"""Inspect Robots' CaP-X code-as-policy (``capx``) behind Colosseum's Local Protocol.

The model writes Python that calls perception (a local SAM3 server), inverse
kinematics and motion helpers; queued motion comes back as SO-101 absolute
joint targets (five calibrated degrees plus a 0..100 gripper), the same
contract as the joint-mode LLM agent. The robot stays on the Client, so this
backend supplies what the arm adapter provides when capx runs natively: base-
frame tool pose by forward kinematics, an IK solver over the SO-101 URDF and,
when the Client streams it, depth aligned to the color camera plus intrinsics.

A model is routed here when its public name ends in ``-capx`` (for example
``grok-4.7-capx``); the provider is still read from the name prefix.
"""
from __future__ import annotations

import asyncio
import json
import re
import socket
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from .base import LocalModel
from .inspect_agent import OFFICIAL_LLM_URLS
from ..model_adapters.bounds import check_bounds
from ..model_adapters.observation import _rgb, _state
from ..model_adapters.robots import llm_robot_contract, session_robot
from ..tensors import tensor_to_numpy

CAPX_SUFFIX = "-capx"
PROVIDERS = {"gpt-": ("openai", "OPENAI_API_KEY"),
             "grok-": ("x-ai", "XAI_API_KEY"),
             "claude-": ("anthropic", "ANTHROPIC_API_KEY")}
# capx speaks the OpenAI chat or Responses wire; Anthropic is reached through
# its OpenAI-compatible chat endpoint.
DEFAULT_WIRES = {"openai": "responses", "x-ai": "chat", "anthropic": "chat"}
CAPX_OPTIONS = ("effort", "temperature", "max_llm_calls", "max_code_failures", "max_speed_frac",
                "gripper_speed_frac", "image_horizon", "prior_learnings", "transcript_echo",
                "sam3_url", "graspnet_url", "request_timeout_s")
ORIENTATION_NOTE_DEG = 10.0
STOP_REASONS = {"FINISH": "done", "GIVE_UP": "give_up"}


def is_capx(model: LocalModel) -> bool:
    """True for an LLM identity that should run the CaP-X code-as-policy loop."""
    return model.model_type == "llm" and model.name.endswith(CAPX_SUFFIX)


def create_backend(options: Mapping[str, Any]) -> "InspectCapxBackend":
    return InspectCapxBackend(options)


def _check_server(url: str, timeout: float = 3.0) -> None:
    parsed = urlsplit(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", port), timeout=timeout):
            return
    except OSError as exc:
        raise ValueError(f"SAM3 segmentation server is not reachable at {url} ({exc}); start it "
                         "with: servers/sam3/.venv/bin/python -m sam3_server --port 8114 "
                         "(inspect-robots checkout)") from exc


class InspectCapxBackend:
    """One capx policy per exclusive runtime session, with bounded SO-101 joint actions."""

    control_mode = "joint_pos"

    def __init__(self, options: Mapping[str, Any], *, policy_factory=None, kinematics_factory=None,
                 server_check=None):
        self.options = dict(options)
        self.policy_factory = policy_factory
        self.kinematics_factory = kinematics_factory
        self.server_check = server_check
        self.policy = None
        self._inflight = None
        self.model = None
        self.observation_extras: tuple[str, ...] = ()
        self.internal_action_dim = 6

    # ------------------------------------------------------------------ session
    async def start_session(self, model: LocalModel, request: Mapping[str, Any]) -> None:
        if self.policy is not None or self._inflight is not None:
            raise RuntimeError("previous capx session has not closed")
        if not is_capx(model):
            raise ValueError("inspect_capx serves LLM identities ending in -capx")
        settings = {**self.options, **model.backend_options}
        if session_robot(model, request, self.options) != "so101":
            raise ValueError("capx currently supports the SO-101 joint contract only")
        self.robot = llm_robot_contract("so101")
        self.robot.validate(model)
        low = self._vector(settings, "joint_low", 5)
        high = self._vector(settings, "joint_high", 5)
        max_step = self._vector(settings, "joint_max_step", 5)
        if np.any(low >= high) or np.any(max_step <= 0):
            raise ValueError("joint bounds must increase and joint_max_step must be positive")
        if settings.get("gripper_open_value", 1) != 1:
            raise ValueError("so101 requires gripper_open_value: 1")
        self.action_low = self.robot.pack(low, [0.0])
        self.action_high = self.robot.pack(high, [self.robot.gripper_high])
        self.action_max_step = self.robot.pack(max_step, [self.robot.gripper_max_step])
        self.sensors = tuple(settings.get("sensors", ("head_image", "left_image")))
        if not self.sensors or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be nonempty unique sensor names")
        self.camera = settings.get("capx_camera", self.sensors[0])
        if self.camera not in self.sensors:
            raise ValueError("capx_camera must be one of the configured sensors")
        self.docs = str(settings.get("robot_notes", ""))
        if not self.docs.strip():
            raise ValueError("robot_notes must describe the rig and joint directions")
        depth = settings.get("capx_depth", True)
        if type(depth) is not bool:
            raise ValueError("capx_depth must be true or false")
        self.observation_extras = ("depth",) if depth else ()

        urdf = settings.get("urdf_path")
        if not isinstance(urdf, str) or not urdf:
            raise ValueError("capx needs urdf_path: the SO-101 URDF with its meshes "
                             "(see docs/inspect-capx.md)")
        factory = self.kinematics_factory
        if factory is None:
            from inspect_robots_capx.kinematics import ArmKinematics as factory
        self.kinematics = factory(urdf, tip_frame=settings.get("tip_frame", "gripper_frame_link"),
                                  joint_names=self.robot.labels[:5], joint_low_deg=low,
                                  joint_high_deg=high)
        self.ik_tolerance = float(settings.get("ik_position_tolerance_m", 0.005))
        self.ik_weight = float(settings.get("ik_orientation_weight", 0.1))
        if not 0 < self.ik_tolerance < 1 or not 0 < self.ik_weight:
            raise ValueError("ik_position_tolerance_m and ik_orientation_weight must be positive")

        provider, key_env, prefix = next(
            ((value[0], value[1], key) for key, value in PROVIDERS.items()
             if model.name.startswith(key)), (None, None, None))
        if provider is None:
            raise ValueError("Unsupported LLM model name")
        api_model = settings.get("api_model", model.name.removesuffix(CAPX_SUFFIX))
        if (not isinstance(api_model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", api_model)
                or not api_model.startswith(prefix)):
            raise ValueError("api_model must be a model ID for the assigned LLM provider")
        kwargs = {key: settings[key] for key in CAPX_OPTIONS if key in settings}
        # Same defaults as inspect_agent in joint mode, so "-capx" and agent runs of one
        # LLM get the same call budget and per-step joint speed (1x joint_max_step).
        kwargs.setdefault("max_llm_calls", 30)
        kwargs.setdefault("max_speed_frac", 0.1)
        kwargs.setdefault("gripper_speed_frac", 0.5)
        kwargs.setdefault("image_horizon", 2)
        kwargs.setdefault("sam3_url", "http://127.0.0.1:8114")
        kwargs.update(ik="embodiment", camera=self.camera)
        (self.server_check or _check_server)(str(kwargs["sam3_url"]))

        # CaP-X runs on provider API keys only (Claude included); account logins are
        # not part of this backend.
        if request.get("auth_method", "api-key") != "api-key":
            raise ValueError("CaP-X models use the Client's API key; account logins are not supported")
        api_key = request.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Client must supply its API key for this LLM")
        official = OFFICIAL_LLM_URLS[provider]
        endpoint = settings.get("llm_endpoint", "router")
        if endpoint not in {"router", "official"}:
            raise ValueError("llm_endpoint must be router or official")
        base_url = str(settings.get("api_base_url")
                       or (official if endpoint == "official" else model.url)).rstrip("/")
        if settings.get("api_base_url") and base_url != official:
            raise ValueError("api_base_url override must be the selected provider's official API URL")
        if provider == "anthropic" and not urlsplit(base_url).path.strip("/"):
            base_url += "/v1"
        wire = settings.get("capx_wire", DEFAULT_WIRES[provider])
        if wire not in {"chat", "responses"}:
            raise ValueError("capx_wire must be chat or responses")
        kwargs.update(model=api_model, wire=wire, base_url=base_url, api_key_env=key_env,
                      env={key_env: api_key})

        self.log_dir = Path(settings.get("log_dir", "agent-logs"))
        self.log_id = uuid4().hex  # Never use a peer's run_id as a path.
        self.run_id = str(request["run_id"])
        self.model = model
        self.bound, self.last_step = False, -1
        self.stop_reason = self.stop_step = self.stop_detail = None
        self.hold_target = self.last_actions = self.pending = None
        factory = self.policy_factory
        if factory is None:
            from inspect_robots_capx import CapxPolicy as factory
        self.policy = factory(**kwargs)

    @staticmethod
    def _vector(settings, name, size):
        value = np.asarray(settings.get(name), dtype=np.float64)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain {size} finite numbers")
        return value

    # ---------------------------------------------------------------- inference
    async def infer(self, model, observation):
        if self.policy is None or model != self.model:
            raise RuntimeError("capx session was not prepared for this model")
        if self._inflight is not None:
            raise RuntimeError("previous inference is still running")
        task = asyncio.create_task(asyncio.to_thread(self._infer, observation))
        self._inflight = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._inflight = None

    def _solve_ik(self, position, quaternion_wxyz, seed):
        solution = self.kinematics.solve(position, quaternion_wxyz, seed,
                                         position_tolerance_m=self.ik_tolerance,
                                         orientation_weight=self.ik_weight)
        if solution.orientation_error_deg > ORIENTATION_NOTE_DEG:
            print(f"solve_ik note: orientation residual {solution.orientation_error_deg:.0f} deg "
                  "(a five-joint arm cannot reach every orientation; position was kept)",
                  file=sys.stderr)
        return solution.joints

    def _depth_extras(self, observation):
        depth_key, k_key = f"{self.camera}_depth_m", f"{self.camera}_intrinsics"
        if "depth" not in self.observation_extras or depth_key not in observation.state:
            return {}
        depth = np.asarray(tensor_to_numpy(observation.state[depth_key]), dtype=np.float32)
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise ValueError(f"invalid observation state {depth_key}")
        intrinsics = _state(observation, k_key, (9,)).reshape(3, 3).astype(np.float64)
        return {"depth": depth, "intrinsics": intrinsics}

    def _infer(self, observation):
        from inspect_robots.embodiment import EmbodimentInfo
        from inspect_robots.scene import Scene
        from inspect_robots.spaces import (ActionSemantics, Box, CameraSpec, ObservationSpace,
                                           StateField, StateSpec)
        from inspect_robots.types import Observation
        from inspect_robots_capx.kinematics import quaternion_wxyz

        if not observation.instruction.strip() or observation.control_step <= self.last_step:
            raise ValueError("instruction must be nonempty and control_step must advance")
        joints = _state(observation, self.robot.state_key, (5,)).astype(np.float64)
        gripper = _state(observation, "gripper_position", (1,)).astype(np.float64)
        position = self.robot.pack(joints, gripper)
        check_bounds(position, self.action_low, self.action_high, self.robot.labels,
                     "observed state is outside the configured rig bounds")
        if self.hold_target is not None:
            self.last_step = observation.control_step
            return self.hold_target[None, :].copy()
        if self.pending is not None:
            # Keep executing the capx chunk the Router horizon cut short; the
            # model planned it from the observation before the previous slice.
            return self._emit(self.pending, self.last_actions[-1], observation.control_step)
        raw = {item.sensor_id: item for item in observation.sensors}
        images = {name: _rgb(raw.get(name), name) for name in self.sensors}
        pose = self.kinematics.forward(joints)
        if not self.bound:
            cameras = tuple(CameraSpec(name, img.shape[0], img.shape[1]) for name, img in images.items())
            depth_note = (f" obs['depth'] (metres, 0 = no reading) and obs['intrinsics'] belong to "
                          f"camera {self.camera!r}." if self.observation_extras
                          else " No depth is available on this rig.")
            depth_note += (" No camera extrinsics are available (obs has no 'extrinsics'): locate "
                           "objects relative to the gripper and correct each turn.")
            self.policy.bind(EmbodimentInfo(
                name="so101_capx", control_hz=self.model.control_hz,
                action_space=Box(shape=(6,), low=self.action_low, high=self.action_high,
                    semantics=ActionSemantics(control_mode="joint_pos", gripper="continuous",
                        dim_labels=self.robot.labels, max_step=tuple(self.action_max_step))),
                observation_space=ObservationSpace(cameras=cameras, state=StateSpec((
                    StateField("joint_pos", (6,), self.robot.state_unit),
                    StateField("eef_pos", (3,), "m"),
                    StateField("eef_quat_wxyz", (4,), "unit_quaternion")))),
                docs=(self.docs + "\nmove_to_joints takes the five arm joints "
                      "[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] in "
                      "calibrated degrees; the gripper is 0 closed to 100 open. solve_ik(position, "
                      "quaternion_wxyz) solves the gripper_frame_link pose in the robot-base frame "
                      "(metres; x forward, y left, z up), keeps position over orientation, and "
                      "raises when the position is unreachable. The tool's approach axis is its "
                      "local +z; straight down is close to quaternion_wxyz [0, 0, 1, 0]. state "
                      "eef_pos/eef_quat_wxyz are the measured tool pose; obs['eef_rotation'] is "
                      "the same orientation as a 3x3 matrix. No camera-to-base calibration exists: "
                      "locate objects relative to the gripper in the images and correct each turn."
                      + depth_note + " Motion may continue across several control steps; the "
                      "next observation shows the result."),
            ))
            self.policy.reset(Scene(id=self.run_id, instruction=observation.instruction))
            self.bound = True
        extra = {"env_step": observation.control_step, "ik_solver": self._solve_ik,
                 "eef_rotation": pose[:3, :3].copy(), **self._depth_extras(observation)}
        obs = Observation(images=images, instruction=observation.instruction, extra=extra,
                          state={"joint_pos": position, "eef_pos": pose[:3, 3].copy(),
                                 "eef_quat_wxyz": quaternion_wxyz(pose[:3, :3])})
        chunk = self.policy.act(obs)
        stop = next((a.meta for a in chunk.actions if a.meta.get("request_stop")), None)
        if stop is not None or chunk.meta.get("request_stop"):
            meta = stop or chunk.meta
            self.stop_reason = STOP_REASONS.get(str(meta.get("stop_reason", "GIVE_UP")).upper(), "give_up")
            self.stop_step = observation.control_step
            self.stop_detail = str(meta.get("stop_detail", ""))
            grip = (self.last_actions[-1, 5:6].copy() if self.last_actions is not None else gripper)
            self.hold_target = self.robot.pack(joints, grip).astype(np.float32)
            self.last_step = observation.control_step
            return self.hold_target[None, :].copy()
        if chunk.control_hz is not None and chunk.control_hz != self.model.control_hz:
            raise ValueError("capx returned a different control frequency")
        actions = np.asarray([action.data for action in chunk.actions], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 6 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("capx returned malformed actions")
        return self._emit(actions, position, observation.control_step)

    def _emit(self, actions, previous_command, control_step):
        """Send up to max_horizon actions, keeping the rest for the next steps."""
        check_bounds(actions, self.action_low, self.action_high, self.robot.labels,
                     "capx actions exceed rig bounds")
        previous = np.vstack((previous_command, actions[:-1]))
        delta = actions - previous
        exceeded = np.abs(delta) > self.action_max_step + 1e-7
        if np.any(exceeded):
            row, column = (int(i) for i in np.argwhere(exceeded)[0])
            raise ValueError(f"capx actions exceed per-step limits: action_index={row} "
                             f"{self.robot.labels[column]} delta={delta[row, column]:.6f} "
                             f"limit={self.action_max_step[column]:.6f}")
        horizon = self.model.max_horizon
        sent, rest = actions[:horizon], actions[horizon:]
        self.pending = rest if len(rest) else None
        self.last_step = control_step
        self.last_actions = sent.astype(np.float32)
        return self.last_actions.copy()

    async def end_session(self):
        task = self._inflight
        if task is not None:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            self._inflight = None
        policy, self.policy = self.policy, None
        if policy is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{self.log_id}.json").write_text(json.dumps({
                "run_id": self.run_id, "model": self.model.public_spec(), "robot_type": "so101",
                "policy": "capx", "stop_reason": self.stop_reason, "stop_step": self.stop_step,
                "stop_detail": self.stop_detail,
                "transcript": policy.transcript() if hasattr(policy, "transcript") else None,
            }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        finally:
            close = getattr(policy, "close", None)
            if close is not None:
                close()
            self.model = None
            self.hold_target = self.last_actions = self.pending = None
