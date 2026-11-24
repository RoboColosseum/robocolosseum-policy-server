"""Inspect Robots' CaP-X code-as-policy (``capx``) behind Colosseum's Local Protocol.

The model writes Python that calls perception (a local SAM3 server), inverse
kinematics and motion helpers; queued motion comes back as absolute joint
targets in the robot's joint contract, the same as the joint-mode LLM agent:

- SO-101: five calibrated degrees plus a 0..100 gripper.
- Bimanual YAM: left six radians, left 0..1 gripper, right six radians, right
  0..1 gripper. capx drives both arms through its multi-arm profile.

The robot stays on the Client, so this backend supplies what the arm adapter
provides when capx runs natively: base-frame tool poses by forward
kinematics, an IK solver over the arm's URDF and, when the Client streams it,
depth aligned to the color camera plus intrinsics.

A model is routed here when its public name ends in ``-capx`` (for example
``grok-4.7-capx``); the provider is still read from the name prefix.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import shutil
import signal
import socket
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from .base import LocalModel
from .inspect_agent import OFFICIAL_LLM_URLS
from ..local_runtime import _die_with_parent, _libc, _signal_group
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
# Loading sam3.pt from the local cache takes well under a minute; the first
# start may also download it (3.45 GB).
SAM3_START_TIMEOUT_S = 600.0

YAM_SIDES = ("left", "right")
YAM_SENSORS = ("head_image", "left_image", "right_image")
YAM_URDF_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
# i2rt's YAM URDF ends at the gripper mount frame "gripper" (+z = approach).
# Inspect Robots' native YAM capx solves for the LINEAR_4310 grasp_site, the
# midpoint of the fingertips: 0.14465 m along the approach axis, turned 90
# degrees about it so the jaws open along the tool's local +y.
YAM_TOOL_OFFSET_XYZ = (0.0, 0.0, 0.14465)
YAM_TOOL_OFFSET_RPY = (0.0, 0.0, math.pi / 2)
# Arm raised over the table: seeds IK when the queued joints are folded at home.
YAM_READY_SEED = (0.0, 1.05, 1.05, -0.52, 0.0, 0.0)
YAM_JOINT_SPEED_RAD_S = 0.3
YAM_GRIPPER_SPEED_PER_S = 1.0
YAM_GRIPPER_MAX_STEP = 0.15


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
                         "with: servers/sam3/.venv/bin/python -m sam3_server --port 8114, "
                         "or set capx.sam3_launcher so each CaP-X session starts it") from exc


async def _listening(url: str) -> bool:
    parsed = urlsplit(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(parsed.hostname or "127.0.0.1", port), timeout=1)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


class Sam3Launcher:
    """A SAM3 server started for one CaP-X session and stopped when it ends.

    Like a VLA worker it runs in its own process group and dies with this
    server, so its GPU memory is free for whichever model runs next.
    """

    stop_grace_s = 10.0

    def __init__(self, command, url, log_dir, timeout_s):
        self.command, self.url, self.timeout_s = list(command), url, float(timeout_s)
        self.log_path = Path(log_dir) / f"sam3-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}.log"
        self.process = None

    async def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        preexec = _die_with_parent(os.getpid()) if _libc is not None else None
        with self.log_path.open("xb") as log:
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *self.command, stdin=asyncio.subprocess.DEVNULL, stdout=log,
                    stderr=asyncio.subprocess.STDOUT, start_new_session=True, preexec_fn=preexec)
            except OSError as exc:
                raise ValueError(f"sam3_launcher could not run {self.command[0]!r} ({exc}); use an "
                                 "absolute path, or start the Policy Server from the repository "
                                 "root") from exc
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                raise RuntimeError(f"SAM3 server exited with status {self.process.returncode} before "
                                   f"listening at {self.url}; see {self.log_path}")
            if await _listening(self.url):
                return
            await asyncio.sleep(0.5)
        await self.stop()
        raise RuntimeError(f"SAM3 server did not listen at {self.url} within {self.timeout_s:g} s; "
                           f"see {self.log_path}")

    async def stop(self):
        process = self.process
        if process is None or process.returncode is not None:
            return
        _signal_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), self.stop_grace_s)
        except asyncio.TimeoutError:
            _signal_group(process, signal.SIGKILL)
            await process.wait()
        # Only right after reaping it: a long-dead group id may have been reused.
        _signal_group(process, signal.SIGKILL)


def _rotation_rpy(roll, pitch, yaw):
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _base_pose(value, side):
    """Return the 4x4 pose of one YAM arm base in the shared rig frame."""
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (4,) or not np.isfinite(pose).all():
        raise ValueError(f"arm_base_poses.{side} must be [x, y, z, yaw_deg] in metres and degrees")
    transform = np.eye(4)
    transform[:3, :3] = _rotation_rpy(0.0, 0.0, math.radians(pose[3]))
    transform[:3, 3] = pose[:3]
    return transform


def yam_tool_urdf(urdf_path, out_dir, *, mount_frame="gripper", tool_frame="grasp_site",
                  xyz=YAM_TOOL_OFFSET_XYZ, rpy=YAM_TOOL_OFFSET_RPY):
    """Write a copy of a YAM URDF with a fixed ``tool_frame`` link on ``mount_frame``.

    Mesh paths are made absolute, so the copy loads from ``out_dir``. Returns
    the copy's path and the URDF's six arm joint limits (radians).
    """
    source = Path(urdf_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"YAM urdf_path is missing: {source}")
    tree = ET.parse(source)
    root = tree.getroot()
    links = {link.get("name") for link in root.iter("link")}
    if mount_frame not in links:
        raise ValueError(f"YAM URDF has no link {mount_frame!r} to mount the tool frame on")
    if tool_frame in links:
        raise ValueError(f"YAM URDF already has a link {tool_frame!r}")
    limits = {}
    for joint in root.iter("joint"):
        limit = joint.find("limit")
        if joint.get("name") in YAM_URDF_JOINTS and limit is not None:
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    if set(limits) != set(YAM_URDF_JOINTS):
        raise ValueError(f"YAM URDF must define limits for {', '.join(YAM_URDF_JOINTS)}")
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename and "://" not in filename and not Path(filename).is_absolute():
            mesh.set("filename", str((source.parent / filename).resolve()))
    ET.SubElement(root, "link", name=tool_frame)
    joint = ET.SubElement(root, "joint", name=f"{tool_frame}_fixed", type="fixed")
    ET.SubElement(joint, "origin", xyz=" ".join(f"{v:.9g}" for v in xyz),
                  rpy=" ".join(f"{v:.17g}" for v in rpy))
    ET.SubElement(joint, "parent", link=mount_frame)
    ET.SubElement(joint, "child", link=tool_frame)
    target = Path(out_dir) / source.name
    tree.write(target, encoding="unicode")
    low = np.array([limits[name][0] for name in YAM_URDF_JOINTS])
    high = np.array([limits[name][1] for name in YAM_URDF_JOINTS])
    return target, low, high


class InspectCapxBackend:
    """One capx policy per exclusive runtime session, with bounded joint actions."""

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
        self.sam3 = None
        self.observation_extras: tuple[str, ...] = ()
        self.internal_action_dim = 6

    # ------------------------------------------------------------------ session
    async def start_session(self, model: LocalModel, request: Mapping[str, Any]) -> None:
        if self.policy is not None or self._inflight is not None or self.sam3 is not None:
            raise RuntimeError("previous capx session has not closed")
        if not is_capx(model):
            raise ValueError("inspect_capx serves LLM identities ending in -capx")
        settings = {**self.options, **model.backend_options}
        robot = session_robot(model, request, self.options)
        if robot not in {"so101", "yam"}:
            raise ValueError("capx supports the SO-101 and bimanual YAM joint contracts only")
        self.robot = llm_robot_contract(robot)
        self.robot.validate(model)
        self.internal_action_dim = self.robot.action_dim
        if settings.get("gripper_open_value", 1) != 1:
            raise ValueError(f"{robot} requires gripper_open_value: 1")
        if robot == "yam":
            self._start_yam_bounds(settings, model.control_hz)
        else:
            self._start_so101_bounds(settings)
        self.sensors = tuple(settings.get("sensors", YAM_SENSORS if robot == "yam"
                                          else ("head_image", "left_image")))
        if not self.sensors or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be nonempty unique sensor names")
        if robot == "yam" and set(self.sensors) != set(YAM_SENSORS):
            raise ValueError("YAM requires head_image, left_image and right_image")
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

        self.ik_tolerance = float(settings.get("ik_position_tolerance_m", 0.005))
        # Six-joint YAM arms reach most orientations; the five-joint SO-101
        # cannot, so it starts nearly position-only.
        self.ik_weight = float(settings.get("ik_orientation_weight", 1.0 if robot == "yam" else 0.1))
        if not 0 < self.ik_tolerance < 1 or not 0 < self.ik_weight:
            raise ValueError("ik_position_tolerance_m and ik_orientation_weight must be positive")
        if robot == "yam":
            self._start_yam_kinematics(settings)
        else:
            self._start_so101_kinematics(settings)

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
        if robot == "yam":
            # YAM pacing is the declared per-step capx_max_step (joint_speed_rad_s,
            # gripper_speed_per_s); 0.1 makes capx use it unscaled. The shared llm
            # max_speed_frac would otherwise allow the rig's full joint_max_step.
            kwargs.update(max_speed_frac=0.1, gripper_speed_frac=0.1)
        kwargs.update(ik="embodiment", camera=self.camera)
        sam3_url = str(kwargs["sam3_url"])
        sam3_command = self._sam3_command(settings, sam3_url)
        sam3_timeout = self._positive(settings, "sam3_start_timeout_s", SAM3_START_TIMEOUT_S)
        if sam3_command is None:
            (self.server_check or _check_server)(sam3_url)

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
        if sam3_command is not None and not await _listening(sam3_url):
            # Started last, once the session is otherwise valid; end_session stops it.
            self.sam3 = Sam3Launcher(sam3_command, sam3_url, self.log_dir, sam3_timeout)
            await self.sam3.start()
        factory = self.policy_factory
        if factory is None:
            from inspect_robots_capx import CapxPolicy as factory
        self.policy = factory(**kwargs)

    @staticmethod
    def _sam3_command(settings, url):
        """Validate capx.sam3_launcher, the SAM3 server command for each session."""
        command = settings.get("sam3_launcher")
        if command is None:
            return None
        if (not isinstance(command, list) or not command
                or not all(isinstance(part, (str, int)) and not isinstance(part, bool)
                           for part in command)
                or not str(command[0]).strip()):
            raise ValueError("sam3_launcher must be a nonempty list of command arguments")
        command = [str(part) for part in command]
        parsed = urlsplit(url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("sam3_launcher needs a loopback sam3_url")
        if "--port" in command:
            index = command.index("--port")
            if index + 1 >= len(command) or command[index + 1] != str(parsed.port):
                raise ValueError(f"sam3_launcher --port must match the sam3_url port ({parsed.port})")
        return command

    @staticmethod
    def _vector(settings, name, size, default=None):
        value = np.asarray(settings.get(name, default), dtype=np.float64)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain {size} finite numbers")
        return value

    @staticmethod
    def _positive(settings, name, default):
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value <= 0:
            raise ValueError(f"{name} must be a positive number")
        return float(value)

    def _joint_bounds(self, settings):
        count = self.robot.joint_count
        low = self._vector(settings, "joint_low", count)
        high = self._vector(settings, "joint_high", count)
        max_step = self._vector(settings, "joint_max_step", count)
        if np.any(low >= high) or np.any(max_step <= 0):
            raise ValueError("joint bounds must increase and joint_max_step must be positive")
        return low, high, max_step

    def _start_so101_bounds(self, settings):
        low, high, max_step = self._joint_bounds(settings)
        self.joint_low, self.joint_high = low, high
        self.action_low = self.robot.pack(low, [0.0])
        self.action_high = self.robot.pack(high, [self.robot.gripper_high])
        self.action_max_step = self.robot.pack(max_step, [self.robot.gripper_max_step])
        self.capx_max_step = self.action_max_step

    def _start_yam_bounds(self, settings, hz):
        low, high, max_step = self._joint_bounds(settings)
        self.joint_low, self.joint_high = low, high
        gripper_step = self._positive(settings, "gripper_max_step", YAM_GRIPPER_MAX_STEP)
        if gripper_step > 1:
            raise ValueError("gripper_max_step must be in (0, 1]")
        self.action_low = self.robot.pack(low, np.zeros(2))
        self.action_high = self.robot.pack(high, np.ones(2))
        # The rig's limits bound every emitted step; capx paces well below them.
        self.action_max_step = self.robot.pack(max_step, np.full(2, gripper_step))
        joint_step = self._positive(settings, "joint_speed_rad_s", YAM_JOINT_SPEED_RAD_S) / hz
        grip_step = self._positive(settings, "gripper_speed_per_s", YAM_GRIPPER_SPEED_PER_S) / hz
        self.capx_max_step = self.robot.pack(np.minimum(max_step, joint_step),
                                             np.full(2, min(gripper_step, grip_step)))

    def _start_so101_kinematics(self, settings):
        urdf = settings.get("urdf_path")
        if not isinstance(urdf, str) or not urdf:
            raise ValueError("capx needs urdf_path: the SO-101 URDF with its meshes "
                             "(see docs/inspect-capx.md)")
        factory = self.kinematics_factory
        if factory is None:
            from inspect_robots_capx.kinematics import ArmKinematics as factory
        self.kinematics = factory(urdf, tip_frame=settings.get("tip_frame", "gripper_frame_link"),
                                  joint_names=self.robot.labels[:5], joint_low_deg=self.joint_low,
                                  joint_high_deg=self.joint_high)

    def _start_yam_kinematics(self, settings):
        urdf = settings.get("urdf_path")
        if not isinstance(urdf, str) or not urdf:
            raise ValueError("capx needs urdf_path: i2rt's YAM arm URDF with its meshes "
                             "(see docs/inspect-capx.md)")
        xyz = self._vector(settings, "tool_offset_xyz", 3, YAM_TOOL_OFFSET_XYZ)
        rpy = self._vector(settings, "tool_offset_rpy", 3, YAM_TOOL_OFFSET_RPY)
        seed = self._vector(settings, "ik_ready_seed", 6, YAM_READY_SEED)
        poses = settings.get("arm_base_poses")
        if poses is None:
            self.base_poses = None
        elif not isinstance(poses, Mapping) or set(poses) != set(YAM_SIDES):
            raise ValueError("arm_base_poses must map left and right to [x, y, z, yaw_deg]")
        else:
            self.base_poses = {side: _base_pose(poses[side], side) for side in YAM_SIDES}
        factory = self.kinematics_factory
        if factory is None:
            from inspect_robots_capx.kinematics import ArmKinematics as factory
        workdir = tempfile.mkdtemp(prefix="yam-capx-urdf-")
        try:
            path, urdf_low, urdf_high = yam_tool_urdf(urdf, workdir,
                                                      mount_frame=settings.get("tool_mount_frame", "gripper"),
                                                      xyz=xyz, rpy=rpy)
            self.kinematics, self.ik_seed = {}, {}
            for index, side in enumerate(YAM_SIDES):
                arm = slice(6 * index, 6 * index + 6)
                low = np.maximum(self.joint_low[arm], urdf_low)
                high = np.minimum(self.joint_high[arm], urdf_high)
                if np.any(low >= high):
                    raise ValueError(f"{side} joint limits do not overlap the YAM URDF limits")
                # The solver keeps solutions inside the rig limits and the URDF ranges.
                self.kinematics[side] = factory(str(path), tip_frame="grasp_site",
                                                joint_names=YAM_URDF_JOINTS,
                                                joint_low_deg=np.rad2deg(low),
                                                joint_high_deg=np.rad2deg(high))
                self.ik_seed[side] = np.clip(seed, low, high)
        finally:
            # The solver holds its model in memory; the patched copy is not needed.
            shutil.rmtree(workdir, ignore_errors=True)

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

    def _yam_forward(self, side, joints):
        pose = np.asarray(self.kinematics[side].forward(np.rad2deg(joints)), dtype=np.float64)
        return pose if self.base_poses is None else self.base_poses[side] @ pose

    def _yam_solve_ik(self, side, position, quaternion_wxyz, seed):
        """Solve one arm in radians, retrying from the ready pose when the seed is poor."""
        from inspect_robots_capx.kinematics import quaternion_wxyz as to_quaternion

        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if self.base_poses is not None:
            target = np.eye(4)
            target[:3, :3] = _quaternion_matrix(quaternion_wxyz)
            target[:3, 3] = position
            target = np.linalg.inv(self.base_poses[side]) @ target
            position, quaternion_wxyz = target[:3, 3], to_quaternion(target[:3, :3])
        kinematics = self.kinematics[side]
        best, error = None, None
        for start in (np.asarray(seed, dtype=np.float64), self.ik_seed[side]):
            try:
                solution = kinematics.solve(position, quaternion_wxyz, np.rad2deg(start),
                                            position_tolerance_m=self.ik_tolerance,
                                            orientation_weight=self.ik_weight)
            except ValueError as exc:
                error = exc
                continue
            if best is None or solution.orientation_error_deg < best.orientation_error_deg:
                best = solution
            if best.orientation_error_deg <= ORIENTATION_NOTE_DEG:
                break
        if best is None:
            raise error
        if best.orientation_error_deg > ORIENTATION_NOTE_DEG:
            print(f"solve_ik note: {side} orientation residual {best.orientation_error_deg:.0f} deg "
                  "(position was kept; try the jaw direction rotated by 180 degrees)",
                  file=sys.stderr)
        return np.deg2rad(best.joints)

    def _depth_extras(self, observation):
        depth_key, k_key = f"{self.camera}_depth_m", f"{self.camera}_intrinsics"
        if "depth" not in self.observation_extras or depth_key not in observation.state:
            return {}
        depth = np.asarray(tensor_to_numpy(observation.state[depth_key]), dtype=np.float32)
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise ValueError(f"invalid observation state {depth_key}")
        intrinsics = _state(observation, k_key, (9,)).reshape(3, 3).astype(np.float64)
        return {"depth": depth, "intrinsics": intrinsics}

    def _depth_note(self):
        note = (f" obs['depth'] (metres, 0 = no reading) and obs['intrinsics'] belong to "
                f"camera {self.camera!r}." if self.observation_extras
                else " No depth is available on this rig.")
        return note + (" No camera extrinsics are available (obs has no 'extrinsics'): locate "
                       "objects relative to the gripper and correct each turn.")

    def _infer(self, observation):
        if not observation.instruction.strip() or observation.control_step <= self.last_step:
            raise ValueError("instruction must be nonempty and control_step must advance")
        joints = _state(observation, self.robot.state_key, (self.robot.joint_count,)).astype(np.float64)
        gripper = _state(observation, "gripper_position",
                         (len(self.robot.gripper_indices),)).astype(np.float64)
        measured = self.robot.pack(joints, gripper)
        check_bounds(measured, self.action_low, self.action_high, self.robot.labels,
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
        if self.robot.name == "yam":
            obs, position, grip = self._yam_observation(observation, images, joints, gripper)
        else:
            obs, position, grip = self._so101_observation(observation, images, joints, gripper)
        chunk = self.policy.act(obs)
        stop = next((a.meta for a in chunk.actions if a.meta.get("request_stop")), None)
        if stop is not None or chunk.meta.get("request_stop"):
            meta = stop or chunk.meta
            self.stop_reason = STOP_REASONS.get(str(meta.get("stop_reason", "GIVE_UP")).upper(), "give_up")
            self.stop_step = observation.control_step
            self.stop_detail = str(meta.get("stop_detail", ""))
            # Hold the measured joints with the last commanded grippers: an
            # obstructed closed gripper reads open-ish and must not reopen.
            self.hold_target = self.robot.pack(joints, grip).astype(np.float32)
            self.last_step = observation.control_step
            return self.hold_target[None, :].copy()
        if chunk.control_hz is not None and chunk.control_hz != self.model.control_hz:
            raise ValueError("capx returned a different control frequency")
        actions = np.asarray([action.data for action in chunk.actions], dtype=np.float64)
        if (actions.ndim != 2 or actions.shape[1] != self.robot.action_dim or not len(actions)
                or not np.isfinite(actions).all()):
            raise ValueError("capx returned malformed actions")
        return self._emit(actions, position, observation.control_step)

    def _commanded_grippers(self, gripper, control_step):
        """Return the last gripper command the Client executed, else the measured grippers."""
        if self.last_actions is None:
            return gripper
        executed = min(control_step - self.last_step, len(self.last_actions))
        return self.last_actions[max(executed, 1) - 1, list(self.robot.gripper_indices)].astype(np.float64)

    def _so101_observation(self, observation, images, joints, gripper):
        from inspect_robots.embodiment import EmbodimentInfo
        from inspect_robots.scene import Scene
        from inspect_robots.spaces import (ActionSemantics, Box, CameraSpec, ObservationSpace,
                                           StateField, StateSpec)
        from inspect_robots.types import Observation
        from inspect_robots_capx.kinematics import quaternion_wxyz

        position = self.robot.pack(joints, gripper)
        grip = (self.last_actions[-1, 5:6].copy() if self.last_actions is not None else gripper)
        pose = self.kinematics.forward(joints)
        if not self.bound:
            cameras = tuple(CameraSpec(name, img.shape[0], img.shape[1]) for name, img in images.items())
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
                      + self._depth_note() + " Motion may continue across several control steps; the "
                      "next observation shows the result."),
            ))
            self.policy.reset(Scene(id=self.run_id, instruction=observation.instruction))
            self.bound = True
        extra = {"env_step": observation.control_step, "ik_solver": self._solve_ik,
                 "eef_rotation": pose[:3, :3].copy(), **self._depth_extras(observation)}
        obs = Observation(images=images, instruction=observation.instruction, extra=extra,
                          state={"joint_pos": position, "eef_pos": pose[:3, 3].copy(),
                                 "eef_quat_wxyz": quaternion_wxyz(pose[:3, :3])})
        # capx plans from the measured state; stop holds keep the last commanded gripper.
        return obs, position, grip

    def _yam_docs(self):
        frame = ("one shared rig frame in which the two arm bases sit at the configured "
                 "arm_base_poses, so left and right coordinates can be compared directly"
                 if self.base_poses is not None else
                 "that arm's own base frame; the two base frames are not calibrated against "
                 "each other, so never reuse one arm's coordinates for the other")
        steps = self.capx_max_step
        joint_speed = float(steps[list(self.robot.joint_indices)].max()) * self.model.control_hz
        gripper_speed = float(steps[list(self.robot.gripper_indices)].max()) * self.model.control_hz
        return (self.docs + "\nBimanual YAM driven through the multi-arm helpers, with arm "
                "'left' or 'right'. move_to_joints(joints, arm) takes that arm's six joints "
                "[<arm>_joint1 .. <arm>_joint6] in radians; move_arms(left=..., right=...) moves "
                "both arms together; open_gripper(arm)/close_gripper(arm) drive that arm's 0..1 "
                "gripper (1 = open, about 9.5 cm between the jaws). solve_ik(position, "
                "quaternion_wxyz, arm) solves the grasp point between the fingertips in "
                f"{frame} (metres; x forward out of the base, y left, z up) within the joint "
                f"limits. It raises when the position residual exceeds "
                f"{self.ik_tolerance * 1000:.0f} mm and prints a note to stderr when orientation "
                f"misses by more than {ORIENTATION_NOTE_DEG:.0f} degrees. The gripper approach "
                "axis is the tool frame's local +z (wrist toward fingertips), and the jaws open "
                "and close along the tool's local +y. At all-zero joints the tool +z points along "
                "base +x (horizontal, forward); a straight-down approach is close to "
                "quaternion_wxyz [0, 0, 1, 0], and rotating about the approach axis (a yaw of the "
                "whole quaternion) selects the jaw direction; the wrist roll cannot turn a full "
                "half turn, so when IK notes a large orientation residual try the jaw direction "
                "rotated by 180 degrees. state joint_pos is [left joints, left gripper, right "
                "joints, right gripper] with the last commanded gripper values; state "
                "gripper_measured is the measured [left, right] opening, which stays clearly "
                "above its empty-closed value while the jaws hold an object. state "
                "<arm>_eef_pos and <arm>_eef_quat_wxyz report each measured tool pose at the start "
                "of each turn, and obs['<arm>_eef_rotation'] is the same orientation as a matrix. "
                "Cameras: head_image looks over the table, left_image and right_image are the "
                "wrist cameras of those arms." + self._depth_note()
                + f" Queued motion runs at up to {joint_speed:.2f} rad/s per "
                f"joint and {gripper_speed:.2f} gripper units/s, "
                f"{self.model.control_hz:g} steps per second; it may continue across several "
                "control steps, and the next observation shows the result. Keep the two arms "
                "clear of each other: nothing checks for collisions.")

    def _yam_observation(self, observation, images, joints, gripper):
        from inspect_robots.embodiment import EmbodimentInfo
        from inspect_robots.scene import Scene
        from inspect_robots.spaces import (ActionSemantics, Box, CameraSpec, ObservationSpace,
                                           StateField, StateSpec)
        from inspect_robots.types import Observation
        from inspect_robots_capx.kinematics import quaternion_wxyz

        commanded = self._commanded_grippers(gripper, observation.control_step)
        position = self.robot.pack(joints, commanded)
        poses = {side: self._yam_forward(side, joints[6 * i:6 * i + 6])
                 for i, side in enumerate(YAM_SIDES)}
        if not self.bound:
            cameras = tuple(CameraSpec(name, img.shape[0], img.shape[1]) for name, img in images.items())
            eef_fields = tuple(field for side in YAM_SIDES for field in (
                StateField(f"{side}_eef_pos", (3,), "m"),
                StateField(f"{side}_eef_quat_wxyz", (4,), "unit_quaternion")))
            self.policy.bind(EmbodimentInfo(
                name="yam_capx", control_hz=self.model.control_hz,
                action_space=Box(shape=(14,), low=self.action_low, high=self.action_high,
                    semantics=ActionSemantics(control_mode="joint_pos", gripper="continuous",
                        dim_labels=self.robot.labels, max_step=tuple(self.capx_max_step))),
                observation_space=ObservationSpace(cameras=cameras, state=StateSpec((
                    StateField("joint_pos", (14,), self.robot.state_unit),
                    StateField("gripper_measured", (2,), "normalized"), *eef_fields))),
                docs=self._yam_docs(),
            ))
            self.policy.reset(Scene(id=self.run_id, instruction=observation.instruction))
            self.bound = True
        solvers = {side: (lambda position, quaternion, seed, side=side:
                          self._yam_solve_ik(side, position, quaternion, seed))
                   for side in YAM_SIDES}
        state = {"joint_pos": position, "gripper_measured": gripper.copy()}
        extra = {"env_step": observation.control_step, "ik_solver": solvers,
                 **self._depth_extras(observation)}
        for side, pose in poses.items():
            state[f"{side}_eef_pos"] = pose[:3, 3].copy()
            state[f"{side}_eef_quat_wxyz"] = quaternion_wxyz(pose[:3, :3])
            extra[f"{side}_eef_rotation"] = pose[:3, :3].copy()
        obs = Observation(images=images, instruction=observation.instruction, extra=extra,
                          state=state)
        # capx plans and holds with the commanded grippers: a gripper closed on an
        # object reads partly open, and replaying that reading would loosen the grasp.
        return obs, position, commanded

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
        try:
            await self._close_policy()
        finally:
            sam3, self.sam3 = self.sam3, None
            if sam3 is not None:
                await sam3.stop()

    async def _close_policy(self):
        task = self._inflight
        if task is not None:
            if not task.done() and self.policy is not None:
                # The Client has left mid-plan: closing capx's LLM and SAM3
                # clients ends its loop at the next call instead of letting it
                # use its remaining LLM turns while the next session waits.
                with contextlib.suppress(Exception):
                    self.policy.close()
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
                "run_id": self.run_id, "model": self.model.public_spec(),
                "robot_type": self.robot.name,
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


def _quaternion_matrix(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError("quaternion_wxyz must be a unit quaternion")
    w, x, y, z = q / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
