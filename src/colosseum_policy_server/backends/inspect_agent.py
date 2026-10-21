"""Stateful Inspect Robots LLM policy behind Colosseum's Local Protocol.

Robot drivers stay on the Client. Optional Inspect dependencies load on session startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from .base import LocalModel
from ..model_adapters.bounds import check_bounds
from ..model_adapters.observation import _rgb, _state
from ..model_adapters.robots import session_robot, llm_robot_contract

def create_backend(options: Mapping[str, Any]) -> "InspectAgentBackend":
    return InspectAgentBackend(options)


class InspectAgentBackend:
    """One isolated agent per exclusive runtime session, with bounded actions."""

    def __init__(self, options: Mapping[str, Any], *, agent_factory=None):
        self.options = dict(options)
        self.agent_factory = agent_factory
        self.agent = None
        self._inflight = None
        self.model = None
        self.bound = False
        self.last_step = -1
        self.stop_reason = None
        self.stop_step = None
        self.stop_detail = None
        self.hold_target = None
        self.last_actions = None

    async def start_session(self, model: LocalModel, request: Mapping[str, Any]) -> None:
        if self.agent is not None or self._inflight is not None:
            raise RuntimeError("previous agent session has not closed")
        settings = {**self.options, **model.backend_options}
        robot_type = session_robot(model, request, self.options)
        self.control_mode = settings.get("control_mode", "joint_pos")
        self.internal_action_dim = model.action_dim
        if self.control_mode == "eef_delta_pose":
            if robot_type != "so101" or model.action_space != "joint_position" or model.action_dim != 6:
                raise ValueError("SO101 EEF requires a 6-D public joint_position contract")
            from inspect_robots.spaces import ActionSemantics
            if not {'agent_eef_target', 'max_chunk_steps'} <= set(ActionSemantics.__dataclass_fields__):
                raise RuntimeError('Installed Inspect Robots lacks SO101 EEF agent semantics')
            self.robot = SimpleNamespace(name="so101")
            self.internal_action_dim = 7
        elif self.control_mode == "joint_pos":
            self.robot = llm_robot_contract(robot_type)
            self.robot.validate(model)
            self.joint_count = self.robot.joint_count
            self.joint_indices = list(self.robot.joint_indices)
            self.gripper_indices = list(self.robot.gripper_indices)
            self.gripper_count = len(self.gripper_indices)
        else:
            raise ValueError(f"unsupported LLM control_mode {self.control_mode!r}")
        identity = urlsplit(model.url)
        typed = model.model_type == "llm"
        providers = {"gpt-": ("openai", "OPENAI_API_KEY"),
                     "grok-": ("x-ai", "XAI_API_KEY"),
                     "claude-": ("anthropic", "ANTHROPIC_API_KEY")}
        if typed:
            provider, key_env, model_prefix = next(
                ((value[0], value[1], prefix) for prefix, value in providers.items()
                 if model.name.startswith(prefix)), (None, None, None))
            if provider is None:
                raise ValueError("Unsupported LLM model name")
            # Router names may include a robot suffix. Keep that exact public
            # identity for matching, but send the provider's model ID to its API.
            api_model = settings.get("api_model", model.name)
            if (not isinstance(api_model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", api_model)
                    or not api_model.startswith(model_prefix)):
                raise ValueError("api_model must be a model ID for the assigned LLM provider")
            identity = urlsplit(f"llm://{provider}/{model.name}")
        if model.url.startswith("https://huggingface.co/"):
            identity = urlsplit("llm://" + str(model.backend_options.get("agent_model", "")))
        wires = {"openai": "responses", "x-ai": "chat", "anthropic": "messages"}
        if identity.scheme != "llm" or identity.netloc not in wires or model.endpoint != "inprocess://inspect-agent":
            raise ValueError("inspect_agent requires an llm://provider/model identity")
        if self.control_mode == "joint_pos":
            self.low = self._vector(settings, "joint_low", self.joint_count)
            self.high = self._vector(settings, "joint_high", self.joint_count)
            self.max_step = self._vector(settings, "joint_max_step", self.joint_count)
            if np.any(self.low >= self.high) or np.any(self.max_step <= 0):
                raise ValueError("joint bounds must increase and joint_max_step must be positive")
            self.open_value = settings.get("gripper_open_value")
            if type(self.open_value) is not int or self.open_value not in {0, 1}:
                raise ValueError("declare gripper_open_value as 0 or 1 for the Client driver")
            if self.robot.name in {'yam', 'so101'} and self.open_value != 1:
                raise ValueError(f'{self.robot.name} requires gripper_open_value: 1')
        default_sensors = ['head_image', 'left_image', 'right_image'] if self.robot.name == 'yam' else ['head_image', 'left_image']
        self.sensors = tuple(settings.get("sensors", default_sensors))
        if self.robot.name == 'yam' and set(self.sensors) != set(default_sensors):
            raise ValueError('YAM requires head_image, left_image and right_image')
        if self.control_mode == "joint_pos":
            self.action_low = self.robot.pack(self.low, np.zeros(self.gripper_count))
            self.action_high = self.robot.pack(
                self.high, np.full(self.gripper_count, self.robot.gripper_high))
            self.action_max_step = self.robot.pack(
                self.max_step, np.full(self.gripper_count, self.robot.gripper_max_step))
        if not self.sensors or any(not isinstance(s, str) or not s for s in self.sensors) or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be nonempty unique sensor names")
        self.docs = str(settings.get("robot_notes", ""))
        if not self.docs.strip() and self.control_mode == "joint_pos":
            raise ValueError("robot_notes must describe the rig and joint directions")
        self.log_dir = Path(settings.get("log_dir", "agent-logs"))
        self.log_id = uuid4().hex  # Never use a peer's run_id as a path.
        self.run_id = str(request["run_id"])
        self.model = model
        self.bound, self.last_step, self.stop_reason = False, -1, None
        self.stop_step = self.stop_detail = self.hold_target = self.last_actions = None
        factory = self.agent_factory
        if factory is None:
            from inspect_robots_agent import LLMAgentPolicy
            factory = LLMAgentPolicy
        # Keep endpoint and credentials provider-owned; keys are read from env
        # by Inspect, never carried in the public model contract or config logs.
        kwargs = {key: settings[key] for key in (
            "effort", "max_llm_calls", "max_output_tokens", "max_speed_frac",
            "images", "image_horizon",
        ) if key in settings}
        kwargs.setdefault("effort", "low")
        if self.control_mode == "eef_delta_pose":
            # Inspect requires a positive integer call budget, but the Router
            # governs trial steps. Its step budget cannot double as a call cap:
            # malformed tool calls can consume calls without moving the robot.
            kwargs["max_llm_calls"] = 2**31 - 1
            kwargs.pop("max_speed_frac", None)
        else:
            kwargs.setdefault("max_llm_calls", 30)
        if typed:
            api_key = request.get("api_key")
            auth = request.get("llm_auth")
            if auth not in (None, "claude-code"):
                raise ValueError("unsupported LLM authentication mode")
            account = auth == "claude-code"
            if account:
                if provider != "anthropic" or api_key is not None:
                    raise ValueError("Claude Code account login requires Anthropic without an API key")
                kwargs.update(
                    env={key: value for key, value in os.environ.items() if key not in {
                        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                        "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID",
                        "ANTHROPIC_ORGANIZATION_ID", "CLAUDE_CODE_USE_BEDROCK",
                        "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                        "CLAUDE_CODE_USE_ANTHROPIC_AWS"}},
                    claude_command=settings.get("claude_command"),
                    claude_timeout_s=settings.get("claude_timeout_s", 120),
                )
            elif not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("Client must supply its API key for this LLM")
            # Per-agent credential environment; never mutate process env or
            # fall back to another Client's / the host's provider credentials.
            base_url = str(settings.get("api_base_url") or model.url).rstrip("/")
            official = {"openai": "https://api.openai.com/v1",
                        "x-ai": "https://api.x.ai/v1",
                        "anthropic": "https://api.anthropic.com/v1"}[provider]
            if settings.get("api_base_url") and base_url != official:
                raise ValueError("api_base_url override must be the selected provider's official API URL")
            if self.control_mode == "eef_delta_pose" and base_url != official:
                raise ValueError("SO101 EEF LLM requires the provider's official API URL")
            if account and base_url != official:
                raise ValueError("Claude Code account login requires the official Anthropic API identity")
            # Inspect appends /messages; Anthropic URLs may be supplied at root.
            if provider == "anthropic" and not urlsplit(base_url).path.strip("/"):
                base_url += "/v1"
            if not account:
                kwargs.update(env={key_env: api_key}, base_url=base_url, api_key_env=key_env)
                auth = settings.get("api_auth", "api_key" if provider == "anthropic" else "bearer")
                if auth not in {"api_key", "bearer"}:
                    raise ValueError("api_auth must be api_key or bearer")
                if provider == "anthropic" and auth == "bearer":
                    from .llm_transport import BearerTransport
                    kwargs["transport"] = BearerTransport(api_key)
        try:
            self.agent = factory(model=(f"anthropic/{api_model}" if typed and account else api_model
                                        if typed else f"{identity.netloc}/{identity.path.lstrip('/')}"),
                                 wire=("claude-code" if typed and account else settings.get("api_format", wires[identity.netloc])),
                                 wire_capture=False, **kwargs)
            if typed and urlsplit(base_url).hostname != "api.openai.com":
                # Pinned Inspect enables explicit GPT cache breakpoints by model
                # name alone. Relays may reject them; keep stateless full history.
                client = getattr(self.agent, "_client", None)
                if hasattr(client, "_cache_anchors"):
                    client._cache_anchors = False
        except Exception:
            if "transport" in kwargs:
                kwargs["transport"].close()
            raise

    @staticmethod
    def _vector(settings, name, size):
        value = np.asarray(settings.get(name), dtype=np.float64)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain {size} finite numbers")
        return value

    async def infer(self, model, observation):
        if self.agent is None or model != self.model:
            raise RuntimeError("agent session was not prepared for this model")
        if self._inflight is not None:
            raise RuntimeError("previous inference is still running")
        # Inspect's HTTP clients are synchronous. Shield the worker so timeout
        # cannot detach a still-mutating conversation from session cleanup.
        task = asyncio.create_task(asyncio.to_thread(self._infer, observation))
        self._inflight = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._inflight = None

    def _infer(self, observation):
        if self.control_mode == "eef_delta_pose":
            return self._infer_so101_eef(observation)
        from inspect_robots.embodiment import EmbodimentInfo
        from inspect_robots.scene import Scene
        from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
        from inspect_robots.types import Observation

        if not observation.instruction.strip() or observation.control_step <= self.last_step:
            raise ValueError("instruction must be nonempty and control_step must advance")
        joints = _state(observation, self.robot.state_key, (self.joint_count,)).astype(np.float64)
        gripper = _state(observation, "gripper_position", (self.gripper_count,)).astype(np.float64)
        check_bounds(self.robot.pack(joints, gripper), self.action_low, self.action_high,
                     self.robot.labels, "observed state is outside the configured rig bounds")
        if self.hold_target is not None:
            return self._hold(joints, observation.control_step)
        raw = {item.sensor_id: item for item in observation.sensors}
        images = {name: _rgb(raw.get(name), name) for name in self.sensors}
        # Inspect uses 1=open. RobotEnv's polarity is explicitly configured.
        position = self.robot.pack(joints, gripper if self.open_value == 1 else 1 - gripper)
        if not self.bound:
            cameras = tuple(CameraSpec(name, img.shape[0], img.shape[1]) for name, img in images.items())
            self.agent.bind(EmbodimentInfo(
                name=self.robot.name, control_hz=self.model.control_hz,
                action_space=Box(shape=(self.robot.action_dim,), low=self.action_low, high=self.action_high,
                    semantics=ActionSemantics(control_mode="joint_pos", gripper="binary" if self.robot.binary_gripper else "continuous",
                        dim_labels=self.robot.labels,
                        max_step=tuple(self.action_max_step))),
                observation_space=ObservationSpace(cameras=cameras,
                    state=StateSpec((StateField("joint_pos", (self.robot.action_dim,), self.robot.state_unit),))),
                docs=self.docs + f"\nAction order: {', '.join(self.robot.labels)}. "
                    f"Gripper: 0 closed, {self.robot.gripper_high:g} open. "
                    "Motion may be truncated to max_horizon; use the next measured state to replan.",
            ))
            self.agent.reset(Scene(id=self.run_id, instruction=observation.instruction))
            self.bound = True
        obs = Observation(images=images, state={"joint_pos": position},
                          instruction=observation.instruction, extra={"env_step": observation.control_step})
        chunk = self.agent.act(obs)
        if chunk.meta.get("request_stop") or any(action.meta.get("request_stop") for action in chunk.actions):
            meta = next((a.meta for a in chunk.actions if a.meta.get("request_stop")), chunk.meta)
            self.stop_reason = meta.get("stop_reason", "give_up")
            if self.stop_reason not in {"done", "give_up"}:
                raise ValueError("invalid agent stop reason")
            self.stop_step = observation.control_step
            self.stop_detail = str(meta.get("stop_detail", ""))
            # Freeze measured joints, not the LLM's proposed stop action. Keep
            # the last executed gripper commands: an obstructed closed
            # gripper's measured width can look open and must not reopen it.
            grip = (gripper > .5).astype(float) if self.robot.binary_gripper else gripper.copy()
            if self.last_actions is not None:
                executed = min(observation.control_step - self.last_step, len(self.last_actions))
                grip = self.last_actions[executed - 1, self.gripper_indices].copy()
            self.hold_target = self.robot.pack(joints, grip).astype(np.float32)
            return self._hold(joints, observation.control_step)
        if chunk.control_hz is not None and chunk.control_hz != self.model.control_hz:
            raise ValueError("agent returned a different control frequency")
        actions = np.asarray([action.data for action in chunk.actions], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.robot.action_dim or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("agent returned malformed actions")
        check_bounds(actions, self.action_low, self.action_high, self.robot.labels,
                     "agent actions exceed rig bounds")
        previous = np.vstack((position, actions[:-1]))
        step_indices = (self.robot.joint_indices + self.robot.gripper_indices
                        if self.robot.name == 'so101' else self.robot.joint_indices)
        step_limits = self.action_max_step[list(step_indices)]
        delta = actions[:, step_indices] - previous[:, step_indices]
        exceeded = np.abs(delta) > step_limits + 1e-7
        if np.any(exceeded):
            details = '; '.join(
                f'action_index={row} {self.robot.labels[step_indices[j]]}: '
                f'previous={previous[row, step_indices[j]]:.6f}, '
                f'target={actions[row, step_indices[j]]:.6f}, '
                f'delta={delta[row, j]:.6f}, limit={step_limits[j]:.6f}'
                for row, j in zip(*np.nonzero(exceeded)))
            kind = 'action' if self.robot.name == 'so101' else 'joint'
            error = f"agent actions exceed per-step {kind} limits: " + details
            logging.getLogger(__name__).error('%s', error)
            raise ValueError(error)
        # Franka uses binary grippers; YAM preserves continuous openings.
        if self.robot.binary_gripper:
            actions[:, self.gripper_indices] = (actions[:, self.gripper_indices] > .5).astype(float)
        if self.open_value == 0:
            actions[:, self.gripper_indices] = 1 - actions[:, self.gripper_indices]
        self.last_step = observation.control_step
        self.last_actions = actions[:self.model.max_horizon].astype(np.float32)
        return self.last_actions.copy()

    def _infer_so101_eef(self, observation):
        from inspect_robots.embodiment import EmbodimentInfo
        from inspect_robots.scene import Scene
        from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
        from inspect_robots.types import Observation

        if not observation.instruction.strip() or observation.control_step <= self.last_step:
            raise ValueError("instruction must be nonempty and control_step must advance")
        joints = _state(observation, "joint_position", (5,))
        gripper = _state(observation, "gripper_position", (1,))
        eef = _state(observation, "eef_state", (7,))
        quat = _state(observation, "eef_quat_wxyz", (4,))
        if not 0 <= gripper[0] <= 100 or not 0 <= eef[6] <= 100 or abs(np.linalg.norm(quat) - 1) > .01:
            raise ValueError("invalid SO101 EEF state")
        raw = {item.sensor_id: item for item in observation.sensors}
        images = {name: _rgb(raw.get(name), name) for name in self.sensors}
        if not self.bound:
            cameras = tuple(CameraSpec(name, img.shape[0], img.shape[1]) for name, img in images.items())
            self.agent.bind(EmbodimentInfo(
                name="so101_eef", control_hz=self.model.control_hz,
                # Inspect's bind requires finite Box bounds. The Router EEF
                # tool installed below does not use these placeholder bounds.
                action_space=Box(shape=(7,), low=-np.ones(7), high=np.ones(7),
                    semantics=ActionSemantics(control_mode="eef_delta_pose", rotation_repr="axis_angle",
                        gripper="continuous", frame="base",
                        dim_labels=("x", "y", "z", "rx", "ry", "rz", "gripper"),
                        agent_eef_target=True, max_chunk_steps=self.model.max_horizon)),
                observation_space=ObservationSpace(cameras=cameras,
                    state=StateSpec((StateField("joint_pos", (6,), "degrees+0_100"),
                        StateField("eef_state", (7,), "m+axis_angle_rad+gripper_0_100"),
                        StateField("eef_quat_wxyz", (4,), "unit_quaternion")))),
                docs=(self.docs + "\nSO101 EEF base-frame positions are metres; rotations are axis-angle "
                    "radians; gripper is 0 closed to 100 open. The arm has five joints and cannot reach "
                    "every six-dimensional pose. Use measured EEF state after each motion. "
                    "Each motion uses at most the Router-assigned max_horizon steps."),
            ))
            from .router_eef_tool import align_router_eef_prompt, install_router_eef_tool
            install_router_eef_tool(self.agent, self.model.max_horizon)
            self.agent.reset(Scene(id=self.run_id, instruction=observation.instruction))
            align_router_eef_prompt(self.agent)
            self.bound = True
        if self.hold_target is not None:
            self.last_step = observation.control_step
            return np.zeros((1, 7), dtype=np.float32)
        obs = Observation(images=images,
            state={"joint_pos": np.r_[joints, gripper], "eef_state": eef, "eef_quat_wxyz": quat},
            instruction=observation.instruction, extra={"env_step": observation.control_step})
        chunk = self.agent.act(obs)
        if chunk.meta.get("request_stop") or any(action.meta.get("request_stop") for action in chunk.actions):
            meta = next((a.meta for a in chunk.actions if a.meta.get("request_stop")), chunk.meta)
            self.stop_reason = meta.get("stop_reason", "give_up")
            if self.stop_reason not in {"done", "give_up"}:
                raise ValueError("invalid agent stop reason")
            self.stop_step = observation.control_step
            self.stop_detail = str(meta.get("stop_detail", ""))
            self.hold_target = np.zeros(7, dtype=np.float32)
            self.last_step = observation.control_step
            return self.hold_target[None, :].copy()
        if chunk.control_hz is not None and chunk.control_hz != self.model.control_hz:
            raise ValueError("agent returned a different control frequency")
        actions = np.asarray([action.data for action in chunk.actions], dtype=np.float64)
        if (actions.ndim != 2 or actions.shape[1] != 7 or not len(actions)
                or len(actions) > self.model.max_horizon or not np.isfinite(actions).all()):
            raise ValueError("agent returned invalid SO101 EEF trajectory")
        self.last_step = observation.control_step
        self.last_actions = actions.astype(np.float32)
        return self.last_actions.copy()

    def _hold(self, joints, control_step):
        """Repeat the latched target until the Client ends the trial.

        Tracking drift must not turn an agent stop request into inference
        failure. _infer still validates observed absolute joint/gripper bounds;
        regular agent actions retain their per-step guards.
        """
        self.last_step = control_step
        return self.hold_target[None, :].copy()

    async def end_session(self):
        # A timed-out HTTP worker must finish before closing its client or
        # allowing another trial into this stateful backend.
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
                task.exception()  # Retrieve late failures after a timeout.
            self._inflight = None
        agent, self.agent = self.agent, None
        if agent is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            record = SimpleNamespace(scene_id="trial", epoch=0, metadata={})
            agent.on_trial_end(record, str(self.log_dir), self.log_id)
            (self.log_dir / f"{self.log_id}.json").write_text(json.dumps({
                "run_id": self.run_id, "model": self.model.public_spec(),
                "robot_type": self.robot.name,
                "stop_reason": self.stop_reason, "stop_step": self.stop_step,
                "stop_detail": self.stop_detail,
                "hold_target": self.hold_target.tolist() if self.hold_target is not None else None,
                "metadata": record.metadata,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            # Inspect 0.29 exposes close on its transport, not its policy.
            # Keep this version-specific access isolated here.
            client = getattr(agent, "_client", None)
            if client is not None:
                client.close()
            self.model = None
            self.hold_target = self.last_actions = None
