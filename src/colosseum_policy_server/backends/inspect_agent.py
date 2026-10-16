"""Stateful Inspect Robots LLM policy behind Colosseum's Local Protocol.

No robot driver is loaded here. The Client executes Franka commands or bimanual
YAM joint/EE commands. Optional Inspect dependencies load on session startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
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


OFFICIAL_LLM_URLS = {
    'openai': 'https://api.openai.com/v1',
    'x-ai': 'https://api.x.ai/v1',
    'anthropic': 'https://api.anthropic.com/v1',
}
YAM_EEF_LABELS = tuple(f'{side}_{axis}' for side in ('left', 'right')
                       for axis in ('x', 'y', 'z', 'yaw', 'pitch', 'roll', 'gripper'))


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
        self.robot = llm_robot_contract(session_robot(model, request, self.options))
        self.robot.validate(model)
        self.eef_mode = self.robot.name == 'yam' and settings.get('control_mode') == 'eef_abs_pose'
        self.joint_count = self.robot.joint_count
        self.joint_indices = list(self.robot.joint_indices)
        self.gripper_indices = list(self.robot.gripper_indices)
        self.gripper_count = len(self.gripper_indices)
        identity = urlsplit(model.url)
        typed = model.model_type == "llm"
        providers = {"gpt-": ("openai", "OPENAI_API_KEY"),
                     "grok-": ("x-ai", "XAI_API_KEY"),
                     "claude-": ("anthropic", "ANTHROPIC_API_KEY")}
        if typed:
            provider, key_env = next((value for prefix, value in providers.items()
                                      if model.name.startswith(prefix)), (None, None))
            if provider is None:
                raise ValueError("Unsupported LLM model name")
            identity = urlsplit(f"llm://{provider}/{model.name}")
        if model.url.startswith("https://huggingface.co/"):
            identity = urlsplit("llm://" + str(model.backend_options.get("agent_model", "")))
        wires = {"openai": "responses", "x-ai": "chat", "anthropic": "messages"}
        if identity.scheme != "llm" or identity.netloc not in wires or model.endpoint != "inprocess://inspect-agent":
            raise ValueError("inspect_agent requires an llm://provider/model identity")
        self.low = self._vector(settings, "joint_low", self.joint_count)
        self.high = self._vector(settings, "joint_high", self.joint_count)
        self.max_step = self._vector(settings, "joint_max_step", self.joint_count)
        if np.any(self.low >= self.high) or np.any(self.max_step <= 0):
            raise ValueError("joint bounds must increase and joint_max_step must be positive")
        self.open_value = settings.get("gripper_open_value")
        if type(self.open_value) is not int or self.open_value not in {0, 1}:
            raise ValueError("declare gripper_open_value as 0 or 1 for the Client driver")
        if self.robot.name == 'yam' and self.open_value != 1:
            raise ValueError('YAM requires gripper_open_value: 1')
        default_sensors = ['head_image', 'left_image', 'right_image'] if self.robot.name == 'yam' else ['head_image', 'left_image']
        self.sensors = tuple(settings.get("sensors", default_sensors))
        if self.robot.name == 'yam' and set(self.sensors) != set(default_sensors):
            raise ValueError('YAM requires head_image, left_image and right_image')
        self.joint_low = self.robot.pack(self.low, np.zeros(self.gripper_count))
        self.joint_high = self.robot.pack(self.high, np.ones(self.gripper_count))
        self.action_low = (self._vector(settings, 'eef_low', 14) if self.eef_mode else self.joint_low)
        self.action_high = (self._vector(settings, 'eef_high', 14) if self.eef_mode else self.joint_high)
        if np.any(self.action_low > self.action_high):
            raise ValueError('EEF workspace requires eef_low <= eef_high')
        self.action_max_step = self.robot.pack(self.max_step, np.ones(self.gripper_count))
        if not self.sensors or any(not isinstance(s, str) or not s for s in self.sensors) or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be nonempty unique sensor names")
        self.docs = str(settings.get("robot_notes", ""))
        if not self.docs.strip():
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
        kwargs.setdefault("max_llm_calls", 30)
        auth_method = request.get("auth_method", "api-key")
        if auth_method not in {"api-key", "claude-code"}:
            raise ValueError("Unsupported LLM authentication method")
        endpoint = settings.get("llm_endpoint", "router")
        if endpoint not in {"router", "official"}:
            raise ValueError("llm_endpoint must be router or official")
        if auth_method == "claude-code":
            if not typed or provider != "anthropic" or request.get("api_key"):
                raise ValueError("Claude Code login is only valid for Claude without an API key")
            kwargs.update(
                claude_command=settings.get("claude_command"),
                claude_timeout_s=settings.get("claude_timeout_s", 120),
            )
        elif typed:
            api_key = request.get("api_key")
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("Client must supply its API key for this LLM")
            # Per-agent credential environment; never mutate process env or
            # fall back to another Client's / the host's provider credentials.
            # official: the Router URL stays the model identity only, so
            # official keys are never sent to a relay URL in that descriptor.
            base_url = OFFICIAL_LLM_URLS[provider] if endpoint == "official" else model.url.rstrip("/")
            # Inspect appends /messages; Anthropic URLs may be supplied at root.
            if provider == "anthropic" and not urlsplit(base_url).path.strip("/"):
                base_url += "/v1"
            kwargs.update(env={key_env: api_key}, base_url=base_url, api_key_env=key_env)
            auth = settings.get("api_auth", "api_key" if provider == "anthropic" else "bearer")
            if auth not in {"api_key", "bearer"}:
                raise ValueError("api_auth must be api_key or bearer")
            if provider == "anthropic" and auth == "bearer":
                from .llm_transport import BearerTransport
                kwargs["transport"] = BearerTransport(api_key)
        try:
            wire = "claude-code" if auth_method == "claude-code" else settings.get("api_format", wires[identity.netloc])
            self.agent = factory(model=model.name if typed else f"{identity.netloc}/{identity.path.lstrip('/')}",
                                 wire=wire, wire_capture=False, **kwargs)
            if typed and auth_method != "claude-code" and urlsplit(kwargs["base_url"]).hostname != "api.openai.com":
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
        from inspect_robots.embodiment import EmbodimentInfo
        from inspect_robots.scene import Scene
        from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
        from inspect_robots.types import Observation

        if not observation.instruction.strip() or observation.control_step <= self.last_step:
            raise ValueError("instruction must be nonempty and control_step must advance")
        joints = _state(observation, self.robot.state_key, (self.joint_count,)).astype(np.float64)
        gripper = _state(observation, "gripper_position", (self.gripper_count,)).astype(np.float64)
        check_bounds(self.robot.pack(joints, gripper), self.joint_low, self.joint_high,
                     self.robot.labels, "observed state is outside the configured rig bounds")
        eef_state = (_state(observation, 'cartesian_position', (14,)).astype(np.float64)
                     if self.eef_mode else None)
        # FK feedback can drift slightly beyond the command workspace after a
        # bounded IK step; validate its shape/finiteness but guard commands.
        if self.hold_target is not None:
            return self._hold(eef_state if self.eef_mode else joints, observation.control_step)
        raw = {item.sensor_id: item for item in observation.sensors}
        images = {name: _rgb(raw.get(name), name) for name in self.sensors}
        # Inspect uses 1=open. RobotEnv's polarity is explicitly configured.
        position = self.robot.pack(joints, gripper if self.open_value == 1 else 1 - gripper)
        if not self.bound:
            cameras = tuple(CameraSpec(name, img.shape[0], img.shape[1]) for name, img in images.items())
            self.agent.bind(EmbodimentInfo(
                name=self.robot.name, control_hz=self.model.control_hz,
                action_space=Box(shape=(self.robot.action_dim,), low=self.action_low, high=self.action_high,
                    semantics=ActionSemantics(
                        control_mode='eef_abs_pose' if self.eef_mode else 'joint_pos',
                        **({'frame': 'base'} if self.eef_mode else {}),
                        gripper="binary" if self.robot.binary_gripper else "continuous",
                        dim_labels=YAM_EEF_LABELS if self.eef_mode else self.robot.labels,
                        max_step=None if self.eef_mode else tuple(self.action_max_step))),
                observation_space=ObservationSpace(cameras=cameras,
                    state=StateSpec((StateField("joint_pos", (self.robot.action_dim,), "rad+normalized"),
                        *((StateField('eef_state', (14,), 'm+rad+normalized'),) if self.eef_mode else ())))),
                docs=self.docs + f"\nAction order: {', '.join(YAM_EEF_LABELS if self.eef_mode else self.robot.labels)}. Grippers: 0 closed, 1 open. "
                    "Motion may be truncated to max_horizon; use the next measured state to replan."
                    + (" Each arm's EE x/y/z is metres in its own base frame; yaw/pitch/roll "
                       "are radians relative to reset orientation. Verify eef_state after moving."
                       if self.eef_mode else ""),
            ))
            self.agent.reset(Scene(id=self.run_id, instruction=observation.instruction))
            self.bound = True
        obs = Observation(images=images, state={"joint_pos": position,
                          **({'eef_state': eef_state} if self.eef_mode else {})},
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
            self.hold_target = (eef_state.copy() if self.eef_mode else self.robot.pack(joints, grip)).astype(np.float32)
            self.hold_target[self.gripper_indices] = grip
            return self._hold(eef_state if self.eef_mode else joints, observation.control_step)
        if chunk.control_hz is not None and chunk.control_hz != self.model.control_hz:
            raise ValueError("agent returned a different control frequency")
        actions = np.asarray([action.data for action in chunk.actions], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.robot.action_dim or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("agent returned malformed actions")
        check_bounds(actions, self.action_low, self.action_high,
                     YAM_EEF_LABELS if self.eef_mode else self.robot.labels,
                     "agent actions exceed rig bounds")
        previous = np.vstack((position, actions[:-1])) if not self.eef_mode else None
        delta = actions[:, self.joint_indices] - previous[:, self.joint_indices] if previous is not None else None
        exceeded = np.abs(delta) > self.max_step + 1e-7 if delta is not None else np.empty((0, 0), dtype=bool)
        if np.any(exceeded):
            details = '; '.join(
                f'action_index={row} {self.robot.labels[self.joint_indices[j]]}: '
                f'previous={previous[row, self.joint_indices[j]]:.6f}, '
                f'target={actions[row, self.joint_indices[j]]:.6f}, '
                f'delta={delta[row, j]:.6f}, limit={self.max_step[j]:.6f}'
                for row, j in zip(*np.nonzero(exceeded)))
            error = "agent actions exceed per-step joint limits: " + details
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
