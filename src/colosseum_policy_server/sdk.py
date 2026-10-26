from __future__ import annotations

import asyncio
import contextlib
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml
from websockets.asyncio.client import ClientConnection, connect as ws_connect
from websockets.exceptions import ConnectionClosed

from . import colosseum_pb2 as pb
from .tensors import tensor_from_numpy, tensor_to_numpy

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class ImageFrame:
    image_id: str
    encoding: str
    capture_time_ns: int
    width: int
    height: int
    data: bytes

    def as_rgb(self) -> np.ndarray:
        """Decode this frame into an ``(height, width, 3)`` RGB uint8 array."""

        if self.encoding == "RAW_RGB":
            expected = self.width * self.height * 3
            if len(self.data) != expected:
                raise ValueError(
                    f"{self.image_id} RAW_RGB has {len(self.data)} bytes; expected {expected}"
                )
            return np.frombuffer(self.data, dtype=np.uint8).reshape(self.height, self.width, 3)
        if self.encoding not in {"JPEG", "PNG"}:
            raise ValueError(f"unsupported image encoding: {self.encoding}")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "JPEG/PNG decoding requires the demo dependency: uv sync --extra demo"
            ) from exc
        image = cv2.imdecode(np.frombuffer(self.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode {self.image_id} as {self.encoding}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ObservationState:
    """Convenient, standardized policy inputs backed by an Observation."""

    _IMAGE_IDS = {
        "left_image": ("left_image", "camera.left", "camera.left_image"),
        "right_image": ("right_image", "camera.right", "camera.right_image"),
        "head_image": ("head_image", "camera.head", "camera.head_image"),
        "front_image": ("front_image", "camera.front", "camera.front_image"),
        "rear_image": ("rear_image", "camera.rear", "camera.rear_image"),
        "goal_image": ("goal_image",),
    }

    def __init__(self, observation: "Observation") -> None:
        self._observation = observation

    def _state(self, *names: str) -> np.ndarray | None:
        for name in names:
            value = self._observation.raw_state.get(name)
            if value is not None:
                return value
        return None

    def _image(self, name: str) -> np.ndarray | None:
        for image_id in self._IMAGE_IDS[name]:
            image = self._observation.image(image_id)
            if image is not None:
                return image.as_rgb()
        return None

    @property
    def joints(self) -> np.ndarray | None:
        return self._state("joint_position", "joints", "observation/joint_position")

    @property
    def gripper(self) -> np.ndarray | None:
        return self._state("gripper_position", "gripper", "observation/gripper_position")

    @property
    def left_image(self) -> np.ndarray | None:
        return self._image("left_image")

    @property
    def right_image(self) -> np.ndarray | None:
        return self._image("right_image")

    @property
    def head_image(self) -> np.ndarray | None:
        return self._image("head_image")

    # Earth Rover Mini+: cameras, the navigation state vector and the goal.
    @property
    def front_image(self) -> np.ndarray | None:
        return self._image("front_image")

    @property
    def rear_image(self) -> np.ndarray | None:
        return self._image("rear_image")

    @property
    def goal_image(self) -> np.ndarray | None:
        return self._image("goal_image")

    @property
    def navigation(self) -> np.ndarray | None:
        """``[latitude, longitude, heading_deg, speed, battery, rpm_1..rpm_4]`` as float64."""
        return self._state("navigation_state")

    @property
    def goal_gps(self) -> np.ndarray | None:
        return self._state("goal_gps")


@dataclass(frozen=True)
class RobotConfiguration:
    robot_type: str
    joint_count: int
    has_gripper: bool
    control_hz: int
    action_spaces: Mapping[str, int]


@dataclass(frozen=True)
class Observation:
    """NumPy-friendly observation returned by the sync or async SDK."""

    session_id: str
    sequence: int
    deadline_ms: int
    robot_time_ns: int
    control_step: int
    instruction: str
    raw_state: Mapping[str, np.ndarray]
    images: tuple[ImageFrame, ...]
    _received_monotonic_ns: int

    def image(self, image_id: str) -> ImageFrame | None:
        return next((image for image in self.images if image.image_id == image_id), None)

    @property
    def state(self) -> ObservationState:
        return ObservationState(self)


class SDKConnectionError(RuntimeError):
    pass


class AsyncColosseumPolicySDK:
    """Pull-based SDK for running inference against the latest robot observation."""

    def __init__(
        self,
        *,
        router_url: str,
        token: str,
        server_id: str = "",
        policy_id: str = "",
        policy_revision: str = "",
        action_spaces: tuple[str, ...] = ("joint_position",),
        control_hz: int = 15,
        max_horizon: int = 16,
    ) -> None:
        if not router_url.startswith(("wss://", "ws://")):
            raise ValueError("router_url must use wss:// or ws://")
        if not token:
            raise ValueError("token is required")
        if control_hz <= 0 or max_horizon <= 0:
            raise ValueError("control_hz and max_horizon must be positive")
        self.router_url = router_url
        self.token = token
        self.server_id = server_id
        self.policy_id = policy_id
        self.policy_revision = policy_revision
        self.action_spaces = action_spaces
        self.control_hz = control_hz
        self.max_horizon = max_horizon
        self.connection: ClientConnection | None = None
        self.session_id = ""
        self.robot: RobotConfiguration | None = None
        self.selected_action_space = ""
        self._observations: asyncio.Queue[Observation | BaseException] = asyncio.Queue(maxsize=1)
        self._last_observation: Observation | None = None
        self._sent_sequences: set[int] = set()
        self._receiver_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AsyncColosseumPolicySDK":
        """Create an SDK from a YAML file with ``url`` and ``token``.

        Optional keys ``action_spaces`` (list of names), ``control_hz`` and
        ``max_horizon`` override the manipulation defaults, e.g. for a rover policy.
        """

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError("SDK config must be a YAML mapping")
        required = {"url", "token"}
        optional = {"action_spaces", "control_hz", "max_horizon"}
        missing = required - config.keys()
        unknown = config.keys() - required - optional
        if missing:
            raise ValueError(f"SDK config is missing: {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"SDK config contains unsupported keys: {', '.join(sorted(unknown))}")
        if not isinstance(config["url"], str) or not isinstance(config["token"], str):
            raise ValueError("SDK config url and token must be strings")
        options = {}
        if "action_spaces" in config:
            spaces = config["action_spaces"]
            if (not isinstance(spaces, list) or not spaces
                    or any(not isinstance(name, str) or not name for name in spaces)):
                raise ValueError("SDK config action_spaces must be a non-empty list of names")
            options["action_spaces"] = tuple(spaces)
        for key in ("control_hz", "max_horizon"):
            if key in config:
                value = config[key]
                if type(value) is not int or value <= 0:
                    raise ValueError(f"SDK config {key} must be a positive integer")
                options[key] = value
        return cls(router_url=config["url"], token=config["token"], **options)

    @staticmethod
    def _frame(
        message_type: int,
        payload=None,
        *,
        session_id: str = "",
        sequence: int = 0,
    ) -> pb.RelayFrame:
        return pb.RelayFrame(
            protocol_version=PROTOCOL_VERSION,
            type=message_type,
            session_id=session_id,
            sequence=sequence,
            sent_at_ns=time.time_ns(),
            payload=payload.SerializeToString() if payload is not None else b"",
        )

    async def connect(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        if self.connection is not None:
            raise RuntimeError("SDK is already connected")
        options = {
            "additional_headers": {"Authorization": f"Bearer {self.token}"},
            "compression": None,
            "max_size": 32 * 1024 * 1024,
            "ping_interval": 10,
            "ping_timeout": 10,
        }
        if ssl_context is not None:
            options["ssl"] = ssl_context
        self.connection = await ws_connect(self.router_url, **options)
        hello = pb.Hello(role=pb.POLICY_SERVER, peer_id=self.server_id, policy_id=self.policy_id)
        await self._send(self._frame(pb.HELLO, hello))
        registered = await self._receive()
        if registered.type == pb.ERROR:
            error = pb.Error.FromString(registered.payload)
            await self.close()
            raise SDKConnectionError(f"{error.code}: {error.message}")
        if registered.type != pb.REGISTERED:
            await self.close()
            raise SDKConnectionError("router did not register policy server")
        registration = pb.Registered.FromString(registered.payload)
        self.server_id = registration.peer_id
        self._receiver_task = asyncio.create_task(self._receiver(), name="colosseum-policy-receiver")

    async def get_obs(self, *, timeout: float | None = None) -> Observation:
        """Return the latest unclaimed observation, dropping older queued observations."""

        if self.connection is None:
            raise SDKConnectionError("SDK is not connected")
        try:
            item = (
                await asyncio.wait_for(self._observations.get(), timeout)
                if timeout is not None
                else await self._observations.get()
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("timed out waiting for an observation") from exc
        if isinstance(item, BaseException):
            raise item
        self._last_observation = item
        return item

    async def send_action(
        self,
        actions: np.ndarray,
        *,
        observation: Observation | None = None,
        plan_id: int | None = None,
        start_step: int | None = None,
        valid_until_step: int | None = None,
        control_hz: int | None = None,
        done: bool = False,
    ) -> None:
        """Send an action chunk correlated with an observation returned by ``get_obs``.

        ``done=True`` declares the task complete (navigation policies that reached the
        goal); the chunk must then be all zeros so the robot stops.
        """

        source = observation or self._last_observation
        if source is None:
            raise RuntimeError("get_obs() must be called before send_action()")
        if self.connection is None or not self.session_id or source.session_id != self.session_id:
            raise SDKConnectionError("observation does not belong to the active session")
        if source.sequence in self._sent_sequences:
            raise ValueError(f"an action was already sent for observation sequence {source.sequence}")
        if source.deadline_ms:
            elapsed_ms = (time.monotonic_ns() - source._received_monotonic_ns) / 1_000_000
            if elapsed_ms > source.deadline_ms:
                raise TimeoutError(
                    f"observation sequence {source.sequence} exceeded its {source.deadline_ms} ms deadline"
                )

        chunk = np.asarray(actions)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] == 0:
            raise ValueError("actions must have shape (horizon, action_dim) or (action_dim,)")
        if chunk.shape[0] > self.max_horizon:
            raise ValueError(f"action horizon {chunk.shape[0]} exceeds max_horizon {self.max_horizon}")
        if self.robot is not None:
            if not self.selected_action_space:
                raise ValueError("policy and robot don't share a compatible action space")
            expected_dimension = self.robot.action_spaces[self.selected_action_space]
            if chunk.shape[1] != expected_dimension:
                raise ValueError(
                    f"{self.selected_action_space} requires action_dim {expected_dimension}, "
                    f"got {chunk.shape[1]}"
                )
        if not np.issubdtype(chunk.dtype, np.number) or not np.all(np.isfinite(chunk)):
            raise ValueError("actions must contain finite numeric values")
        if not isinstance(done, bool):
            raise ValueError("done must be a boolean")
        if done and np.any(chunk != 0):
            raise ValueError("done requires an all-zero action chunk")
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        first_step = source.control_step if start_step is None else start_step
        final_step = first_step + len(chunk) - 1 if valid_until_step is None else valid_until_step
        if final_step < first_step:
            raise ValueError("valid_until_step must be greater than or equal to start_step")

        default_control_hz = self.robot.control_hz if self.robot is not None else self.control_hz
        selected_control_hz = default_control_hz if control_hz is None else control_hz
        if selected_control_hz <= 0:
            raise ValueError("control_hz must be positive")
        plan = pb.ActionPlan(
            request_sequence=source.sequence,
            plan_id=source.sequence if plan_id is None else plan_id,
            start_step=first_step,
            valid_until_step=final_step,
            control_hz=selected_control_hz,
            actions=tensor_from_numpy(chunk),
            done=done,
        )
        await self._send(
            self._frame(
                pb.ACTION_PLAN,
                plan,
                session_id=self.session_id,
                sequence=source.sequence,
            )
        )
        self._sent_sequences.add(source.sequence)

    async def close(self) -> None:
        receiver = self._receiver_task
        self._receiver_task = None
        if receiver is not None:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
        connection = self.connection
        self.connection = None
        if connection is not None:
            await connection.close()
        self._clear_session()

    async def _receiver(self) -> None:
        try:
            assert self.connection is not None
            async for raw in self.connection:
                if isinstance(raw, str):
                    continue
                frame = pb.RelayFrame.FromString(raw)
                await self._handle_frame(frame)
            self._put_latest(SDKConnectionError("router connection closed"))
        except ConnectionClosed as exc:
            self._put_latest(SDKConnectionError(f"router connection closed: {exc}"))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._put_latest(exc)

    async def _handle_frame(self, frame: pb.RelayFrame) -> None:
        if frame.protocol_version != PROTOCOL_VERSION:
            self._put_latest(SDKConnectionError("protocol version mismatch"))
            return
        if frame.type == pb.MATCHED:
            match = pb.MatchResult.FromString(frame.payload)
            self.session_id = match.session_id
            self.policy_id = match.policy_id
            if match.HasField("robot"):
                self.robot = RobotConfiguration(
                    robot_type=match.robot.robot_type,
                    joint_count=match.robot.joint_count,
                    has_gripper=match.robot.has_gripper,
                    control_hz=match.robot.control_hz,
                    action_spaces={spec.name: spec.action_dim for spec in match.robot.action_spaces},
                )
                self.selected_action_space = next(
                    (name for name in self.action_spaces if name in self.robot.action_spaces),
                    "",
                )
            else:
                self.robot = None
                self.selected_action_space = self.action_spaces[0] if self.action_spaces else ""
            self._sent_sequences.clear()
            metadata = pb.SessionReady(
                policy_id=self.policy_id,
                policy_revision=self.policy_revision,
                action_spaces=self.action_spaces,
                control_hz=self.control_hz,
                max_horizon=self.max_horizon,
            )
            await self._send(self._frame(pb.SESSION_READY, metadata, session_id=self.session_id))
        elif frame.type == pb.OBSERVATION and frame.session_id == self.session_id:
            message = pb.Observation.FromString(frame.payload)
            observation = Observation(
                session_id=frame.session_id,
                sequence=frame.sequence,
                deadline_ms=frame.deadline_ms,
                robot_time_ns=message.robot_time_ns,
                control_step=message.control_step,
                instruction=message.instruction,
                raw_state={name: tensor_to_numpy(tensor) for name, tensor in message.state.items()},
                images=tuple(
                    ImageFrame(
                        image_id=sensor.sensor_id,
                        encoding=pb.ImageEncoding.Name(sensor.encoding),
                        capture_time_ns=sensor.capture_time_ns,
                        width=sensor.width,
                        height=sensor.height,
                        data=sensor.data,
                    )
                    for sensor in message.sensors
                ),
                _received_monotonic_ns=time.monotonic_ns(),
            )
            self._put_latest(observation)
        elif frame.type == pb.RESET:
            self._drain_observations()
            self._last_observation = None
            self._sent_sequences.clear()
        elif frame.type == pb.SESSION_CLOSE:
            self._clear_session()
        elif frame.type == pb.ERROR:
            error = pb.Error.FromString(frame.payload)
            self._put_latest(SDKConnectionError(f"{error.code}: {error.message}"))

    def _put_latest(self, item: Observation | BaseException) -> None:
        self._drain_observations()
        self._observations.put_nowait(item)

    def _drain_observations(self) -> None:
        while not self._observations.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._observations.get_nowait()

    def _clear_session(self) -> None:
        self.session_id = ""
        self.robot = None
        self.selected_action_space = ""
        self._last_observation = None
        self._sent_sequences.clear()
        self._drain_observations()

    async def _send(self, frame: pb.RelayFrame) -> None:
        if self.connection is None:
            raise SDKConnectionError("SDK is not connected")
        async with self._send_lock:
            await self.connection.send(frame.SerializeToString())

    async def _receive(self) -> pb.RelayFrame:
        if self.connection is None:
            raise SDKConnectionError("SDK is not connected")
        raw = await self.connection.recv()
        if isinstance(raw, str):
            raise SDKConnectionError("binary protobuf frame required")
        return pb.RelayFrame.FromString(raw)

    async def __aenter__(self) -> "AsyncColosseumPolicySDK":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
