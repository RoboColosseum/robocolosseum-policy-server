from __future__ import annotations

import numpy as np
import pytest

from colosseum_policy_server import AsyncColosseumPolicySDK, ColosseumPolicySDK
from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.tensors import tensor_from_numpy, tensor_to_numpy


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send(self, data: bytes) -> None:
        self.sent.append(data)


def relay_frame(message_type: int, payload, *, session_id: str, sequence: int = 0) -> pb.RelayFrame:
    return pb.RelayFrame(
        protocol_version=1,
        type=message_type,
        session_id=session_id,
        sequence=sequence,
        payload=payload.SerializeToString(),
    )


@pytest.fixture
def sdk() -> AsyncColosseumPolicySDK:
    instance = AsyncColosseumPolicySDK(
        router_url="ws://127.0.0.1:8443",
        token="pol_test_token",
        server_id="gpu-01",
        policy_id="test-policy",
        policy_revision="v1",
        max_horizon=8,
    )
    instance.connection = FakeConnection()  # type: ignore[assignment]
    return instance


def test_from_yaml_requires_only_url_and_token(tmp_path):
    config = tmp_path / "policy.yaml"
    config.write_text("url: ws://router:8443\ntoken: pol_test\n", encoding="utf-8")
    instance = ColosseumPolicySDK.from_yaml(config)
    assert instance.router_url == "ws://router:8443"
    assert instance.server_id == ""
    assert instance.policy_id == ""
    assert instance.action_spaces == ("joint_position",)
    assert instance.control_hz == 15
    assert instance.max_horizon == 16

    config.write_text("url: ws://router:8443\ntoken: pol_test\nserver_id: forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported keys"):
        ColosseumPolicySDK.from_yaml(config)


@pytest.mark.asyncio
async def test_get_obs_returns_latest_and_converts_tensors(sdk):
    match = pb.MatchResult(session_id="session-1", policy_server_id="gpu-01", policy_id="test-policy")
    await sdk._handle_frame(relay_frame(pb.MATCHED, match, session_id="session-1"))

    first = pb.Observation(control_step=10, instruction="first")
    first.state["joint_position"].CopyFrom(tensor_from_numpy(np.arange(7, dtype=np.float32)))
    second = pb.Observation(control_step=11, instruction="latest")
    second.state["joint_position"].CopyFrom(tensor_from_numpy(np.arange(7, dtype=np.float32) + 1))
    second.sensors.add(sensor_id="camera.wrist", encoding=pb.JPEG, width=640, height=480, data=b"jpeg")

    await sdk._handle_frame(relay_frame(pb.OBSERVATION, first, session_id="session-1", sequence=1))
    await sdk._handle_frame(relay_frame(pb.OBSERVATION, second, session_id="session-1", sequence=2))
    observation = await sdk.get_obs(timeout=0.1)

    assert observation.sequence == 2
    assert observation.control_step == 11
    assert observation.instruction == "latest"
    np.testing.assert_array_equal(observation.state.joints, np.arange(7, dtype=np.float32) + 1)
    assert observation.image("camera.wrist").data == b"jpeg"


@pytest.mark.asyncio
async def test_send_action_correlates_sequence_and_validates_chunk(sdk):
    robot = pb.RobotSpec(
        robot_type="DROID",
        joint_count=7,
        has_gripper=True,
        control_hz=20,
        action_spaces=[pb.ActionSpaceSpec(name="joint_position", action_dim=8)],
    )
    match = pb.MatchResult(
        session_id="session-1",
        policy_server_id="gpu-01",
        policy_id="test-policy",
        robot=robot,
    )
    await sdk._handle_frame(relay_frame(pb.MATCHED, match, session_id="session-1"))
    message = pb.Observation(control_step=20, instruction="move")
    await sdk._handle_frame(relay_frame(pb.OBSERVATION, message, session_id="session-1", sequence=7))
    observation = await sdk.get_obs(timeout=0.1)

    actions = np.ones((4, 8), dtype=np.float64)
    await sdk.send_action(actions)
    sent = pb.RelayFrame.FromString(sdk.connection.sent[-1])  # type: ignore[union-attr]
    plan = pb.ActionPlan.FromString(sent.payload)
    assert sent.type == pb.ACTION_PLAN
    assert sent.sequence == plan.request_sequence == 7
    assert plan.start_step == 20
    assert plan.valid_until_step == 23
    assert plan.control_hz == 20
    assert sdk.robot.joint_count == 7
    assert sdk.selected_action_space == "joint_position"
    assert tensor_to_numpy(plan.actions).dtype == np.float32
    np.testing.assert_array_equal(tensor_to_numpy(plan.actions), actions)

    with pytest.raises(ValueError, match="already sent"):
        await sdk.send_action(actions, observation=observation)


@pytest.mark.asyncio
async def test_send_action_enforces_client_registered_action_dimension(sdk):
    robot = pb.RobotSpec(
        joint_count=6,
        control_hz=10,
        action_spaces=[pb.ActionSpaceSpec(name="joint_position", action_dim=7)],
    )
    match = pb.MatchResult(
        session_id="session-1",
        policy_server_id="gpu-01",
        policy_id="test-policy",
        robot=robot,
    )
    await sdk._handle_frame(relay_frame(pb.MATCHED, match, session_id="session-1"))
    await sdk._handle_frame(
        relay_frame(pb.OBSERVATION, pb.Observation(control_step=1), session_id="session-1", sequence=1)
    )
    observation = await sdk.get_obs(timeout=0.1)

    with pytest.raises(ValueError, match="requires action_dim 7, got 8"):
        await sdk.send_action(np.zeros((2, 8)), observation=observation)


@pytest.mark.asyncio
async def test_send_action_rejects_invalid_chunks(sdk):
    match = pb.MatchResult(session_id="session-1", policy_server_id="gpu-01", policy_id="test-policy")
    await sdk._handle_frame(relay_frame(pb.MATCHED, match, session_id="session-1"))
    message = pb.Observation(control_step=1)
    await sdk._handle_frame(relay_frame(pb.OBSERVATION, message, session_id="session-1", sequence=1))
    observation = await sdk.get_obs(timeout=0.1)

    with pytest.raises(ValueError, match="max_horizon"):
        await sdk.send_action(np.zeros((9, 8)), observation=observation)
    with pytest.raises(ValueError, match="finite"):
        await sdk.send_action(np.full((2, 8), np.nan), observation=observation)


@pytest.mark.asyncio
async def test_send_action_done_requires_zero_motion_and_serializes(sdk):
    robot = pb.RobotSpec(
        robot_type="earth_rover_mini_plus",
        joint_count=0,
        has_gripper=False,
        control_hz=10,
        action_spaces=[pb.ActionSpaceSpec(name="base_velocity", action_dim=2)],
    )
    match = pb.MatchResult(session_id="session-1", policy_server_id="gpu-01", policy_id="rover", robot=robot)
    sdk.action_spaces = ("base_velocity",)
    await sdk._handle_frame(relay_frame(pb.MATCHED, match, session_id="session-1"))
    message = pb.Observation(control_step=3)
    message.state["navigation_state"].CopyFrom(tensor_from_numpy(np.arange(9, dtype=np.float64)))
    await sdk._handle_frame(relay_frame(pb.OBSERVATION, message, session_id="session-1", sequence=1))
    observation = await sdk.get_obs(timeout=0.1)
    assert sdk.selected_action_space == "base_velocity" and sdk.robot.joint_count == 0
    assert observation.raw_state["navigation_state"].dtype == np.float64

    with pytest.raises(ValueError, match="all-zero"):
        await sdk.send_action(np.array([[0.3, 0.0]]), observation=observation, done=True)
    with pytest.raises(ValueError, match="boolean"):
        await sdk.send_action(np.zeros((1, 2)), observation=observation, done="yes")
    await sdk.send_action(np.zeros((2, 2)), observation=observation, done=True)
    plan = pb.ActionPlan.FromString(pb.RelayFrame.FromString(sdk.connection.sent[-1]).payload)
    assert plan.done is True and plan.start_step == 3
    assert pb.ActionPlan().done is False
