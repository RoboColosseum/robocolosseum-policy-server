"""Remote-inference SDK options: YAML policy identity, robot presets, keepalive, reconnect."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from websockets.asyncio.server import serve

from colosseum_policy_server import AsyncColosseumPolicySDK, ColosseumPolicySDK, StaleObservationError
from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.tensors import tensor_from_numpy


def write(tmp_path, text):
    path = tmp_path / "policy.yaml"
    path.write_text("url: ws://router:8443\ntoken: pol_test\n" + text, encoding="utf-8")
    return path


def test_yaml_sets_policy_identity_and_keepalive(tmp_path):
    sdk = AsyncColosseumPolicySDK.from_yaml(write(tmp_path, (
        "policy_id: acme-yam-v1\npolicy_revision: r2\nping_interval: 30\nping_timeout: 120\nreconnect: true\n")))
    assert (sdk.policy_id, sdk.policy_revision) == ("acme-yam-v1", "r2")
    assert (sdk.ping_interval, sdk.ping_timeout, sdk.reconnect) == (30, 120, True)


def test_yaml_defaults_are_unchanged(tmp_path):
    sdk = AsyncColosseumPolicySDK.from_yaml(write(tmp_path, ""))
    assert (sdk.policy_id, sdk.reconnect, sdk.ping_interval, sdk.ping_timeout) == ("", False, 20, 60)


def test_robot_preset_fills_contract_and_explicit_keys_win(tmp_path):
    sdk = AsyncColosseumPolicySDK.from_yaml(write(tmp_path, "robot: yam\n"))
    assert (sdk.action_spaces, sdk.control_hz, sdk.max_horizon) == (("joint_position",), 30, 50)
    sdk = AsyncColosseumPolicySDK.from_yaml(write(tmp_path, "robot: yam\nmax_horizon: 32\n"))
    assert sdk.max_horizon == 32
    sdk = AsyncColosseumPolicySDK.from_yaml(write(tmp_path, "robot: earth_rover_mini_plus\n"))
    assert (sdk.action_spaces, sdk.control_hz) == (("base_velocity",), 10)


@pytest.mark.parametrize("text, message", [
    ("robot: unicycle\n", "robot must be one of"),
    ("policy_id: 7\n", "policy_id must be a string"),
    ("ping_timeout: 0\n", "ping_timeout must be a positive"),
    ("reconnect: yes-please\n", "reconnect must be true or false"),
])
def test_yaml_rejects_invalid_options(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        AsyncColosseumPolicySDK.from_yaml(write(tmp_path, text))


class FakeConnection:
    async def send(self, data):
        pass


@pytest.mark.asyncio
async def test_action_for_a_closed_session_is_a_stale_observation():
    sdk = AsyncColosseumPolicySDK(router_url="ws://x", token="pol_t", policy_id="p")
    sdk.connection = FakeConnection()
    frame = lambda kind, payload, seq=0: pb.RelayFrame(protocol_version=1, type=kind, session_id="s1",
                                                       sequence=seq, payload=payload.SerializeToString())
    await sdk._handle_frame(frame(pb.MATCHED, pb.MatchResult(session_id="s1", policy_id="p")))
    await sdk._handle_frame(frame(pb.OBSERVATION, pb.Observation(control_step=0), 1))
    observation = await sdk.get_obs(timeout=1)
    await sdk._handle_frame(pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id="s1"))
    with pytest.raises(StaleObservationError):
        await sdk.send_action(np.zeros((1, 8)), observation=observation)


def router_frame(kind, payload=None, *, session_id="", sequence=0):
    return pb.RelayFrame(protocol_version=1, type=kind, session_id=session_id, sequence=sequence,
                         payload=payload.SerializeToString() if payload is not None else b"").SerializeToString()


@pytest.mark.asyncio
async def test_reconnects_after_router_drop_and_keeps_serving():
    connections = []

    async def router(ws):
        hello = pb.Hello.FromString(pb.RelayFrame.FromString(await ws.recv()).payload)
        connections.append(hello.policy_id)
        await ws.send(router_frame(pb.REGISTERED, pb.Registered(peer_id=f"peer-{len(connections)}")))
        if len(connections) == 1:
            await ws.close()  # Router restart
            return
        await ws.send(router_frame(pb.MATCHED, pb.MatchResult(session_id="s2", policy_id=hello.policy_id),
                                   session_id="s2"))
        assert pb.RelayFrame.FromString(await ws.recv()).type == pb.SESSION_READY
        observation = pb.Observation(control_step=0, instruction="after reconnect")
        observation.state["joint_position"].CopyFrom(tensor_from_numpy(np.zeros(7, np.float32)))
        await ws.send(router_frame(pb.OBSERVATION, observation, session_id="s2", sequence=1))
        reply = pb.RelayFrame.FromString(await ws.recv())
        assert reply.type == pb.ACTION_PLAN and reply.sequence == 1
        await ws.wait_closed()

    async with serve(router, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        sdk = AsyncColosseumPolicySDK(router_url=f"ws://127.0.0.1:{port}", token="pol_t",
                                      policy_id="acme", reconnect=True)
        await sdk.connect()
        try:
            observation = await sdk.get_obs(timeout=10)
            assert observation.instruction == "after reconnect"
            await sdk.send_action(np.zeros((4, 8)), observation=observation)
            assert connections == ["acme", "acme"] and sdk.server_id == "peer-2"
        finally:
            await sdk.close()


@pytest.mark.asyncio
async def test_without_reconnect_a_drop_is_reported():
    async def router(ws):
        await ws.recv()
        await ws.send(router_frame(pb.REGISTERED, pb.Registered(peer_id="peer")))
        await ws.close()

    async with serve(router, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        sdk = AsyncColosseumPolicySDK(router_url=f"ws://127.0.0.1:{port}", token="pol_t")
        await sdk.connect()
        try:
            with pytest.raises(Exception, match="router connection closed"):
                await sdk.get_obs(timeout=5)
        finally:
            await sdk.close()


def test_sync_sdk_forwards_remote_options():
    sdk = ColosseumPolicySDK(router_url="ws://x", token="pol_t", policy_id="p",
                             ping_interval=5, ping_timeout=90, reconnect=True)
    inner = sdk._async_sdk
    assert (inner.ping_interval, inner.ping_timeout, inner.reconnect) == (5, 90, True)
