from __future__ import annotations

import numpy as np

from colosseum_policy_server import ColosseumPolicySDK


class FakeAsyncSDK:
    router_url = "ws://router:8443"
    server_id = "server-1"
    policy_id = "policy-1"
    session_id = "session-1"
    action_spaces = ("joint_position",)
    control_hz = 15
    max_horizon = 16

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.actions = None

    async def connect(self, *, ssl_context=None) -> None:
        self.connected = True

    async def get_obs(self, *, timeout=None):
        return "latest-observation"

    async def send_action(self, actions, **kwargs) -> None:
        self.actions = np.asarray(actions)
        self.done = kwargs.get("done")

    async def close(self) -> None:
        self.closed = True


def test_sync_sdk_runs_async_transport_on_background_loop():
    transport = FakeAsyncSDK()
    sdk = ColosseumPolicySDK.__new__(ColosseumPolicySDK)
    sdk._initialize(transport)

    with sdk:
        assert transport.connected
        assert sdk.get_obs() == "latest-observation"
        sdk.send_action(np.ones((2, 8), dtype=np.float32))
        np.testing.assert_array_equal(transport.actions, np.ones((2, 8), dtype=np.float32))
        assert transport.done is False
        sdk.send_action(np.zeros((1, 2)), done=True)
        assert transport.done is True

    assert transport.closed
