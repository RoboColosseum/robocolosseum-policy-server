from types import SimpleNamespace
import json
import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.action_plan import make_action_plan, RAW_SOURCE
from colosseum_policy_server.model_adapters.yam_vla import YAMVLAAdapter
from colosseum_policy_server.numpy_wire import _json_array
from test_model_workers import Tensor
from test_yam_groot import runtime as groot_runtime, payload as groot_payload
from test_yam_pi05 import runtime as pi_runtime, payload as pi_payload
from test_yam_g05 import runtime as g05_factory, assets as g05_assets, payload as g05_payload
from test_yam_lingbot import runtime as ling_factory, assets as ling_assets, payload as ling_payload


@pytest.fixture
def g05_runtime(g05_assets, monkeypatch):
    yield from g05_factory.__wrapped__(g05_assets, monkeypatch)


@pytest.fixture
def ling_runtime(ling_assets, monkeypatch):
    return ling_factory.__wrapped__(ling_assets, monkeypatch)


@pytest.mark.parametrize('fixture,payload,size', [
    ('groot_runtime', groot_payload, 16), ('pi_runtime', pi_payload, 30)])
def test_original_chunk_survives_worker_clipping(request, fixture, payload, size):
    worker = request.getfixturevalue(fixture)[0]
    original = np.full((1, size, 14), .2, np.float32)
    original[0, 3, 6], original[0, 4, 13] = -.1, 1.5
    worker.post = lambda _: Tensor(original.copy())
    result = worker.infer(payload())
    np.testing.assert_array_equal(result['raw_actions'], original[0])
    assert result['actions'][3, 6] == 0 and result['actions'][4, 13] == 1
    assert result['raw_action_source'] == RAW_SOURCE


def test_g05_original_chunk_survives_gripper_clipping(g05_runtime):
    worker, _ = g05_runtime
    original = {k: np.full((1, 32, end-start), .2, np.float32)
                for k, start, end in worker.parts}
    original['left_gripper'][0, 4, 0] = -.1
    original['right_gripper'][0, 5, 0] = 1.4
    worker.inferencer.infer = lambda _: [original]
    result = worker.infer(g05_payload())
    assert result['raw_actions'][4, 6] == np.float32(-.1)
    assert result['raw_actions'][5, 13] == np.float32(1.4)
    assert result['actions'][4, 6] == 0 and result['actions'][5, 13] == 1


def test_lingbot_keeps_full_original_when_execution_returns_prefix(ling_runtime):
    worker, _ = ling_runtime
    original = np.full((30, 14), .2, np.float32)
    original[25, 6] = -.2  # Outside the returned/executed prefix.
    worker.return_steps = 3
    worker.policy.infer = lambda _: {'action': original}
    result = worker.infer(ling_payload())
    assert result['actions'].shape == (3, 14)
    np.testing.assert_array_equal(result['raw_actions'], original)


def test_http_adapter_and_plan_keep_full_raw_dtype_and_legacy_wire(monkeypatch):
    original = np.arange(30*14, dtype=np.float64).reshape(30, 14) / 100
    executable = original[:3].copy()
    executable[:, [6, 13]] = np.clip(executable[:, [6, 13]], 0, 1)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps(dict(actions=executable, raw_actions=original,
                raw_action_source=RAW_SOURCE), default=_json_array).encode()
    monkeypatch.setattr('colosseum_policy_server.model_adapters.yam_vla.urlopen', lambda *a, **k: Response())
    model = SimpleNamespace(max_horizon=3, action_dim=14, action_space='joint_position',
                            control_hz=30, endpoint='http://offline.invalid')
    output = YAMVLAAdapter({})._request(model, {})
    plan = make_action_plan(output, model, sequence=7, start_step=8)
    np.testing.assert_array_equal(np.frombuffer(plan.raw_actions.data, np.float64).reshape(30, 14), original)
    # An original deployment reader still sees exactly the executable actions.
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(pb.DESCRIPTOR.serialized_pb)
    message = next(m for m in descriptor.message_type if m.name == 'ActionPlan')
    for index in reversed(range(len(message.field))):
        if message.field[index].number >= 7:
            del message.field[index]
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(descriptor.SerializeToString())
    legacy = message_factory.GetMessageClass(pool.FindMessageTypeByName('colosseum.v1.ActionPlan'))
    received = legacy.FromString(plan.SerializeToString())
    assert received.actions.data == plan.actions.data and received.start_step == 8


def test_raw_provenance_is_required_but_legacy_backends_still_work():
    model = SimpleNamespace(max_horizon=3, action_dim=14, action_space='joint_position', control_hz=30)
    actions = np.zeros((3, 14), np.float32)
    assert not make_action_plan(actions, model, sequence=1, start_step=0).HasField('raw_actions')
    with pytest.raises(ValueError, match='provenance'):
        make_action_plan(dict(actions=actions, raw_actions=actions), model, sequence=1, start_step=0)


async def test_full_original_chunk_crosses_http_backend_and_websocket(tmp_path):
    import threading
    from dataclasses import replace
    from colosseum_policy_server.backends.vla import VLABackend
    from colosseum_policy_server.local_runtime import LocalPolicyRuntime, RuntimeConfig
    from colosseum_policy_server.local_protocol import encode_control, decode_control
    from colosseum_policy_server.model_service import HTTPServer, http_handler
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
    from test_yam_backend import model, observation
    original = np.arange(30*14, dtype=np.float64).reshape(30, 14) / 100
    executable = original[:3].astype(np.float32)
    executable[:, [6, 13]] = np.clip(executable[:, [6, 13]], 0, 1)
    class Worker:
        action_dim = 14
        image_keys = ('top_cam', 'left_cam', 'right_cam')
        def infer(self, payload):
            return dict(actions=executable, raw_actions=original, raw_action_source=RAW_SOURCE)
    http = HTTPServer(('127.0.0.1', 0), http_handler(Worker()))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    item = replace(model(f'http://127.0.0.1:{http.server_port}'), max_horizon=3)
    backend = VLABackend({})
    config = RuntimeConfig('127.0.0.1', 0, 'vla', {}, tmp_path, 1, {item.name: item})
    runtime = LocalPolicyRuntime(config, backend=backend)
    try:
        async with serve(runtime.handle, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}', proxy=None) as ws:
                await ws.send(encode_control(dict(type='prepare', protocol_version=1,
                    run_id='offline-raw', preparation_id='prep', robot_type='yam',
                    model=item.public_spec())))
                assert decode_control(await ws.recv())['type'] == 'ready'
                await ws.send(pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION,
                    session_id='offline-raw', sequence=1, deadline_ms=2000,
                    payload=observation().SerializeToString()).SerializeToString())
                response = pb.RelayFrame.FromString(await ws.recv())
                assert response.type == pb.ACTION_PLAN
                plan = pb.ActionPlan.FromString(response.payload)
                assert list(plan.actions.shape) == [3, 14]
                assert list(plan.raw_actions.shape) == [30, 14]
                assert plan.raw_action_source == RAW_SOURCE
                np.testing.assert_array_equal(np.frombuffer(plan.raw_actions.data, np.float64).reshape(30, 14), original)
                await ws.send(pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE,
                    session_id='offline-raw').SerializeToString())
    finally:
        await backend.end_session()
        http.shutdown()
        thread.join(timeout=3)
        http.server_close()
