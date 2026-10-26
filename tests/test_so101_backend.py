from dataclasses import replace
import threading

import msgpack
import numpy as np
import pytest
import websockets

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.vla import VLABackend
from colosseum_policy_server.model_adapters.so101_vla import LEGACY_OFFSETS, LEGACY_SIGNS, SO101VLAAdapter
from colosseum_policy_server.model_service import HTTPServer, http_handler, parse_args
from colosseum_policy_server.numpy_wire import _msgpack_default, _msgpack_object
from colosseum_policy_server.tensors import tensor_from_numpy

STATE = np.array([10, -20, 30, 40, -50, 60], np.float32)
LEGACY_STATE = np.array([10, 110, 120, 40, -50, 60], np.float32)


def model(adapter='molmoact2_so101', endpoint='http://127.0.0.1:9100', **options):
    return LocalModel.from_mapping(dict(name=adapter, url='https://huggingface.co/example/so101',
        revision='a'*40, action_space='joint_position', action_dim=6, control_hz=30,
        max_horizon=30, endpoint=endpoint, backend_options={'adapter': adapter, 'robot_type': 'so101', **options}))


def observation(instruction='pick'):
    return pb.Observation(instruction=instruction, state={
        'joint_position': tensor_from_numpy(STATE[:5]),
        'gripper_position': tensor_from_numpy(STATE[5:]),
    }, sensors=[pb.Image(sensor_id=name, encoding=pb.RAW_RGB, height=2, width=3,
                        data=np.full((2, 3, 3), i + 1, np.uint8).tobytes())
                for i, name in enumerate(('head_image', 'left_image'))])


@pytest.fixture
def worker():
    servers = []
    def start(horizon, expected_state=LEGACY_STATE):
        class Runtime:
            action_dim = 6
            image_keys = ('external_cam', 'wrist_cam')
            def infer(self, payload):
                np.testing.assert_array_equal(payload['state'], expected_state)
                assert payload['external_cam'].shape == (2, 3, 3) and payload['external_cam'].dtype == np.uint8
                assert [payload[key][0, 0, 0] for key in self.image_keys] == [1, 2]
                assert payload['instruction'] == 'pick' and payload['num_steps'] == 10
                return {'actions': payload['state'] + np.arange(horizon, dtype=np.float32)[:, None]}
        server = HTTPServer(('127.0.0.1', 0), http_handler(Runtime()))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f'http://127.0.0.1:{server.server_port}'
    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('adapter,horizon,expected', [('molmoact2_so101', 30, 30), ('pi05_so101', 50, 30),
                                                      ('groot_so101', 40, 30)])
async def test_http_roundtrip_routing_and_chunk_truncation(worker, adapter, horizon, expected):
    backend = VLABackend({})
    item = model(adapter, worker(horizon))
    await backend.start_session(item, {'robot_type': 'so101'})
    actions = await backend.infer(item, observation())
    assert actions.shape == (expected, 6)
    # The worker echoes legacy-frame state + k; shoulder_lift is mirrored on the way back.
    np.testing.assert_array_equal(actions[0], STATE)
    np.testing.assert_array_equal(actions[-1], STATE + (expected - 1) * LEGACY_SIGNS)
    assert actions.dtype == np.float32
    await backend.end_session()


async def test_current_frame_checkpoint_passes_units_through(worker):
    item = model('molmoact2_so101', worker(2, STATE), joint_frame='current')
    actions = await SO101VLAAdapter({}).infer(item, observation())
    np.testing.assert_array_equal(actions, STATE + np.arange(2, dtype=np.float32)[:, None])


async def test_g05_converts_joint_frame_pads_camera_and_drains_cached_chunk():
    requests = []
    async def handler(socket):
        await socket.send(msgpack.packb({'action_steps': 4}))
        step = 0
        async for message in socket:
            request = msgpack.unpackb(message, object_hook=_msgpack_object, raw=False, strict_map_key=False)
            requests.append(request)
            step += 1
            action = LEGACY_SIGNS * (STATE + step) + LEGACY_OFFSETS
            await socket.send(msgpack.packb({'action': {'right_arm': action}, 'need_obs': step >= 4},
                                            default=_msgpack_default))
    async with websockets.serve(handler, '127.0.0.1', 0) as server:
        port = server.sockets[0].getsockname()[1]
        item = model('g05_so101', f'ws://127.0.0.1:{port}')
        backend = VLABackend({})
        await backend.start_session(item, {'robot_type': 'so101'})
        actions = await backend.infer(item, observation())
        await backend.end_session()
    np.testing.assert_allclose(actions, STATE + np.arange(1, 5, dtype=np.float32)[:, None])
    first = requests[0]
    np.testing.assert_allclose(first['state']['right_arm'], [10, 110, 120, 40, -50, 60])
    assert first['task'] == 'pick' and first['embodiment_type'] == 'so100' and first['frequency'] == 30.0
    assert set(first['images']) == {'exterior', 'wrist_right', 'wrist_left'}
    assert first['images']['exterior'].shape == (3, 2, 3) and first['images']['exterior'][0, 0, 0] == 1
    assert first['images']['wrist_right'][0, 0, 0] == 2 and not first['images']['wrist_left'].any()
    assert requests[1:] == [{}, {}, {}]


async def test_g05_stops_at_max_horizon_and_reports_service_errors():
    async def handler(socket):
        await socket.send(msgpack.packb({'action_steps': 16}))
        async for message in socket:
            if msgpack.unpackb(message, raw=False, strict_map_key=False).get('task') == 'fail':
                await socket.send(msgpack.packb({'error': {'code': 400, 'message': 'bad'}}))
            else:
                await socket.send(msgpack.packb({'action': {'right_arm': np.zeros(6, np.float32)}, 'need_obs': False},
                                                default=_msgpack_default))
    async with websockets.serve(handler, '127.0.0.1', 0) as server:
        item = replace(model('g05_so101', f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"), max_horizon=3)
        adapter = SO101VLAAdapter({})
        assert (await adapter.infer(item, observation())).shape == (3, 6)
        with pytest.raises(ValueError, match='no valid action'):
            await adapter.infer(item, observation('fail'))


@pytest.mark.parametrize('change,match', [
    (dict(action_dim=8), '6-D joint_position'),
    (dict(action_space='joint_velocity'), '6-D joint_position'),
    (dict(endpoint='ws://127.0.0.1:9100'), 'http:// endpoint'),
    (dict(backend_options={'adapter': 'g05_so101'}), 'ws:// endpoint'),
    (dict(backend_options={'adapter': 'molmoact2_so101', 'num_steps': 0}), 'num_steps'),
    (dict(backend_options={'adapter': 'molmoact2_so101', 'joint_frame': 'radians'}), 'joint_frame'),
    (dict(endpoint='ws://127.0.0.1:9100', backend_options={'adapter': 'g05_so101', 'padded_cameras': 'wrist_left'}), 'padded_cameras'),
])
async def test_contract_checked_during_preparation(change, match):
    backend = VLABackend({})
    with pytest.raises(ValueError, match=match):
        await backend.start_session(replace(model(), **change), {'robot_type': 'so101'})
    assert backend.active is None


@pytest.mark.parametrize('robot', ['franka', 'yam'])
async def test_so101_model_cannot_be_selected_for_other_robots(robot):
    backend = VLABackend({})
    with pytest.raises(ValueError, match='robot_type'):
        await backend.start_session(model(), {'robot_type': robot})
    assert backend.active is None


@pytest.mark.parametrize('mutate,match', [
    (lambda obs: obs.sensors.pop(), 'left_image'),
    (lambda obs: obs.state['joint_position'].CopyFrom(tensor_from_numpy(np.zeros(7, np.float32))), 'joint_position'),
    (lambda obs: obs.state['gripper_position'].CopyFrom(tensor_from_numpy(np.array([np.nan], np.float32))), 'gripper_position'),
    (lambda obs: setattr(obs, 'instruction', ' '), 'instruction is empty'),
])
async def test_malformed_observation_is_rejected_before_any_request(mutate, match):
    obs = observation()
    mutate(obs)
    with pytest.raises(ValueError, match=match):
        await SO101VLAAdapter({}).infer(model(endpoint='http://127.0.0.1:1'), obs)


def test_sensors_are_configurable_and_must_differ():
    assert SO101VLAAdapter({'external_sensor': 'left_image', 'wrist_sensor': 'right_image'}).wrist_sensor == 'right_image'
    with pytest.raises(ValueError, match='must differ'):
        SO101VLAAdapter({'wrist_sensor': 'head_image'})


def test_worker_cli_accepts_so101_only_for_supported_models(tmp_path):
    base = ['--checkpoint', str(tmp_path), '--port', '9100', '--robot-type', 'so101']
    assert parse_args(['molmoact2', *base]).robot_type == 'so101'
    assert parse_args(['pi05_lerobot', *base]).tokenizer is None
    # GR00T's VLM processor is bundled with the checkpoint; only the base snapshot is separate.
    assert parse_args(['groot_n17', *base, '--base-model', str(tmp_path)]).processor is None
    with pytest.raises(SystemExit):
        parse_args(['groot_n17', *base])
    with pytest.raises(SystemExit):
        parse_args(['lap_3b', *base, '--tokenizer', str(tmp_path)])
    with pytest.raises(SystemExit):
        parse_args(['molmoact2', *base, '--g05-override', 'no-equals'])


def test_g05_so101_launches_upstream_with_chunking_and_overrides(tmp_path, monkeypatch):
    from colosseum_policy_server import model_service
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/serve_policy.py').write_text('')
    checkpoint = tmp_path / 'model_state_dict.pt'
    checkpoint.write_bytes(b'')
    args = parse_args(['g05', '--checkpoint', str(checkpoint), '--source-root', str(tmp_path), '--port', '9104',
                       '--robot-type', 'so101', '--action-steps', '30', '--g05-native-attention',
                       '--g05-override', 'model.use_torch_compile=false'])
    seen = []
    monkeypatch.setattr(model_service.runpy, 'run_path', lambda *a, **k: seen.extend(model_service.sys.argv))
    model_service.run_g05(args)
    assert seen[-4:] == ['--action_steps', '30', 'eval_embodiment=so100', 'model.use_torch_compile=false']


@pytest.mark.parametrize('robot_type, module, runtime_name, expected', [
    ('so101', 'so101_lingbot', 'LingBotSO101Runtime', {'use_compile': False}),
    ('yam', 'yam_lingbot', 'LingBotYAMRuntime', {'use_compile': False, 'return_steps': 12}),
])
def test_lingbot_worker_passes_return_steps_only_to_yam(tmp_path, monkeypatch, robot_type, module, runtime_name,
                                                        expected):
    import importlib
    from colosseum_policy_server import model_service
    runtime_module = importlib.import_module(f'colosseum_policy_server.{module}')
    seen = {}

    class Runtime:
        def __init__(self, checkpoint, *, source_root, processor, device, dtype, **options):
            seen.update(options)

    class Server:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def serve_forever(self): pass

    monkeypatch.setattr(runtime_module, 'validate_assets', lambda *args: None)
    monkeypatch.setattr(runtime_module, runtime_name, Runtime)
    monkeypatch.setattr(model_service, 'HTTPServer', Server)
    model_service.main(['lingbot_v2', '--robot-type', robot_type, '--checkpoint', str(tmp_path),
                        '--source-root', str(tmp_path), '--processor', str(tmp_path),
                        '--lingbot-return-steps', '12', '--port', '9218'])
    assert seen == expected
