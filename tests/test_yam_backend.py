from dataclasses import replace
import threading

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.vla import VLABackend
from colosseum_policy_server.model_adapters.yam_vla import YAMVLAAdapter
from colosseum_policy_server.model_service import HTTPServer, http_handler
from colosseum_policy_server.tensors import tensor_from_numpy


def model(endpoint='http://127.0.0.1:9100'):
    return LocalModel.from_mapping(dict(name='molmoact2-yam', url='https://huggingface.co/allenai/MolmoAct2-BimanualYAM',
        revision='a'*40, action_space='joint_position', action_dim=14, control_hz=30,
        max_horizon=30, endpoint=endpoint, backend_options={'adapter': 'molmoact2_yam', 'robot_type': 'yam'}))


@pytest.mark.asyncio
@pytest.mark.parametrize('adapter', ['molmoact2_yam', 'groot_yam', 'pi05_yam', 'g05_yam', 'lingbot_yam', 'isaac_yam'])
async def test_gripper_bounds_logged_before_network(adapter, caplog):
    item = replace(model(), backend_options={'adapter': adapter, 'robot_type': 'yam'})
    obs = observation()
    obs.state['gripper_position'].CopyFrom(tensor_from_numpy(np.array([-.25, 1.25], np.float32)))
    with pytest.raises(ValueError) as exc:
        await YAMVLAAdapter({}).infer(item, obs)
    assert 'left_gripper: measured=-0.250000, low=0.000000, high=1.000000' in str(exc.value)
    assert 'right_gripper: measured=1.250000' in str(exc.value)
    assert str(exc.value) in caplog.text


@pytest.mark.parametrize('adapter', ['molmoact2_yam', 'groot_yam', 'pi05_yam', 'g05_yam', 'lingbot_yam', 'isaac_yam'])
def test_action_gripper_bounds_logged_with_chunk_index(adapter, caplog, monkeypatch):
    import io
    import json
    from colosseum_policy_server.model_adapters import yam_vla
    actions = np.zeros((2, 14))
    actions[1, 13] = 1.25
    monkeypatch.setattr(yam_vla, 'urlopen', lambda *a, **k:
                        io.BytesIO(json.dumps({'actions': actions.tolist()}).encode()))
    item = replace(model(), backend_options={'adapter': adapter, 'robot_type': 'yam'})
    with pytest.raises(ValueError) as exc:
        YAMVLAAdapter({})._request(item, {})
    assert 'action_index=1 right_gripper: target=1.250000, low=0.000000, high=1.000000' in str(exc.value)
    assert str(exc.value) in caplog.text


def observation():
    return pb.Observation(instruction='pick', state={
        'joint_position': tensor_from_numpy(np.arange(12, dtype=np.float32)),
        'gripper_position': tensor_from_numpy(np.array([.25, .75], np.float32)),
    }, sensors=[pb.Image(sensor_id=name, encoding=pb.RAW_RGB, height=2, width=3,
                        data=np.full((2, 3, 3), i, np.uint8).tobytes())
                for i, name in enumerate(('head_image', 'left_image', 'right_image'))])


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter,horizon", [("molmoact2_yam", 30), ("groot_yam", 16), ("pi05_yam", 30), ("g05_yam", 32), ("lingbot_yam", 30), ("isaac_yam", 60)])
async def test_http_roundtrip_and_vla_routing(tmp_path, adapter, horizon):
    class Runtime:
        action_dim = 14
        image_keys = ('top_cam', 'left_cam', 'right_cam')
        def infer(self, payload):
            np.testing.assert_array_equal(payload['state'], [0,1,2,3,4,5,.25,6,7,8,9,10,11,.75])
            assert [payload[key][0,0,0] for key in self.image_keys] == [0,1,2]
            assert payload['instruction'] == 'pick'
            return {'actions': np.tile(payload['state'], (horizon, 1))}
    server = HTTPServer(('127.0.0.1', 0), http_handler(Runtime()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = VLABackend({})
    item = replace(model(f'http://127.0.0.1:{server.server_port}'), max_horizon=horizon,
                   backend_options={'adapter': adapter, 'robot_type': 'yam'})
    try:
        await backend.start_session(item, {'robot_type': 'yam'})
        actions = await backend.infer(item, observation())
        assert actions.shape == (horizon,14)
        assert actions[0,6] == .25 and actions[0,13] == .75
        await backend.end_session()
        from colosseum_policy_server.local_runtime import LocalPolicyRuntime, RuntimeConfig
        from colosseum_policy_server.local_protocol import encode_control, decode_control
        from websockets.asyncio.server import serve
        from websockets.asyncio.client import connect
        cfg = RuntimeConfig('127.0.0.1', 0, 'vla', {}, tmp_path, 1, {item.name: item})
        runtime = LocalPolicyRuntime(cfg, backend=backend)
        async with serve(runtime.handle, '127.0.0.1', 0) as ws_server:
            port = ws_server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}', proxy=None) as ws:
                await ws.send(encode_control({'type': 'prepare', 'protocol_version': 1,
                    'run_id': 'yam-run', 'preparation_id': 'prep', 'robot_type': 'yam',
                    'model': item.public_spec()}))
                assert decode_control(await ws.recv())['type'] == 'ready'
                await ws.send(pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION,
                    session_id='yam-run', sequence=1, deadline_ms=2000,
                    payload=observation().SerializeToString()).SerializeToString())
                frame = pb.RelayFrame.FromString(await ws.recv())
                assert frame.type == pb.ACTION_PLAN
                plan = pb.ActionPlan.FromString(frame.payload)
                assert list(plan.actions.shape) == [horizon, 14] and plan.control_hz == 30
                await ws.send(pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE,
                    session_id='yam-run').SerializeToString())
    finally:
        await backend.end_session()
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.asyncio
async def test_reject_wrong_robot():
    with pytest.raises(ValueError, match='robot_type'):
        await VLABackend({}).start_session(model(), {'robot_type': 'franka'})


@pytest.mark.parametrize('updates', [dict(action_dim=8), dict(control_hz=15), dict(action_space='joint_velocity')])
def test_wrong_model_contract(updates):
    with pytest.raises(ValueError, match='14-D'):
        YAMVLAAdapter({}).validate(replace(model(), **updates))


@pytest.mark.asyncio
async def test_missing_camera_and_wrong_state_rejected_before_network():
    backend = YAMVLAAdapter({})
    obs = observation()
    del obs.sensors[-1]
    with pytest.raises(ValueError):
        await backend.infer(model(), obs)
    obs = observation()
    obs.state['joint_position'].CopyFrom(tensor_from_numpy(np.zeros(7, np.float32)))
    with pytest.raises(ValueError):
        await backend.infer(model(), obs)


def test_synthetic_worker_check_command():
    import subprocess
    import sys
    from pathlib import Path
    class Runtime:
        action_dim = 14
        image_keys = ('top_cam', 'left_cam', 'right_cam')
        def infer(self, payload):
            assert payload['top_cam'].shape == (480, 640, 3)
            return {'actions': np.tile(payload['state'], (16, 1))}
    server = HTTPServer(('127.0.0.1', 0), http_handler(Runtime()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        script = Path(__file__).parents[1] / 'scripts/check_yam_worker.py'
        result = subprocess.run([sys.executable, str(script), '--endpoint',
            f'http://127.0.0.1:{server.server_port}', '--max-horizon', '16'],
            capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert 'PASS: shape=(16, 14)' in result.stdout
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
