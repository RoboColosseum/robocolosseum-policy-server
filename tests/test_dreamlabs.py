import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends import dreamlabs
from colosseum_policy_server.backends.auto import AutoBackend
from colosseum_policy_server.local_runtime import RuntimeConfig
from colosseum_policy_server.tensors import tensor_from_numpy


def config():
    return RuntimeConfig.from_yaml(Path(__file__).parents[1] / 'configs/local-runtime-yam-dreamlabs.yaml.example')


def test_auto_dispatches_dream_example_without_cloud(monkeypatch):
    value = config()
    start = AsyncMock()
    monkeypatch.setattr(dreamlabs.DreamLabsBackend, 'start_session', start)
    backend = AutoBackend(value.backend_options)
    asyncio.run(backend.start_session(next(iter(value.models.values())), {'robot_type': 'yam'}))
    assert isinstance(backend.active, dreamlabs.DreamLabsBackend)
    assert backend.active.checkpoint == 'dreamlabs/dream-1-beta'
    start.assert_awaited_once()


def test_dream_rejects_other_robot_before_sdk():
    value = config()
    with pytest.raises(ValueError):
        asyncio.run(AutoBackend(value.backend_options).start_session(next(iter(value.models.values())), {'robot_type': 'so101'}))


def observation():
    obs = pb.Observation(instruction='pick up the cup')
    obs.state['joint_position'].CopyFrom(tensor_from_numpy(np.arange(12, dtype=np.float32)))
    obs.state['gripper_position'].CopyFrom(tensor_from_numpy(np.array([.2, .8], np.float32)))
    for name in dreamlabs.CAMERAS:
        obs.sensors.add(sensor_id=name, encoding=pb.RAW_RGB, width=4, height=4,
                        data=np.zeros((4, 4, 3), np.uint8).tobytes())
    return obs


def test_inputs_use_yam_joint_and_camera_order(monkeypatch):
    monkeypatch.setattr(dreamlabs, '_jpeg', lambda rgb: 'encoded')
    frames, state = dreamlabs._model_inputs(observation())
    assert set(frames) == {'image_window.top', 'image_window.left', 'image_window.right'}
    assert state['state.arm_l.joint_pos'] == list(range(6))
    assert state['state.arm_r.joint_pos'] == list(range(6, 12))
    assert state['state.arm_l.gripper_pos'] == pytest.approx([.2])
    assert state['state.arm_r.gripper_pos'] == pytest.approx([.8])


@pytest.mark.parametrize('kind', ['empty_instruction', 'missing_camera', 'invalid_gripper'])
def test_bad_observations_fail(kind, monkeypatch):
    # Input validation must also run in the base environment without OpenCV.
    monkeypatch.setattr(dreamlabs, '_jpeg', lambda rgb: 'encoded')
    obs = observation()
    if kind == 'empty_instruction':
        obs.instruction = ''
    elif kind == 'missing_camera':
        del obs.sensors[-1]
    else:
        obs.state['gripper_position'].CopyFrom(tensor_from_numpy(np.array([-1, 2], np.float32)))
    with pytest.raises(ValueError):
        dreamlabs._model_inputs(obs)


def test_reply_preserves_left_right_order():
    left = np.array([[1, 2, 3, 4, 5, 6, .2]], np.float32)
    right = np.array([[7, 8, 9, 10, 11, 12, .8]], np.float32)
    np.testing.assert_array_equal(dreamlabs._chunk_from_reply({'q_l_chunk': left, 'q_r_chunk': right}),
                                  np.concatenate([left, right], axis=1))


@pytest.mark.parametrize('left,right', [([[0]*6], [[0]*7]), ([[float('nan')]*7], [[0]*7]), ([], [])])
def test_bad_chunks_fail(left, right):
    with pytest.raises(ValueError):
        dreamlabs._chunk_from_reply({'q_l_chunk': left, 'q_r_chunk': right})
