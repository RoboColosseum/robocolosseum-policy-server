"""SO-101 LLM joint control uses the same six-value contract as VLA models."""

import asyncio

import numpy as np
import pytest

pytest.importorskip('inspect_robots')
from inspect_robots.types import Action, ActionChunk

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.inspect_agent import InspectAgentBackend


class Agent:
    def __init__(self):
        self.info = None
        self.actions = [[0.5, 0, 0, 0, 0, 43]]

    def bind(self, info):
        self.info = info

    def reset(self, scene):
        self.scene = scene

    def act(self, observation):
        self.observation = observation
        return ActionChunk(
            actions=tuple(Action(np.asarray(action)) for action in self.actions),
            control_hz=30,
        )


def observation(step=0):
    value = pb.Observation(instruction='Move the arm', control_step=step)
    for key, data in {'joint_position': [0] * 5, 'gripper_position': [40]}.items():
        array = np.asarray(data, dtype=np.float32)
        value.state[key].CopyFrom(pb.Tensor(
            shape=array.shape, dtype=pb.FLOAT32, data=array.tobytes()))
    for name in ('head_image', 'left_image'):
        value.sensors.add(sensor_id=name, encoding=pb.RAW_RGB,
                          width=2, height=2, data=bytes(12))
    return value


def test_joint_agent_uses_degree_and_gripper_contract(tmp_path):
    model = LocalModel.from_mapping(dict(
        name='gpt-6-astra', model_type='llm', url='https://api.openai.com/v1',
        revision='rig-v1', endpoint='inprocess://inspect-agent',
        action_space='joint_position', action_dim=6, control_hz=30,
        max_horizon=4, backend_options={'robot_type': 'so101'}))
    agent = Agent()
    backend = InspectAgentBackend({
        'robot_type': 'so101', 'control_mode': 'joint_pos',
        'sensors': ['head_image', 'left_image'],
        'joint_low': [-120, -104, -98, -102, -180],
        'joint_high': [120, 104, 98, 102, 180],
        'joint_max_step': [1, 1, 1, 1, 2],
        'gripper_open_value': 1, 'robot_notes': 'Calibrated SO-101.',
        'log_dir': str(tmp_path),
    }, agent_factory=lambda **kwargs: agent)
    asyncio.run(backend.start_session(model, {
        'run_id': 'run', 'robot_type': 'so101', 'api_key': 'test-key'}))

    np.testing.assert_allclose(backend._infer(observation()), [[0.5, 0, 0, 0, 0, 43]])
    assert agent.info.action_space.semantics.control_mode == 'joint_pos'
    assert agent.info.action_space.high[5] == 100
    assert agent.info.observation_space.state.fields[0].unit == 'degrees+gripper_0_100'
    np.testing.assert_allclose(agent.observation.state['joint_pos'], [0, 0, 0, 0, 0, 40])

    agent.actions = [[0, 0, 0, 0, 0, 50]]
    with pytest.raises(ValueError, match='gripper.*limit=5'):
        backend._infer(observation(step=1))
