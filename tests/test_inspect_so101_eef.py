"""SO-101 EEF agent contract and official/account authentication."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('inspect_robots')
from inspect_robots.types import Action, ActionChunk

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.inspect_agent import InspectAgentBackend
from colosseum_policy_server.backends.router_eef_tool import install_router_eef_tool


def model(name='gpt-6-astra', url='https://api.openai.com/v1'):
    return LocalModel.from_mapping(dict(name=name, model_type='llm', url=url,
        revision='rig-v1', endpoint='inprocess://inspect-agent', action_space='joint_position',
        action_dim=6, control_hz=30, max_horizon=4,
        backend_options={'robot_type': 'so101', 'control_mode': 'eef_delta_pose'}))


def observation():
    value = pb.Observation(instruction='Move a little', control_step=0)
    for key, data in {'joint_position': [0]*5, 'gripper_position': [40],
                      'eef_state': [0]*6+[40], 'eef_quat_wxyz': [1,0,0,0]}.items():
        array = np.asarray(data, dtype=np.float32)
        value.state[key].CopyFrom(pb.Tensor(shape=array.shape, dtype=pb.FLOAT32, data=array.tobytes()))
    for name in ('head_image', 'left_image'):
        value.sensors.add(sensor_id=name, encoding=pb.RAW_RGB, width=2, height=2, data=bytes(12))
    return value


class Agent:
    def __init__(self):
        self.info = None
        self.actions = [[.001, 0, 0, 0, 0, 0, 0]]

    def bind(self, info):
        self.info = info

    def reset(self, scene):
        self.scene = scene

    def act(self, observation):
        return ActionChunk(actions=tuple(Action(np.asarray(a)) for a in self.actions), control_hz=30)

    def on_trial_end(self, record, log_dir, log_id):
        pass


def test_eef_agent_receives_measured_pose_and_returns_seven_deltas(tmp_path):
    agent = Agent()
    seen = {}
    def factory(**kwargs):
        seen.update(kwargs)
        return agent
    backend = InspectAgentBackend({'log_dir': str(tmp_path)}, agent_factory=factory)
    item = model()
    asyncio.run(backend.start_session(item, {'run_id': 'run', 'robot_type': 'so101',
                                             'api_key': 'key', 'task': {'max_steps': 37}}))
    result = backend._infer_so101_eef(observation())
    assert result.shape == (1, 7)
    assert agent.info.action_space.semantics.agent_eef_target is True
    assert agent.info.action_space.semantics.max_chunk_steps == 4
    assert seen['base_url'] == 'https://api.openai.com/v1'
    assert seen['env'] == {'OPENAI_API_KEY': 'key'}
    assert seen['max_llm_calls'] > 37
    assert 'max_speed_frac' not in seen
    agent.actions = [[.03, 0, 0, 0, 0, 0, 20]]
    next_observation = observation()
    next_observation.control_step = 1
    np.testing.assert_allclose(backend._infer_so101_eef(next_observation), agent.actions)


def test_router_eef_tool_uses_requested_steps_without_inspect_amplitude_cap():
    import json
    from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace, StateField, StateSpec
    from inspect_robots.types import Observation
    from inspect_robots_agent._llm import ToolCall
    from inspect_robots_agent._tools import build_toolset

    toolset = build_toolset(
        Box(shape=(7,), low=-np.ones(7), high=np.ones(7), semantics=ActionSemantics(
            control_mode='eef_delta_pose', rotation_repr='axis_angle', gripper='continuous',
            frame='base', dim_labels=('x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper'),
            agent_eef_target=True, max_chunk_steps=4)),
        ObservationSpace(state=StateSpec((StateField('eef_state', (7,), 'eef'),
                                         StateField('eef_quat_wxyz', (4,), 'quat')))),
        control_hz=30)
    install_router_eef_tool(SimpleNamespace(_toolset=toolset), 4)
    obs = Observation(state={'eef_state': np.array([0, 0, 0, 0, 0, 0, 40]),
                             'eef_quat_wxyz': np.array([1, 0, 0, 0])})
    call = ToolCall(id='1', name='move_eef', arguments=json.dumps({
        'note': 'Move toward the target', 'steps': 2,
        'targets': {'x': 3, 'gripper': 100}}))
    result = toolset.execute(call, obs)
    assert result.error is None
    assert len(result.chunk.actions) == 2
    np.testing.assert_allclose(result.chunk.actions[0].data, [1.5, 0, 0, 0, 0, 0, 30])
    move_schema = toolset.schemas()[0]['function']
    assert move_schema['parameters']['properties']['steps']['maximum'] == 4
    assert 'per-step bounds' not in move_schema['description']


def test_account_login_uses_cli_wire_and_official_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'host-key-must-not-reach-cli')
    seen = {}
    agent = Agent()
    def factory(**kwargs):
        seen.update(kwargs)
        return agent
    backend = InspectAgentBackend({'log_dir': str(tmp_path)}, agent_factory=factory)
    item = model('claude-opus-5-5', 'https://api.anthropic.com/v1')
    asyncio.run(backend.start_session(item, {'run_id': 'run', 'robot_type': 'so101',
                                             'llm_auth': 'claude-code'}))
    assert seen['wire'] == 'claude-code'
    assert seen['model'] == 'anthropic/claude-opus-5-5'
    assert 'api_key_env' not in seen and 'base_url' not in seen
    assert 'ANTHROPIC_API_KEY' not in seen['env']


@pytest.mark.parametrize('public_name,relay_url,api_model,official_url,auth,expected_model', [
    ('gpt-6-astra-so101', 'https://api.yhlxj.ai/v1', 'gpt-6-astra',
     'https://api.openai.com/v1', 'api_key', 'gpt-6-astra'),
    ('grok-4.7-so101', 'https://api.yhlxj.ai/v1', 'grok-4.7',
     'https://api.x.ai/v1', 'api_key', 'grok-4.7'),
    ('claude-opus-5-5-so101', 'https://api.yhlxj.ai', 'claude-opus-5-5',
     'https://api.anthropic.com/v1', 'claude-code', 'anthropic/claude-opus-5-5'),
])
def test_router_relay_identity_uses_official_provider(
        tmp_path, public_name, relay_url, api_model, official_url, auth, expected_model):
    seen = {}
    item = LocalModel.from_mapping(dict(
        name=public_name, model_type='llm', url=relay_url, revision='rig-v1',
        endpoint='inprocess://inspect-agent', action_space='joint_position',
        action_dim=6, control_hz=10, max_horizon=150,
        backend_options={'robot_type': 'so101', 'control_mode': 'eef_delta_pose',
                         'api_model': api_model, 'api_base_url': official_url}))
    backend = InspectAgentBackend({'log_dir': str(tmp_path)},
                                  agent_factory=lambda **kwargs: seen.update(kwargs) or Agent())
    request = {'run_id': 'run', 'robot_type': 'so101'}
    request.update({'llm_auth': 'claude-code'} if auth == 'claude-code' else {'api_key': 'key'})
    asyncio.run(backend.start_session(item, request))
    assert item.public_spec()['name'] == public_name
    assert item.public_spec()['url'] == relay_url
    assert seen['model'] == expected_model
    if auth == 'claude-code':
        assert seen['wire'] == 'claude-code'
        assert 'base_url' not in seen and 'api_key_env' not in seen
    else:
        assert seen['base_url'] == official_url
        assert relay_url not in seen.values()
    asyncio.run(backend.end_session())


def test_relay_override_rejects_provider_crossing_and_relay_inference(tmp_path):
    def configured(api_model, api_base_url):
        return LocalModel.from_mapping(dict(
            name='gpt-6-astra-so101', model_type='llm',
            url='https://api.yhlxj.ai/v1', revision='rig-v1',
            endpoint='inprocess://inspect-agent', action_space='joint_position',
            action_dim=6, control_hz=10, max_horizon=150,
            backend_options={'robot_type': 'so101', 'control_mode': 'eef_delta_pose',
                             'api_model': api_model, 'api_base_url': api_base_url}))
    backend = InspectAgentBackend({'log_dir': str(tmp_path)}, agent_factory=lambda **_: Agent())
    request = {'run_id': 'run', 'robot_type': 'so101', 'api_key': 'key'}
    with pytest.raises(ValueError, match='api_model'):
        asyncio.run(backend.start_session(configured('claude-opus-5-5', 'https://api.openai.com/v1'), request))
    with pytest.raises(ValueError, match='official API URL'):
        asyncio.run(backend.start_session(configured('gpt-6-astra', 'https://api.yhlxj.ai/v1'), request))
