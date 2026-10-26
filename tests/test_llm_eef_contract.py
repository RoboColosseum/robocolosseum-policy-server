"""The Router identity stays intact while local auth and EE actions are selected."""

import numpy as np
import pytest

pytest.importorskip('inspect_robots')

import httpx

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.inspect_agent import InspectAgentBackend


def _model(name):
    return LocalModel.from_mapping(dict(
        name=name, model_type='llm', url='https://relay.example/v1',
        revision='router-revision', action_space='joint_position', action_dim=14,
        control_hz=30, max_horizon=1, endpoint='inprocess://inspect-agent'))


def _options(*, eef=False):
    options = dict(robot_type='yam', joint_low=[-2.] * 12,
                   joint_high=[2.] * 12, joint_max_step=[.1] * 12,
                   gripper_open_value=1, robot_notes='Calibrated YAM rig.')
    if eef:
        options.update(control_mode='eef_abs_pose', eef_low=[-2.] * 14, eef_high=[2.] * 14)
    return options


@pytest.mark.asyncio
@pytest.mark.parametrize('name,key,base,wire', [
    ('gpt-6-astra', 'OPENAI_API_KEY', 'https://api.openai.com/v1', 'responses'),
    ('grok-4.7', 'XAI_API_KEY', 'https://api.x.ai/v1', 'chat'),
])
async def test_official_api_endpoint_is_independent_of_router_url(name, key, base, wire):
    calls = []
    backend = InspectAgentBackend({**_options(), 'llm_endpoint': 'official'},
                                  agent_factory=lambda **kwargs: calls.append(kwargs) or object())
    await backend.start_session(_model(name), {'run_id': 'router-run', 'api_key': 'local-secret'})
    assert calls[0]['base_url'] == base
    assert calls[0]['env'] == {key: 'local-secret'}
    assert calls[0]['wire'] == wire
    assert backend.run_id == 'router-run'
    assert backend.model.url == 'https://relay.example/v1'


@pytest.mark.asyncio
async def test_claude_account_login_uses_cli_without_key_or_http_url(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'host-secret')
    calls = []
    backend = InspectAgentBackend(_options(), agent_factory=lambda **kwargs: calls.append(kwargs) or object())
    await backend.start_session(_model('claude-opus-5-5'),
                                {'run_id': 'router-run', 'auth_method': 'claude-code'})
    assert calls[0]['wire'] == 'claude-code'
    assert calls[0]['model'] == 'claude-opus-5-5'
    assert 'base_url' not in calls[0] and 'api_key_env' not in calls[0]
    # The CLI must use the account login, not a host API key.
    assert 'ANTHROPIC_API_KEY' not in calls[0]['env']


@pytest.mark.asyncio
async def test_eef_contract_uses_measured_cartesian_state():
    from inspect_robots.types import Action, ActionChunk

    class Agent:
        def bind(self, info):
            assert info.action_space.semantics.control_mode == 'eef_abs_pose'
            assert info.action_space.semantics.dim_labels[0] == 'left_x'

        def reset(self, scene):
            assert scene.id == 'router-run'

        def act(self, observation):
            assert observation.state['eef_state'].shape == (14,)
            return ActionChunk((Action(np.full(14, .05)),))

    model = _model('gpt-6-astra')
    backend = InspectAgentBackend(_options(eef=True), agent_factory=lambda **kwargs: Agent())
    await backend.start_session(model, {'run_id': 'router-run', 'api_key': 'local-secret'})
    obs = pb.Observation(instruction='Move the left gripper', control_step=0)
    for key, values in [('joint_position', np.zeros(12, np.float32)),
                        ('gripper_position', np.zeros(2, np.float32)),
                        ('cartesian_position', np.zeros(14, np.float32))]:
        obs.state[key].CopyFrom(pb.Tensor(shape=values.shape, dtype=pb.FLOAT32,
                                        data=values.tobytes()))
    for camera in ('head_image', 'left_image', 'right_image'):
        obs.sensors.add(sensor_id=camera, encoding=pb.RAW_RGB,
                        width=2, height=2, data=bytes(12))
    actions = backend._infer(obs)
    assert actions.shape == (1, 14)
    np.testing.assert_allclose(actions, .05)


@pytest.mark.asyncio
@pytest.mark.parametrize('name,host,wire', [
    ('gpt-6-astra', 'api.openai.com', 'responses'),
    ('grok-4.7', 'api.x.ai', 'chat'),
])
async def test_real_inspect_agent_binds_eef_and_calls_official_host(name, host, wire):
    import json
    from inspect_robots_agent import LLMAgentPolicy

    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == host
        if wire == 'responses':
            return httpx.Response(200, json={
                'id': 'response1', 'status': 'completed', 'output': [{
                    'type': 'function_call', 'id': 'fc1', 'call_id': 'call1',
                    'name': 'done', 'arguments': json.dumps({'note': 'Test complete'})}],
                'usage': {},
            })
        return httpx.Response(200, json={'choices': [{'message': {
            'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'call1', 'type': 'function', 'function': {
                    'name': 'done', 'arguments': json.dumps({'note': 'Test complete'})}}]}}]})

    def factory(**kwargs):
        return LLMAgentPolicy(**kwargs, transport=httpx.MockTransport(handler))

    backend = InspectAgentBackend({**_options(eef=True), 'llm_endpoint': 'official'}, agent_factory=factory)
    model = _model(name)
    await backend.start_session(model, {'run_id': 'router-run', 'api_key': 'local-secret'})
    obs = pb.Observation(instruction='Move the left gripper', control_step=0)
    for key, values in [('joint_position', np.zeros(12, np.float32)),
                        ('gripper_position', np.zeros(2, np.float32)),
                        ('cartesian_position', np.zeros(14, np.float32))]:
        obs.state[key].CopyFrom(pb.Tensor(shape=values.shape, dtype=pb.FLOAT32,
                                        data=values.tobytes()))
    for camera in ('head_image', 'left_image', 'right_image'):
        obs.sensors.add(sensor_id=camera, encoding=pb.RAW_RGB,
                        width=2, height=2, data=bytes(12))
    actions = backend._infer(obs)
    assert actions.shape == (1, 14)
    assert len(requests) == 1
    await backend.end_session()
