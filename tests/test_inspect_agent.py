"""Offline tests using the actual Inspect agent and scripted provider HTTP."""
import asyncio
from dataclasses import replace
import json
import time

import numpy as np
import pytest

pytest.importorskip("inspect_robots_agent")
import httpx
from inspect_robots_agent import LLMAgentPolicy

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.inspect_agent import InspectAgentBackend
from colosseum_policy_server.local_protocol import encode_control, decode_control
from colosseum_policy_server.local_runtime import LocalPolicyRuntime, RuntimeConfig, load_backend


def model(provider="x-ai", name="grok-4.7"):
    return LocalModel.from_mapping(dict(name=name, model_type="llm", url={"openai":"https://api.openai.com/v1", "x-ai":"https://api.x.ai/v1", "anthropic":"https://api.anthropic.com/v1"}[provider],
        revision="rig-v1", endpoint="inprocess://inspect-agent", action_space="joint_position",
        action_dim=8, control_hz=15, max_horizon=2))


def options(tmp_path):
    return dict(joint_low=[-1.] * 7, joint_high=[1.] * 7, joint_max_step=[.02] * 7,
        gripper_open_value=0, robot_notes="Test rig. joint1 rotates counterclockwise.",
        log_dir=str(tmp_path), max_llm_calls=10)


def observation(step=0):
    obs = pb.Observation(instruction="Move joint1 slightly", control_step=step)
    for name, values in (("joint_position", [0.] * 7), ("gripper_position", [0.])):
        array = np.array(values, np.float32)
        obs.state[name].CopyFrom(pb.Tensor(shape=array.shape, dtype=pb.FLOAT32, data=array.tobytes()))
    for name in ("head_image", "left_image"):
        obs.sensors.add(sensor_id=name, encoding=pb.RAW_RGB, width=2, height=2, data=bytes(12))
    return obs


def response(wire, name="move_joints", arguments=None):
    args = arguments if arguments is not None else {"targets": {"joint1": .03}, "note": "Move toward target."}
    if wire == "messages":
        return dict(content=[dict(type="tool_use", id="call1", name=name, input=args)],
                    stop_reason="tool_use", usage={})
    if wire == "responses":
        return dict(id="response1", status="completed", output=[dict(type="function_call",
            id="fc1", call_id="call1", name=name, arguments=json.dumps(args))], usage={})
    return dict(choices=[dict(message=dict(role="assistant", content=None, tool_calls=[
        dict(id="call1", type="function", function=dict(name=name, arguments=json.dumps(args)))]))])


def factory(handler):
    def build(**kwargs):
        kwargs.setdefault("env", {"OPENAI_API_KEY": "test", "XAI_API_KEY": "test", "ANTHROPIC_API_KEY": "test"})
        return LLMAgentPolicy(**kwargs, transport=httpx.MockTransport(handler))
    return build


@pytest.mark.parametrize("provider,name,wire,path", [
    ("openai", "gpt-6-astra", "responses", "/v1/responses"),
    ("x-ai", "grok-4.7", "chat", "/v1/chat/completions"),
    ("anthropic", "claude-opus-5-5", "messages", "/v1/messages"),
])
async def test_real_agent_provider_conversion_and_reset(tmp_path, provider, name, wire, path):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == path
        assert requests[-1]["model"] == name
        return httpx.Response(200, json=response(wire))
    backend = InspectAgentBackend(options(tmp_path), agent_factory=factory(handler))
    item = model(provider, name)
    await backend.start_session(item, {"api_key": "test", "run_id": "../untrusted"})
    actions = await backend.infer(item, observation())
    assert actions.shape == (2, 8)  # Truncate a longer interpolation.
    assert actions.dtype == np.float32
    assert 0 < actions[0, 0] < actions[1, 0] <= .03
    assert np.all(actions[:, 7] == 0)  # Preserve RobotEnv open=0.
    agent = backend.agent
    await backend.infer(item, observation(2))
    assert len(agent.transcript()) > 2
    await backend.end_session()
    assert agent._client._http.is_closed
    assert list(tmp_path.glob("*.json"))
    await backend.start_session(item, {"api_key": "test", "run_id": "next"})
    assert backend.agent is not agent
    await backend.infer(item, observation(0))
    assert backend.last_step == 0
    await backend.end_session()


@pytest.mark.parametrize("reason", ["done", "give_up"])
@pytest.mark.parametrize("open_value", [0, 1])
async def test_stop_holds_without_more_llm_calls_and_resets(tmp_path, reason, open_value):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response("chat", reason,
            {"summary": "Finished", "reason": "Cannot proceed", "hindsight": "none"}))
    backend = InspectAgentBackend({**options(tmp_path), "gripper_open_value": open_value},
                                  agent_factory=factory(handler))
    item = model()
    await backend.start_session(item, {"api_key": "test", "run_id": "run"})
    held = await backend.infer(item, observation())
    assert held.shape == (1, 8) and np.all(held == 0)
    for step in (1, 2, 8):
        obs = observation(step)
        obs.state["joint_position"].data = np.full(7, .005, np.float32).tobytes()
        obs.state["gripper_position"].data = np.array([1.], np.float32).tobytes()
        del obs.sensors[:]  # Hold needs no further images or cloud calls.
        np.testing.assert_array_equal(await backend.infer(item, obs), held)
    assert len(calls) == 1
    obs = observation(9)
    obs.state["joint_position"].data = np.full(7, .5, np.float32).tobytes()
    np.testing.assert_array_equal(await backend.infer(item, obs), held)
    assert len(calls) == 1
    obs = observation(10)
    obs.state["joint_position"].data = np.full(7, 1.1, np.float32).tobytes()
    with pytest.raises(ValueError, match="outside the configured rig bounds"):
        await backend.infer(item, obs)
    await backend.end_session()
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["stop_reason"] == reason and record["stop_step"] == 0
    assert record["hold_target"] == [0.] * 8
    assert 'success' not in record
    await backend.start_session(item, {"api_key": "test", "run_id": "next"})
    assert backend.hold_target is None and backend.stop_reason is None
    await backend.infer(item, observation())
    assert len(calls) == 2
    await backend.end_session()


async def test_stop_keeps_executed_gripper_command(tmp_path):
    from inspect_robots.types import Action, ActionChunk
    backend = InspectAgentBackend(options(tmp_path), agent_factory=factory(lambda r:
        httpx.Response(200, json=response("chat"))))
    item = model()
    await backend.start_session(item, {"api_key": "test", "run_id": "run"})
    await backend.infer(item, observation())
    # RobotEnv open=0; simulate a two-row plan with open then closed grip.
    backend.last_actions = np.array([[0.] * 8, [0.] * 7 + [1.]], np.float32)
    backend.agent.act = lambda obs: ActionChunk([Action(np.ones(8),
        meta={"request_stop": True, "stop_reason": "done"})])
    # Measured width remains open because an object obstructs closure.
    held = await backend.infer(item, observation(2))
    assert held[0, 7] == 1 and np.all(held[0, :7] == 0)
    await backend.end_session()


async def test_timeout_drains_worker_before_new_session(tmp_path):
    def handler(request):
        time.sleep(.08)
        return httpx.Response(200, json=response("chat"))
    backend = InspectAgentBackend(options(tmp_path), agent_factory=factory(handler))
    item = model()
    await backend.start_session(item, {"api_key": "test", "run_id": "run"})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(backend.infer(item, observation()), .01)
    with pytest.raises(RuntimeError, match="previous agent"):
        await backend.start_session(item, {"api_key": "test", "run_id": "next"})
    await backend.end_session()
    assert backend._inflight is None and backend.agent is None
    await backend.start_session(item, {"api_key": "test", "run_id": "next"})
    await backend.end_session()


async def test_bad_observation_and_action_rejected(tmp_path):
    from inspect_robots.types import ActionChunk, Action
    backend = InspectAgentBackend(options(tmp_path), agent_factory=factory(lambda r:
        httpx.Response(200, json=response("chat"))))
    item = model()
    await backend.start_session(item, {"api_key": "test", "run_id": "run"})
    obs = observation()
    del obs.sensors[:]
    with pytest.raises(ValueError, match="RGB sensor"):
        await backend.infer(item, obs)
    await backend.infer(item, observation())
    with pytest.raises(ValueError, match="control_step"):
        await backend.infer(item, observation())
    backend.agent.act = lambda obs: ActionChunk([Action(np.r_[[.5] * 7, 1.])])
    with pytest.raises(ValueError, match="per-step"):
        await backend.infer(item, observation(2))
    backend.agent.act = lambda obs: ActionChunk([Action(np.r_[[2.] * 7, 1.])])
    with pytest.raises(ValueError, match="rig bounds"):
        await backend.infer(item, observation(2))
    await backend.end_session()


@pytest.mark.parametrize("change", [dict(joint_low=[0]), dict(gripper_open_value=None),
                                    dict(joint_max_step=[0] * 7), dict(robot_notes="")])
async def test_rig_declarations_required(tmp_path, change):
    backend = InspectAgentBackend({**options(tmp_path), **change})
    with pytest.raises(ValueError):
        await backend.start_session(model(), {"api_key": "test", "run_id": "run"})


async def test_only_joint_position_supported(tmp_path):
    backend = InspectAgentBackend(options(tmp_path))
    with pytest.raises(ValueError, match="joint_position"):
        await backend.start_session(replace(model(), action_space="joint_velocity"), {"api_key": "test", "run_id": "r"})


class Socket:
    def __init__(self, frames): self.frames, self.sent = list(frames), []
    async def recv(self): return self.frames.pop(0)
    async def send(self, value): self.sent.append(value)


async def test_protobuf_runtime_round_trip(tmp_path):
    replies = iter([response("chat"), response("chat", "done", {"summary": "Finished", "hindsight": "none"})])
    backend = InspectAgentBackend(options(tmp_path), agent_factory=factory(lambda r:
        httpx.Response(200, json=next(replies))))
    item = model()
    cfg = RuntimeConfig("127.0.0.1", 8000, "inspect_agent", {}, tmp_path, 1, {"agent": item})
    frames = [encode_control(dict(type="prepare", run_id="r", model=item.public_spec(), api_key="test"))]
    for sequence, step in ((1, 0), (2, 2), (3, 3)):
        frames.append(pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION,
            session_id="r", sequence=sequence, deadline_ms=5000,
            payload=observation(step).SerializeToString()).SerializeToString())
    frames.append(pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id="r").SerializeToString())
    ws = Socket(frames)
    await LocalPolicyRuntime(cfg, backend).handle(ws)
    assert decode_control(ws.sent[0])["type"] == "ready"
    frame = pb.RelayFrame.FromString(ws.sent[1])
    assert frame.type == pb.ACTION_PLAN
    plan = pb.ActionPlan.FromString(frame.payload)
    assert (plan.request_sequence, plan.start_step, plan.valid_until_step, plan.control_hz) == (1, 0, 1, 15)
    assert list(plan.actions.shape) == [2, 8]
    assert len(ws.sent) == 4
    holds = []
    for index, step in ((2, 2), (3, 3)):
        frame = pb.RelayFrame.FromString(ws.sent[index])
        assert frame.type == pb.ACTION_PLAN
        plan = pb.ActionPlan.FromString(frame.payload)
        assert plan.start_step == plan.valid_until_step == step
        assert list(plan.actions.shape) == [1, 8]
        holds.append(plan.actions.data)
    assert holds[0] == holds[1]
    assert backend.agent is None


def test_entry_point():
    assert isinstance(load_backend("inspect_agent", {}), InspectAgentBackend)


async def test_registered_hf_manifest_identity(tmp_path):
    row = dict(name="registered-agent", url="https://huggingface.co/example/agent-manifest",
        revision="a" * 40, endpoint="inprocess://inspect-agent", action_space="joint_position",
        action_dim=8, control_hz=15, max_horizon=2,
        backend_options={"agent_model": "x-ai/grok-4.7"})
    item = LocalModel.from_mapping(row)
    backend = InspectAgentBackend(options(tmp_path), agent_factory=factory(lambda r:
        httpx.Response(200, json=response("chat"))))
    await backend.start_session(item, {"api_key": "test", "run_id": "r"})
    assert (await backend.infer(item, observation())).shape == (2, 8)
    assert item.public_spec()["url"] == row["url"]
    await backend.end_session()


async def test_byok_no_host_fallback_and_no_cross_session_leak(tmp_path, monkeypatch):
    monkeypatch.setenv('XAI_API_KEY', 'host-secret-must-not-be-used')
    auth = []
    def handler(request):
        auth.append(request.headers['authorization'])
        return httpx.Response(200, json=response('chat'))
    backend = InspectAgentBackend(options(tmp_path), agent_factory=factory(handler))
    item = model()
    with pytest.raises(ValueError, match='Client must supply'):
        await backend.start_session(item, {'run_id': 'missing'})
    await backend.end_session()
    for key in ['client-a-secret', 'client-b-secret']:
        await backend.start_session(item, {'run_id': 'run', 'api_key': key})
        await backend.infer(item, observation())
        await backend.end_session()
    assert auth == ['Bearer client-a-secret', 'Bearer client-b-secret']
    assert 'secret' not in ''.join(p.read_text() for p in tmp_path.rglob('*') if p.is_file())


async def test_auto_backend_routes_types(tmp_path):
    from colosseum_policy_server.backends.auto import AutoBackend
    from colosseum_policy_server.backends.vla import VLABackend
    backend = AutoBackend({'llm': options(tmp_path)})
    legacy = LocalModel.from_mapping(dict(name='vla', url='https://huggingface.co/org/model',
        revision='a'*40, action_space='joint_position', action_dim=8, control_hz=15,
        max_horizon=2, endpoint='http://127.0.0.1:9100', backend_options={'adapter': 'molmoact2'}))
    await backend.start_session(legacy, {})
    assert isinstance(backend.active, VLABackend)
    await backend.end_session()
    await backend.start_session(model(), {'run_id': 'r', 'api_key': 'test'})
    assert isinstance(backend.active, InspectAgentBackend)
    await backend.end_session()


async def test_auto_flat_options_and_session_switch(tmp_path):
    from colosseum_policy_server.backends.auto import AutoBackend
    backend = AutoBackend({**options(tmp_path), 'external_sensor': 'front',
                           'llm': {'robot_notes': 'LLM override'}})
    await backend.start_session(model(), {'run_id': 'r', 'api_key': 'test'})
    assert backend.active.docs.startswith('LLM override')
    assert backend.active.low.tolist() == [-1.] * 7
    with pytest.raises(RuntimeError, match='has not closed'):
        await backend.start_session(model(), {'run_id': 'r2', 'api_key': 'test'})
    await backend.end_session()
    legacy = LocalModel.from_mapping(dict(name='vla', url='https://huggingface.co/org/model',
        revision='a'*40, action_space='joint_position', action_dim=8, control_hz=15,
        max_horizon=2, endpoint='http://127.0.0.1:9100', backend_options={'adapter': 'molmoact2'}))
    await backend.start_session(legacy, {})
    assert backend.active.active.external_sensor == 'front'
    await backend.end_session()
    assert backend.active is None


async def test_llm_rejects_wrong_so101_action_contract_before_creating_agent(tmp_path):
    backend = InspectAgentBackend(options(tmp_path))
    with pytest.raises(ValueError, match='so101 LLM requires 6-D joint_position'):
        await backend.start_session(model(), {'run_id': 'r', 'api_key': 'test', 'robot_type': 'so101'})
    assert backend.agent is None

@pytest.mark.parametrize('provider,name,wire,path', [
    ('openai','gpt-6-astra','responses','/v1/responses'),
    ('x-ai','grok-4.7','chat','/v1/chat/completions'),
    ('anthropic','claude-opus-5-5','messages','/v1/messages'),
])
async def test_relay_endpoint_credentials_and_real_inspect(tmp_path, provider, name, wire, path):
    from colosseum_policy_server.backends.llm_transport import BearerTransport
    seen=[]
    def handler(request):
        assert request.url.host == 'api.yhlxj.ai'
        assert request.url.path == path
        assert request.headers['authorization'] == 'Bearer per-session-secret'
        assert 'x-api-key' not in request.headers
        if wire == 'responses':
            assert b'prompt_cache' not in request.content
        seen.append(json.loads(request.content))
        return httpx.Response(200,json=response(wire, 'done', {'note':'Test complete'}))
    def build(**kwargs):
        custom = kwargs.pop('transport',None)
        if custom:
            custom.close()
            custom=BearerTransport('per-session-secret',httpx.MockTransport(handler))
        return LLMAgentPolicy(**kwargs,transport=custom or httpx.MockTransport(handler))
    original=model(provider,name)
    item=LocalModel.from_mapping({**original.__dict__, 'launcher':[], 'url':'https://api.yhlxj.ai' if provider=='anthropic' else 'https://api.yhlxj.ai/v1', 'backend_options':{'api_auth':'bearer'}})
    backend=InspectAgentBackend(options(tmp_path),agent_factory=build)
    await backend.start_session(item, {'api_key':'per-session-secret','run_id':'relay'})
    actions=await backend.infer(item,observation())
    assert actions.shape==(1,8)
    assert seen[0]['model']==name
    await backend.end_session()


def yam_options(tmp_path):
    return {**options(tmp_path), 'robot_type': 'yam', 'joint_low': [-1.]*12,
            'joint_high': [1.]*12, 'joint_max_step': [.02]*12, 'gripper_open_value': 1}


def yam_observation(step=0):
    obs = observation(step)
    for name, values in [('joint_position', [0.]*12), ('gripper_position', [.25, .75])]:
        array = np.asarray(values, np.float32)
        obs.state[name].CopyFrom(pb.Tensor(shape=array.shape, dtype=pb.FLOAT32, data=array.tobytes()))
    obs.sensors.add(sensor_id='right_image', encoding=pb.RAW_RGB, width=2, height=2, data=bytes(12))
    return obs


def yam_model(provider='x-ai', name='grok-4.7'):
    return replace(model(provider, name), action_dim=14, control_hz=30,
                   backend_options={'robot_type': 'yam'})


@pytest.mark.parametrize('robot_type', ['yam', 'franka'])
async def test_observed_bounds_error_identifies_all_bad_dimensions(tmp_path, robot_type):
    is_yam = robot_type == 'yam'
    backend = InspectAgentBackend(yam_options(tmp_path) if is_yam else options(tmp_path),
        agent_factory=factory(lambda r: pytest.fail('Invalid state must not call provider')))
    item = yam_model() if is_yam else model()
    await backend.start_session(item, {'run_id': 'bounds', 'api_key': 'test', 'robot_type': robot_type})
    obs = yam_observation() if is_yam else observation()
    joints = np.zeros(12 if is_yam else 7, np.float32)
    joints[0], joints[-1] = -1.25, 1.5
    obs.state['joint_position'].data = joints.tobytes()
    obs.state['gripper_position'].data = np.asarray([-.25, 1.25] if is_yam else [1.25], np.float32).tobytes()
    try:
        with pytest.raises(ValueError) as exc:
            await backend.infer(item, obs)
        message = str(exc.value)
        for label, value, low, high in (
            ('left_joint1' if is_yam else 'joint1', -1.25, -1, 1),
            ('right_joint6' if is_yam else 'joint7', 1.5, -1, 1),
            ('right_gripper' if is_yam else 'gripper', 1.25, 0, 1),
        ):
            assert f'{label}: measured={value:.6f}, low={low:.6f}, high={high:.6f}' in message
        if is_yam:
            assert 'left_gripper: measured=-0.250000, low=0.000000, high=1.000000' in message
    finally:
        await backend.end_session()


@pytest.mark.parametrize('provider,name,wire,path', [
    ('openai', 'gpt-6-astra', 'responses', '/v1/responses'),
    ('x-ai', 'grok-4.7', 'chat', '/v1/chat/completions'),
    ('anthropic', 'claude-opus-5-5', 'messages', '/v1/messages'),
])
async def test_yam_three_providers_through_runtime(tmp_path, provider, name, wire, path):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        assert request.url.path == path and calls[-1]['model'] == name
        return httpx.Response(200, json=response(wire, 'move_joints', {
            'targets': {'left_joint1': .03, 'right_joint1': -.03}, 'note': 'Move both arms'}))
    backend = InspectAgentBackend(yam_options(tmp_path), agent_factory=factory(handler))
    item = yam_model(provider, name)
    cfg = RuntimeConfig('127.0.0.1', 8000, 'inspect_agent', {}, tmp_path, 1, {'agent': item})
    ws = Socket([
        encode_control(dict(type='prepare', run_id='yam', robot_type='yam',
                            model=item.public_spec(), api_key='test')),
        pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION, session_id='yam',
            sequence=1, deadline_ms=5000, payload=yam_observation().SerializeToString()).SerializeToString(),
        pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id='yam').SerializeToString(),
    ])
    await LocalPolicyRuntime(cfg, backend).handle(ws)
    assert decode_control(ws.sent[0])['type'] == 'ready'
    frame = pb.RelayFrame.FromString(ws.sent[1])
    assert frame.type == pb.ACTION_PLAN
    plan = pb.ActionPlan.FromString(frame.payload)
    assert list(plan.actions.shape) == [2, 14] and plan.control_hz == 30
    actions = np.frombuffer(plan.actions.data, np.float32).reshape(2,14)
    assert 0 < actions[0,0] < actions[1,0] <= .03
    assert -.03 <= actions[1,7] < actions[0,7] < 0
    np.testing.assert_array_equal(actions[:, [6,13]], [[.25,.75],[.25,.75]])
    assert len(calls) == 1 and backend.agent is None


@pytest.mark.parametrize('reason', ['done', 'give_up'])
async def test_yam_stop_retains_both_executed_grippers(tmp_path, reason):
    from inspect_robots.types import Action, ActionChunk
    backend = InspectAgentBackend(yam_options(tmp_path), agent_factory=factory(lambda r:
        httpx.Response(200, json=response('chat', 'move_joints',
            {'targets': {'left_joint1': .01}, 'note': 'move'}))))
    item = yam_model()
    await backend.start_session(item, {'run_id': 'yam', 'api_key': 'test', 'robot_type': 'yam'})
    await backend.infer(item, yam_observation())
    backend.last_actions = np.zeros((2,14), np.float32)
    backend.last_actions[:, [6,13]] = [[.1,.9],[.8,.2]]
    backend.agent.act = lambda obs: ActionChunk([Action(np.zeros(14),
        meta={'request_stop': True, 'stop_reason': reason})])
    held = await backend.infer(item, yam_observation(1))
    np.testing.assert_allclose(held[0,[6,13]], [.1,.9])
    backend.agent.act = lambda obs: pytest.fail('Holding must not call the LLM again')
    obs = yam_observation(2)
    del obs.sensors[:]
    np.testing.assert_array_equal(await backend.infer(item, obs), held)
    obs = yam_observation(3)
    values = np.zeros(12, np.float32)
    values[-1] = .5
    obs.state['joint_position'].data = values.tobytes()
    np.testing.assert_array_equal(await backend.infer(item, obs), held)
    obs = yam_observation(4)
    values[-1] = 1.1
    obs.state['joint_position'].data = values.tobytes()
    with pytest.raises(ValueError, match='outside the configured rig bounds'):
        await backend.infer(item, obs)
    await backend.end_session()


async def test_yam_rejects_bad_contract_and_right_arm_jump(tmp_path):
    from inspect_robots.types import Action, ActionChunk
    backend = InspectAgentBackend(yam_options(tmp_path), agent_factory=factory(lambda r:
        httpx.Response(200, json=response('chat', 'move_joints',
            {'targets': {'right_joint1': .01}, 'note': 'move'}))))
    for item in [replace(yam_model(), action_dim=8), replace(yam_model(), control_hz=15)]:
        with pytest.raises(ValueError):
            await backend.start_session(item, {'run_id': 'r', 'api_key': 'test', 'robot_type': 'yam'})
        assert backend.agent is None
    item = yam_model()
    await backend.start_session(item, {'run_id': 'r', 'api_key': 'test', 'robot_type': 'yam'})
    obs = yam_observation()
    del obs.sensors[-1]
    with pytest.raises(ValueError, match='RGB sensor'):
        await backend.infer(item, obs)
    await backend.infer(item, yam_observation())
    action = np.zeros(14)
    action[12] = .5
    backend.agent.act = lambda obs: ActionChunk([Action(action)])
    with pytest.raises(ValueError, match='per-step'):
        await backend.infer(item, yam_observation(2))
    action[12], action[6] = 0, 1.1
    with pytest.raises(ValueError, match='rig bounds'):
        await backend.infer(item, yam_observation(2))
    await backend.end_session()


async def test_yam_agent_example_routes_all_three_models(tmp_path):
    from pathlib import Path
    from colosseum_policy_server.backends.auto import AutoBackend
    cfg = RuntimeConfig.from_yaml(Path(__file__).parents[1] / 'configs/local-runtime-yam-agent.yaml.example')
    assert {m.name for m in cfg.models.values()} == {'grok-4.7', 'claude-opus-5-5', 'gpt-6-astra'}
    backend = AutoBackend({'llm': yam_options(tmp_path)})
    for item in cfg.models.values():
        await backend.start_session(item, {'run_id': 'config', 'api_key': 'test', 'robot_type': 'yam'})
        assert isinstance(backend.active, InspectAgentBackend)
        assert backend.active.robot.action_dim == 14
        assert backend.active.gripper_indices == [6, 13]
        await backend.end_session()
