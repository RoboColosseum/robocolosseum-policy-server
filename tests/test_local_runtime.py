import asyncio
import json

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.local_protocol import encode_control, decode_control
from colosseum_policy_server.local_runtime import LocalModel, LocalPolicyRuntime, RuntimeConfig, load_backend


def model():
    return LocalModel.from_mapping({"name":"test", "url":"https://huggingface.co/example/model", "revision":"a" * 40, "subfolder":"", "action_space":"joint_position", "action_dim":8, "control_hz":15, "max_horizon":2, "endpoint":"ws://127.0.0.1:9100"})


def config(tmp_path):
    item = model()
    return RuntimeConfig("127.0.0.1", 8000, "unused", {}, tmp_path, 1, {item.name:item})


def prepare(): return encode_control({"type":"prepare", "protocol_version":1, "run_id":"run", "preparation_id":"prep", "model":model().public_spec()})
def observation(): return pb.RelayFrame(protocol_version=1, type=pb.OBSERVATION, session_id="run", sequence=1, deadline_ms=20, payload=pb.Observation(control_step=0).SerializeToString()).SerializeToString()


class Socket:
    def __init__(self, messages): self.messages, self.sent = list(messages), []
    async def recv(self): return self.messages.pop(0)
    async def send(self, message): self.sent.append(message)


def error_code(message): return pb.Error.FromString(pb.RelayFrame.FromString(message).payload).code


class Backend:
    def __init__(self, result=None, error=None, delay=0): self.result, self.error, self.delay = result, error, delay
    async def infer(self, model, observation):
        if self.delay: await asyncio.sleep(self.delay)
        if self.error: raise self.error
        return self.result


async def test_eef_local_contract_preserves_router_model_and_run_id(tmp_path):
    item = LocalModel.from_mapping({
        'name': 'gpt-6-astra', 'model_type': 'llm',
        'url': 'https://relay.example/v1', 'revision': 'router-revision',
        'action_space': 'joint_position', 'action_dim': 14, 'control_hz': 30,
        'max_horizon': 1, 'endpoint': 'inprocess://inspect-agent',
        'backend_options': {'control_mode': 'eef_abs_pose'},
    })
    cfg = RuntimeConfig('127.0.0.1', 8000, 'unused', {}, tmp_path, 1, {item.name: item})
    close = pb.RelayFrame(type=pb.SESSION_CLOSE, session_id='router-run').SerializeToString()
    request = {'type': 'prepare', 'run_id': 'router-run', 'preparation_id': 'prep',
               'model': item.public_spec(),
               'message': json.dumps({'client_options': {'use_local_action_contract': True}})}
    ws = Socket([encode_control(request), close])
    await LocalPolicyRuntime(cfg, Backend()).handle(ws)
    ready = decode_control(ws.sent[0])
    assert ready['run_id'] == 'router-run'
    assert ready['model'] == item.public_spec()
    effective = json.loads(ready['message'])['effective_model']
    assert effective['action_space'] == 'eef_abs_pose'
    assert effective['url'] == item.url


class FailingSupervisor:
    async def activate(self, model): raise RuntimeError("private launcher detail")
    async def stop(self): pass


class RecordingSupervisor:
    def __init__(self): self.calls = []
    async def activate(self, model): self.calls.append('activate')
    async def stop(self): self.calls.append('stop')


async def test_worker_is_stopped_when_its_session_closes(tmp_path):
    supervisor = RecordingSupervisor()
    close = pb.RelayFrame(type=pb.SESSION_CLOSE, session_id='run').SerializeToString()
    await LocalPolicyRuntime(config(tmp_path), Backend(), supervisor).handle(Socket([prepare(), close]))
    assert supervisor.calls == ['activate', 'stop']


@pytest.mark.parametrize('fails', [False, True])
async def test_progress_reports_wait_and_stops_after_inference(tmp_path, capsys, fails):
    backend = Backend(result=np.zeros((1, 8)), delay=.03,
                      error=RuntimeError('secret-value') if fails else None)
    runtime = LocalPolicyRuntime(config(tmp_path), backend)
    runtime.progress_interval = .005
    try:
        await runtime._infer_with_progress('run', model(), pb.Observation(control_step=7), 1000)
    except RuntimeError:
        assert fails
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]['event'] == 'inference_start'
    assert any(row['event'] == 'inference_waiting' for row in rows)
    assert rows[-1]['event'] == ('inference_failed' if fails else 'inference_returned')
    assert all(row['step'] == 7 and row['model'] == 'test' for row in rows)
    assert 'secret-value' not in json.dumps(rows)
    await asyncio.sleep(.015)
    assert capsys.readouterr().out == ''


async def test_start_failure_is_sanitized(tmp_path):
    ws = Socket([prepare()])
    await LocalPolicyRuntime(config(tmp_path), Backend(), FailingSupervisor()).handle(ws)
    assert error_code(ws.sent[-1]) == "MODEL_START_FAILED"
    assert b"private launcher detail" not in ws.sent[-1]


@pytest.mark.parametrize(("backend", "code"), [(Backend(error=RuntimeError("private failure")), "INFERENCE_FAILED"), (Backend(delay=.05), "INFERENCE_TIMEOUT"), (Backend(result=np.zeros((1, 7))), "INVALID_REQUEST")])
async def test_inference_failures_are_stable(tmp_path, backend, code):
    ws = Socket([prepare(), observation()])
    await LocalPolicyRuntime(config(tmp_path), backend).handle(ws)
    assert error_code(ws.sent[-1]) == code


async def test_observation_idle_timeout_is_not_invalid_request(tmp_path, capsys):
    class IdleSocket(Socket):
        async def recv(self):
            if self.messages:
                return await super().recv()
            await asyncio.Event().wait()
    ws = IdleSocket([observation()])
    runtime = LocalPolicyRuntime(config(tmp_path), Backend(result=np.zeros((1, 8))))
    runtime.observation_timeout_seconds = .01
    await runtime._serve(ws, 'run', model())
    error = pb.Error.FromString(pb.RelayFrame.FromString(ws.sent[-1]).payload)
    assert error.code == 'OBSERVATION_TIMEOUT'
    assert 'waiting for sequence 2' in error.message
    assert 'not a model inference timeout' in error.message
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event['event'] == 'observation_timeout'
    assert event['expected_sequence'] == 2


async def test_first_observation_can_arrive_after_execution_idle_timeout(tmp_path):
    close = pb.RelayFrame(type=pb.SESSION_CLOSE, session_id='run').SerializeToString()
    class SlowStartupSocket(Socket):
        async def recv(self):
            if len(self.messages) == 2:
                await asyncio.sleep(.03)  # Operator approval + hardware startup.
            return await super().recv()
    ws = SlowStartupSocket([observation(), close])
    runtime = LocalPolicyRuntime(config(tmp_path), Backend(result=np.zeros((1, 8))))
    runtime.observation_timeout_seconds = .005
    await asyncio.wait_for(runtime._serve(ws, 'run', model()), 1)
    assert len(ws.sent) == 1
    assert pb.RelayFrame.FromString(ws.sent[0]).type == pb.ACTION_PLAN


async def test_prepared_session_can_close_without_first_observation(tmp_path):
    ws = Socket([pb.RelayFrame(type=pb.SESSION_CLOSE, session_id='run').SerializeToString()])
    await LocalPolicyRuntime(config(tmp_path), Backend())._serve(ws, 'run', model())
    assert ws.sent == []


async def test_prepared_observation_wait_is_cancellable(tmp_path):
    entered = asyncio.Event()
    class WaitingSocket(Socket):
        async def recv(self):
            entered.set()
            await asyncio.Event().wait()
    ws = WaitingSocket([])
    task = asyncio.create_task(LocalPolicyRuntime(config(tmp_path), Backend())._serve(ws, 'run', model()))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ws.sent == []


@pytest.mark.parametrize('field,value,reason', [
    ('sequence', 3, 'sequence mismatch: expected 1, received 3'),
    ('session_id', 'private-other-session', 'session mismatch'),
    ('type', pb.ACTION_PLAN, 'expected OBSERVATION'),
    ('deadline_ms', 0, 'deadline_ms must be positive'),
    ('payload', b'\xffprivate-image', 'Malformed local observation payload'),
])
async def test_observation_rejection_has_safe_diagnostics(tmp_path, capsys, field, value, reason):
    frame = pb.RelayFrame.FromString(observation())
    setattr(frame, field, value)
    ws = Socket([frame.SerializeToString()])
    await LocalPolicyRuntime(config(tmp_path), Backend())._serve(ws, 'run', model())
    error = pb.Error.FromString(pb.RelayFrame.FromString(ws.sent[-1]).payload)
    assert error.code == 'INVALID_REQUEST'
    assert reason in error.message
    output = capsys.readouterr().out
    event = json.loads(output)
    assert event['event'] == 'observation_rejected'
    assert 'private-' not in output + error.message
    assert event['stage'] == ('decode_observation' if field == 'payload' else 'validate_frame')


async def test_malformed_observation_frame_is_reported(tmp_path, capsys):
    ws = Socket([b'\xffprivate-image'])
    await LocalPolicyRuntime(config(tmp_path), Backend())._serve(ws, 'run', model())
    assert error_code(ws.sent[-1]) == 'INVALID_REQUEST'
    output = capsys.readouterr().out
    assert json.loads(output)['stage'] == 'decode_frame'
    assert 'private-image' not in output


def test_backend_entry_point_factory_receives_options(monkeypatch):
    class Plugin:
        def __init__(self, options): self.options = options
        async def infer(self, model, observation): return np.zeros((1, 8))
    class Entry:
        name = "example"
        def load(self): return Plugin
    class Entries:
        def select(self, *, group): return [Entry()]
    monkeypatch.setattr("colosseum_policy_server.local_runtime.metadata.entry_points", lambda: Entries())
    assert load_backend("example", {"value":1}).options == {"value":1}


def test_model_rejects_non_loopback_endpoint():
    with pytest.raises(ValueError):
        LocalModel.from_mapping({"name":"test", "url":"https://huggingface.co/example/model", "revision":"a" * 40, "subfolder":"", "action_space":"joint_position", "action_dim":8, "control_hz":15, "max_horizon":2, "endpoint":"ws://example.invalid:9100"})


def test_model_accepts_loopback_http_endpoint():
    item = LocalModel.from_mapping({"name":"test", "url":"https://huggingface.co/example/model", "revision":"a" * 40, "subfolder":"", "action_space":"joint_position", "action_dim":8, "control_hz":15, "max_horizon":2, "endpoint":"http://127.0.0.1:9100"})
    assert item.endpoint == "http://127.0.0.1:9100"


@pytest.mark.parametrize('subfolder', [None, '', 'ignored-checkpoint'])
@pytest.mark.parametrize('profile', [None, '', 'unknown-legacy-profile'])
async def test_prepare_profile_is_optional_metadata(tmp_path, subfolder, profile):
    from colosseum_policy_server.local_protocol import decode_control
    spec = model().public_spec()
    if subfolder is not None:
        spec['subfolder'] = subfolder
    if profile is not None:
        spec['runtime_profile'] = profile
    request = encode_control({'type': 'prepare', 'protocol_version': 1,
        'run_id': 'run', 'preparation_id': 'prep', 'model': spec})
    close = pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id='run').SerializeToString()
    ws = Socket([request, observation(), close])
    await LocalPolicyRuntime(config(tmp_path), Backend(result=np.ones((1, 8)))).handle(ws)
    ready = decode_control(ws.sent[0])
    assert ready['type'] == 'ready'
    assert ready['model'] == decode_control(request)['model']
    reply = pb.RelayFrame.FromString(ws.sent[1])
    assert reply.type == pb.ACTION_PLAN
    assert list(pb.ActionPlan.FromString(reply.payload).actions.shape) == [1, 8]
    assert ready['model'] == spec


@pytest.mark.parametrize('filename', ['local-runtime.yaml.example', 'local-runtime-droid-example.yaml'])
def test_example_config_needs_no_profile(filename):
    from pathlib import Path
    cfg = RuntimeConfig.from_yaml(Path(__file__).parents[1] / 'configs' / filename)
    assert cfg.models


async def test_new_robot_six_joint_backend_returns_action_plan(tmp_path):
    from colosseum_policy_server.backends import LocalModel as PublicModel
    from colosseum_policy_server.local_protocol import decode_control
    item = PublicModel.from_mapping({
        'name': 'six-joint', 'url': 'https://huggingface.co/example/model',
        'revision': 'a' * 40, 'action_space': 'joint_position', 'action_dim': 6,
        'control_hz': 15, 'max_horizon': 2, 'endpoint': 'http://127.0.0.1:9100'})
    cfg = RuntimeConfig('127.0.0.1', 8000, 'unused', {}, tmp_path, 1, {item.name: item})
    request = encode_control({'type': 'prepare', 'protocol_version': 1,
        'run_id': 'run', 'preparation_id': 'prep', 'model': item.public_spec()})
    close = pb.RelayFrame(protocol_version=1, type=pb.SESSION_CLOSE, session_id='run').SerializeToString()
    ws = Socket([request, observation(), close])
    await LocalPolicyRuntime(cfg, Backend(result=np.ones((2, 6)))).handle(ws)
    assert decode_control(ws.sent[0])['type'] == 'ready'
    frame = pb.RelayFrame.FromString(ws.sent[1])
    assert frame.type == pb.ACTION_PLAN
    assert list(pb.ActionPlan.FromString(frame.payload).actions.shape) == [2, 6]


@pytest.mark.parametrize('robot', ['droid', 'yam', 'custom_robot'])
def test_robot_cli_selects_backend(tmp_path, monkeypatch, robot):
    import yaml
    from colosseum_policy_server.local_server import configured_runtime
    config_path = tmp_path/'runtime.yaml'
    item = model()
    config_path.write_text(yaml.safe_dump({'backend': {'name': 'custom'},
        'models': [{'name': item.name, **item.public_spec(), 'endpoint': item.endpoint}]}))
    assert configured_runtime(config_path).backend_name == 'custom'
    assert configured_runtime(config_path, robot).backend_name == robot
    from colosseum_policy_server import local_server
    selected = []
    async def capture(config):
        selected.append(config.backend_name)
    monkeypatch.setattr(local_server, 'run', capture)
    monkeypatch.setattr('sys.argv', ['colosseum-policy-local', '--robot', robot, '--config', str(config_path)])
    local_server.main()
    assert selected == [robot]


@pytest.mark.parametrize('failure', ['none', 'start', 'infer', 'timeout', 'disconnect'])
async def test_optional_session_hooks_always_cleanup(tmp_path, failure):
    events = []
    class Stateful(Backend):
        async def start_session(self, item, request):
            events.append(('start', request['run_id']))
            if failure == 'start':
                raise RuntimeError('startup failed')
        async def infer(self, item, obs):
            events.append(('infer', obs.control_step))
            if failure == 'infer':
                raise RuntimeError('inference failed')
            if failure == 'timeout':
                await asyncio.sleep(1)
            return np.zeros((1, 8))
        async def end_session(self):
            events.append(('end',))
    close = pb.RelayFrame(type=pb.SESSION_CLOSE, session_id='run').SerializeToString()
    messages = [prepare()] if failure == 'disconnect' else [prepare(), observation(), close]
    await LocalPolicyRuntime(config(tmp_path), Stateful()).handle(Socket(messages))
    assert events[0] == ('start', 'run')
    assert events[-1] == ('end',)
    assert events.count(('end',)) == 1


async def test_session_hook_cleanup_on_handler_cancellation(tmp_path):
    entered, cleaned = asyncio.Event(), asyncio.Event()
    class Stateful(Backend):
        async def start_session(self, item, request):
            entered.set()
            await asyncio.Future()
        async def end_session(self):
            cleaned.set()
    runtime = LocalPolicyRuntime(config(tmp_path), Stateful())
    task = asyncio.create_task(runtime.handle(Socket([prepare()])))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and not runtime.lock.locked()


def test_cloud_identity_does_not_relax_legacy_validation():
    from dataclasses import asdict
    row = asdict(model())
    for change in ({'revision': 'alias'}, {'endpoint': 'https://api.openai.com/v1'},
                   {'url': 'llm://unknown/model', 'endpoint': 'inprocess://inspect-agent'},
                   {'url': 'llm://openai/gpt-6-astra', 'endpoint': 'inprocess://inspect-agent', 'launcher': ['bad']}):
        with pytest.raises(ValueError):
            LocalModel.from_mapping({**row, 'launcher': [], **change})


@pytest.mark.parametrize('backend', [None, {}, {'options': {'external_sensor': 'front'}}])
def test_config_defaults_to_auto_backend(tmp_path, backend):
    from dataclasses import asdict
    import yaml
    row = asdict(model())
    row['launcher'] = list(row['launcher'])
    raw = {'models': [row]}
    if backend is not None:
        raw['backend'] = backend
    path = tmp_path / 'runtime.yaml'
    path.write_text(yaml.safe_dump(raw))
    cfg = RuntimeConfig.from_yaml(path)
    assert cfg.backend_name == 'auto'
    assert cfg.backend_options == (backend or {}).get('options', {})
