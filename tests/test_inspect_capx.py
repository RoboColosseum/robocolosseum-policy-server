"""CaP-X code-as-policy sessions for SO-101 and bimanual YAM joint contracts."""

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('inspect_robots')
from inspect_robots.types import Action, ActionChunk

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.inspect_capx import (
    YAM_READY_SEED, InspectCapxBackend, Sam3Launcher, _listening, yam_tool_urdf)

# Six YAM arm joints (i2rt yam.urdf kinematics) ending at the gripper mount.
YAM_URDF = """<?xml version="1.0"?>
<robot name="yam">
  <link name="base"/><link name="link1"/><link name="link2"/><link name="link3"/>
  <link name="link4"/><link name="link5"/>
  <link name="gripper"><visual><geometry><mesh filename="assets/gripper.stl"/></geometry></visual></link>
  <joint name="joint1" type="revolute"><origin xyz="0 0 0.068" rpy="0 1.5708 3.1416"/>
    <axis xyz="-1 0 0"/><parent link="base"/><child link="link1"/>
    <limit effort="1" velocity="1" lower="-2.61799" upper="3.14159"/></joint>
  <joint name="joint2" type="revolute"><origin xyz="-0.0455 -0.0339 -0.02" rpy="0 0 -3.14159"/>
    <axis xyz="0 1 0"/><parent link="link1"/><child link="link2"/>
    <limit effort="1" velocity="1" lower="0" upper="3.66519"/></joint>
  <joint name="joint3" type="revolute"><origin xyz="0 -0.0688 0.264" rpy="0 0 -3.14159"/>
    <axis xyz="0 1 0"/><parent link="link2"/><child link="link3"/>
    <limit effort="1" velocity="1" lower="0" upper="3.14159"/></joint>
  <joint name="joint4" type="revolute"><origin xyz="-0.0600003 -0.0688 -0.244999" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><parent link="link3"/><child link="link4"/>
    <limit effort="1" velocity="1" lower="-1.69297" upper="1.5708"/></joint>
  <joint name="joint5" type="revolute"><origin xyz="-0.0405003 0.0338995 -0.0739989" rpy="0 0 0"/>
    <axis xyz="1 0 0"/><parent link="link4"/><child link="link5"/>
    <limit effort="1" velocity="1" lower="-1.5708" upper="1.5708"/></joint>
  <joint name="joint6" type="revolute"><origin xyz="0.0404996 0 -0.0356" rpy="3.1415926535897927 0 1.5707963267949736"/>
    <axis xyz="0 0 1"/><parent link="link5"/><child link="gripper"/>
    <limit effort="1" velocity="1" lower="-2.0944" upper="2.0944"/></joint>
</robot>
"""
JOINT_LOW = [-2.5, -0.05, -0.05, -1.68, -1.5, -2.0]
JOINT_HIGH = [3.0, 3.5, 3.0, 1.5, 1.5, 2.0]
SENSORS = ('head_image', 'left_image', 'right_image')


class Policy:
    """Records what the backend hands capx and replays scripted chunks."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chunks = []
        self.observations = []
        self.closed = False

    def bind(self, info):
        self.info = info

    def reset(self, scene):
        self.scene = scene

    def act(self, observation):
        self.observations.append(observation)
        return self.chunks.pop(0)

    def transcript(self):
        return [{'role': 'system', 'content': 'capx'}]

    def close(self):
        self.closed = True


class Kinematics:
    """Fake ArmKinematics: FK puts the tool 1 mm forward per joint degree."""

    def __init__(self, urdf, *, tip_frame, joint_names, joint_low_deg, joint_high_deg):
        self.urdf_path = Path(urdf)
        self.urdf_text = self.urdf_path.read_text()
        self.tip_frame = tip_frame
        self.joint_names = joint_names
        self.low, self.high = np.asarray(joint_low_deg), np.asarray(joint_high_deg)
        self.solves = []
        self.failures = 0
        self.answer = np.full(6, 10.0)

    def forward(self, joints_deg):
        pose = np.eye(4)
        pose[0, 3] = 0.1 + float(np.sum(joints_deg)) / 1000
        return pose

    def solve(self, position, quaternion, seed_deg, *, position_tolerance_m, orientation_weight):
        self.solves.append((np.asarray(position), np.asarray(quaternion), np.asarray(seed_deg),
                            orientation_weight))
        if self.failures:
            self.failures -= 1
            raise ValueError('IK cannot reach position')
        return SimpleNamespace(joints=self.answer.copy(), position_error_m=0.0,
                               orientation_error_deg=0.0)


def chunk(*rows, control_hz=30, **meta):
    return ActionChunk(actions=tuple(Action(np.asarray(row, dtype=np.float64), meta=meta)
                                     for row in rows), control_hz=control_hz)


def tensor(data):
    array = np.asarray(data, dtype=np.float32)
    return pb.Tensor(shape=array.shape, dtype=pb.FLOAT32, data=array.tobytes())


def yam_observation(step=0, joints=None, grippers=(1.0, 1.0), depth=True):
    value = pb.Observation(instruction='Hand the cube from left to right', control_step=step)
    value.state['joint_position'].CopyFrom(tensor(np.zeros(12) if joints is None else joints))
    value.state['gripper_position'].CopyFrom(tensor(grippers))
    if depth:
        value.state['head_image_depth_m'].CopyFrom(tensor(np.full((2, 2), 0.5)))
        value.state['head_image_intrinsics'].CopyFrom(tensor(np.eye(3).reshape(-1)))
    for name in SENSORS:
        value.sensors.add(sensor_id=name, encoding=pb.RAW_RGB, width=2, height=2, data=bytes(12))
    return value


def yam_model(name='claude-opus-5-5-capx', max_horizon=150):
    return LocalModel.from_mapping(dict(
        name=name, model_type='llm', url='https://api.yhlxj.ai', revision='rig-v1',
        endpoint='inprocess://inspect-agent', action_space='joint_position', action_dim=14,
        control_hz=30, max_horizon=max_horizon, backend_options={'robot_type': 'yam'}))


def yam_backend(tmp_path, **overrides):
    urdf = tmp_path / 'yam' / 'yam.urdf'
    urdf.parent.mkdir(exist_ok=True)
    urdf.write_text(YAM_URDF)
    options = {
        'robot_type': 'yam', 'joint_low': JOINT_LOW * 2, 'joint_high': JOINT_HIGH * 2,
        'joint_max_step': [0.15] * 12, 'gripper_open_value': 1, 'sensors': list(SENSORS),
        'robot_notes': 'Dual YAM arms facing the table.', 'max_speed_frac': 0.1,
        'log_dir': str(tmp_path / 'logs'), 'urdf_path': str(urdf),
    }
    options.update(overrides)
    kinematics = []

    def kinematics_factory(*args, **kwargs):
        kinematics.append(Kinematics(*args, **kwargs))
        return kinematics[-1]

    policies = []

    def policy_factory(**kwargs):
        policies.append(Policy(**kwargs))
        return policies[-1]

    backend = InspectCapxBackend(options, policy_factory=policy_factory,
                                 kinematics_factory=kinematics_factory,
                                 server_check=lambda url: None)
    return backend, policies, kinematics


def start(backend, model=None, **request):
    asyncio.run(backend.start_session(model or yam_model(), {
        'run_id': 'run-1', 'robot_type': 'yam', 'api_key': 'test-key', **request}))


def test_yam_session_configures_capx_for_two_arms(tmp_path):
    backend, policies, kinematics = yam_backend(tmp_path)
    start(backend)

    kwargs = policies[0].kwargs
    assert kwargs['model'] == 'claude-opus-5-5' and kwargs['wire'] == 'chat'
    assert kwargs['base_url'] == 'https://api.yhlxj.ai/v1'
    assert kwargs['env'] == {'ANTHROPIC_API_KEY': 'test-key'}
    assert kwargs['ik'] == 'embodiment' and kwargs['camera'] == 'head_image'
    # YAM pacing comes from the declared capx step, not the shared llm speed fraction.
    assert kwargs['max_speed_frac'] == 0.1 and kwargs['gripper_speed_frac'] == 0.1
    assert backend.observation_extras == ('depth',)
    assert backend.internal_action_dim == 14
    assert [k.tip_frame for k in kinematics] == ['grasp_site', 'grasp_site']
    assert kinematics[0].joint_names == tuple(f'joint{i}' for i in range(1, 7))
    # Rig limits intersected with the URDF ranges (joint2/3 cannot go below 0).
    np.testing.assert_allclose(kinematics[0].low, np.rad2deg([-2.5, 0, 0, -1.68, -1.5, -2.0]))
    np.testing.assert_allclose(kinematics[0].high, np.rad2deg([3.0, 3.5, 3.0, 1.5, 1.5, 2.0]))
    assert '<child link="grasp_site" />' in kinematics[0].urdf_text
    assert 'xyz="0 0 0.14465"' in kinematics[0].urdf_text
    assert str((tmp_path / 'yam' / 'assets' / 'gripper.stl').resolve()) in kinematics[0].urdf_text
    assert not kinematics[0].urdf_path.exists()  # the patched copy is removed after loading


def test_yam_binds_the_multi_arm_profile_and_sends_both_arm_poses(tmp_path):
    backend, policies, _ = yam_backend(tmp_path)
    start(backend)
    policy = policies[0]
    policy.chunks.append(chunk(np.r_[np.full(6, 0.01), 1, np.zeros(6), 1]))

    actions = backend._infer(yam_observation())

    np.testing.assert_allclose(actions, [np.r_[np.full(6, 0.01), 1, np.zeros(6), 1]])
    space = policy.info.action_space
    assert space.shape == (14,)
    assert space.semantics.dim_labels[6] == 'left_gripper'
    assert space.semantics.dim_labels[13] == 'right_gripper'
    np.testing.assert_allclose(space.semantics.max_step,
                               np.r_[np.full(6, 0.01), 1 / 30, np.full(6, 0.01), 1 / 30])
    assert [field.key for field in policy.info.observation_space.state.fields] == [
        'joint_pos', 'gripper_measured', 'left_eef_pos', 'left_eef_quat_wxyz',
        'right_eef_pos', 'right_eef_quat_wxyz']
    assert 'move_arms(left=..., right=...)' in policy.info.docs
    assert "that arm's own base frame" in policy.info.docs
    assert 'Dual YAM arms facing the table.' in policy.info.docs
    obs = policy.observations[0]
    assert set(obs.extra['ik_solver']) == {'left', 'right'}
    np.testing.assert_allclose(obs.extra['depth'], np.full((2, 2), 0.5))
    np.testing.assert_allclose(obs.extra['intrinsics'], np.eye(3))
    np.testing.assert_allclose(obs.state['left_eef_pos'], [0.1, 0, 0])
    np.testing.assert_allclose(obs.state['left_eef_quat_wxyz'], [1, 0, 0, 0])


def test_yam_ik_solver_works_in_radians_and_retries_from_the_ready_pose(tmp_path):
    backend, policies, kinematics = yam_backend(tmp_path)
    start(backend)
    policies[0].chunks.append(chunk(np.r_[np.zeros(6), 1, np.zeros(6), 1]))
    backend._infer(yam_observation())
    solver = policies[0].observations[0].extra['ik_solver']['right']
    kinematics[1].failures = 1

    joints = solver(np.array([0.3, 0.0, 0.1]), np.array([0, 0, 1.0, 0]), np.full(6, 0.2))

    np.testing.assert_allclose(joints, np.deg2rad(np.full(6, 10.0)))
    seeds = [seed for _, _, seed, _ in kinematics[1].solves]
    np.testing.assert_allclose(seeds[0], np.rad2deg(np.full(6, 0.2)))
    np.testing.assert_allclose(seeds[1], np.rad2deg(YAM_READY_SEED))
    assert kinematics[1].solves[0][3] == 1.0  # six joints: full orientation weight
    assert kinematics[0].solves == []

    kinematics[1].failures = 2
    with pytest.raises(ValueError, match='cannot reach'):
        solver(np.array([2.0, 0.0, 0.1]), np.array([0, 0, 1.0, 0]), np.zeros(6))


def test_yam_arm_base_poses_put_both_arms_in_one_rig_frame(tmp_path):
    backend, policies, kinematics = yam_backend(
        tmp_path, arm_base_poses={'left': [0, 0.3, 0, 0], 'right': [0, -0.3, 0, 180]})
    start(backend)
    policies[0].chunks.append(chunk(np.r_[np.zeros(6), 1, np.zeros(6), 1]))
    backend._infer(yam_observation())
    obs = policies[0].observations[0]

    np.testing.assert_allclose(obs.state['left_eef_pos'], [0.1, 0.3, 0], atol=1e-12)
    np.testing.assert_allclose(obs.state['right_eef_pos'], [-0.1, -0.3, 0], atol=1e-12)
    assert 'one shared rig frame' in policies[0].info.docs
    obs.extra['ik_solver']['right'](np.array([0.0, -0.5, 0.2]), np.array([1.0, 0, 0, 0]), np.zeros(6))
    position, quaternion, _, _ = kinematics[1].solves[0]
    np.testing.assert_allclose(position, [0, 0.2, 0.2], atol=1e-12)
    np.testing.assert_allclose(np.abs(quaternion), [0, 0, 0, 1], atol=1e-12)


def test_yam_plans_from_commanded_grippers_and_holds_them_on_finish(tmp_path):
    backend, policies, _ = yam_backend(tmp_path)
    start(backend)
    policy = policies[0]
    closing = [np.r_[np.zeros(6), value, np.zeros(6), 1] for value in (0.9, 0.8, 0.7)]
    policy.chunks.append(chunk(*closing))
    backend._infer(yam_observation())

    # The jaws stopped on an object at 0.75, but the command was 0.7.
    joints = np.r_[np.full(6, 0.02), np.zeros(6)]
    packed = np.r_[joints[:6], 0.7, joints[6:], 1]
    policy.chunks.append(chunk(np.ones(14), stop_reason='FINISH', request_stop=True))
    held = backend._infer(yam_observation(step=3, joints=joints, grippers=(0.75, 1.0)))

    state = policy.observations[1].state
    np.testing.assert_allclose(state['joint_pos'][[6, 13]], [0.7, 1.0], rtol=1e-6)
    np.testing.assert_allclose(state['gripper_measured'], [0.75, 1.0])
    np.testing.assert_allclose(held, [packed], rtol=1e-6)
    assert backend.stop_reason == 'done' and backend.stop_step == 3
    np.testing.assert_allclose(backend._infer(yam_observation(step=4, joints=joints)), held)

    asyncio.run(backend.end_session())
    log = json.loads(next((tmp_path / 'logs').glob('*.json')).read_text())
    assert log['robot_type'] == 'yam' and log['policy'] == 'capx' and log['stop_reason'] == 'done'
    assert policy.closed


def test_yam_splits_long_plans_at_the_router_horizon(tmp_path):
    backend, policies, _ = yam_backend(tmp_path)
    start(backend, model=yam_model(max_horizon=2))
    rows = [np.r_[np.full(6, 0.01 * i), 1, np.zeros(6), 1] for i in range(1, 6)]
    policies[0].chunks.append(chunk(*rows))

    first = backend._infer(yam_observation())
    second = backend._infer(yam_observation(step=2, joints=np.r_[np.full(6, 0.02), np.zeros(6)]))
    third = backend._infer(yam_observation(step=4, joints=np.r_[np.full(6, 0.04), np.zeros(6)]))

    np.testing.assert_allclose(np.vstack((first, second, third)), rows, rtol=1e-6)
    assert len(policies[0].observations) == 1  # no new LLM turn while the plan drains


def test_end_session_stops_a_plan_the_client_gave_up_on(tmp_path):
    backend, policies, _ = yam_backend(tmp_path)
    start(backend)
    policy = policies[0]
    llm_call = threading.Event()

    def act(observation):
        llm_call.wait(10)  # an LLM request in flight; it fails once the client is closed
        raise RuntimeError('LLM client was closed')

    def close():
        policy.closed = True
        llm_call.set()

    policy.act, policy.close = act, close

    async def client_gives_up():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(backend.infer(yam_model(), yam_observation()), 0.2)
        started = time.monotonic()
        await backend.end_session()
        return time.monotonic() - started

    assert asyncio.run(client_gives_up()) < 5
    assert policy.closed and backend.policy is None and backend._inflight is None


def test_yam_rejects_capx_steps_beyond_the_rig_limit(tmp_path):
    backend, policies, _ = yam_backend(tmp_path)
    start(backend)
    policies[0].chunks.append(chunk(np.r_[0.2, np.zeros(5), 1, np.zeros(6), 1]))

    with pytest.raises(ValueError, match='left_joint1 delta=0.200000 limit=0.150000'):
        backend._infer(yam_observation())


@pytest.mark.parametrize(('overrides', 'request_extra', 'message'), [
    ({'urdf_path': None}, {}, 'urdf_path'),
    ({'sensors': ['head_image', 'left_image']}, {}, 'right_image'),
    ({'gripper_open_value': 0}, {}, 'gripper_open_value: 1'),
    ({'joint_speed_rad_s': 0}, {}, 'joint_speed_rad_s must be a positive number'),
    ({'arm_base_poses': {'left': [0, 0, 0, 0]}}, {}, 'arm_base_poses'),
    ({'tool_offset_xyz': [0, 0]}, {}, 'tool_offset_xyz must contain 3'),
    ({}, {'auth_method': 'claude-code'}, 'API key'),
    ({'sam3_launcher': 'python -m sam3_server'}, {}, 'sam3_launcher must be a nonempty list'),
    ({'sam3_launcher': ['python', '--port', '9999']}, {}, r'must match the sam3_url port \(8114\)'),
    ({'sam3_launcher': ['python'], 'sam3_url': 'http://10.0.0.5:8114'}, {}, 'loopback sam3_url'),
    ({'sam3_launcher': ['python'], 'sam3_start_timeout_s': 0}, {}, 'sam3_start_timeout_s'),
], ids=['no_urdf', 'two_sensors', 'gripper_polarity', 'speed', 'base_poses', 'tool_offset', 'login',
        'sam3_command', 'sam3_port', 'sam3_host', 'sam3_timeout'])
def test_yam_session_rejects_invalid_setup(tmp_path, overrides, request_extra, message):
    backend, _, _ = yam_backend(tmp_path, **overrides)

    with pytest.raises(ValueError, match=message):
        start(backend, **request_extra)
    assert backend.policy is None and backend.sam3 is None


# Stands in for the SAM3 server: loads for a moment, then listens on --port.
FAKE_SAM3 = """import socket, sys, time
time.sleep(0.3)
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])))
server.listen()
print('Uvicorn running', flush=True)
time.sleep(60)
"""


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def sam3_backend(tmp_path, port, launcher, **overrides):
    return yam_backend(tmp_path, sam3_url=f'http://127.0.0.1:{port}',
                       sam3_launcher=launcher, **overrides)


def session(backend, inside=None):
    """Start and end one session on one event loop, as the runtime does."""
    async def run():
        try:
            await backend.start_session(yam_model(), {
                'run_id': 'run-1', 'robot_type': 'yam', 'api_key': 'test-key'})
            return await inside() if inside else None
        finally:
            await backend.end_session()
    return asyncio.run(run())


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_sam3_launcher_starts_sam3_for_the_session_and_stops_it_after(tmp_path):
    port = free_port()
    backend, policies, _ = sam3_backend(
        tmp_path, port, [sys.executable, '-c', FAKE_SAM3, '--port', port])

    async def inside():
        launcher = backend.sam3
        assert await _listening(f'http://127.0.0.1:{port}') and policies
        return launcher.process.pid, launcher.log_path

    pid, log = session(backend, inside)

    assert gone(pid) and backend.sam3 is None
    assert not asyncio.run(_listening(f'http://127.0.0.1:{port}'))
    assert log.parent == tmp_path / 'logs' and 'Uvicorn running' in log.read_text()


def test_sam3_launcher_leaves_a_running_server_alone(tmp_path):
    port = free_port()
    with socket.socket() as running:
        running.bind(('127.0.0.1', port))
        running.listen()
        backend, policies, _ = sam3_backend(
            tmp_path, port, [sys.executable, '-c', 'raise SystemExit(3)', '--port', str(port)])

        session(backend, lambda: asyncio.sleep(0))

        assert policies and backend.sam3 is None
        assert asyncio.run(_listening(f'http://127.0.0.1:{port}'))


@pytest.mark.parametrize(('launcher', 'error', 'message'), [
    (['-c', 'raise SystemExit(3)'], RuntimeError, 'exited with status 3'),
    (['-c', 'import time; time.sleep(60)'], RuntimeError, 'did not listen .* within 1 s'),
    (None, ValueError, 'could not run'),
], ids=['exits', 'never_listens', 'missing_program'])
def test_sam3_launcher_failures_fail_the_session_and_leave_nothing_running(
        tmp_path, monkeypatch, launcher, error, message):
    port = free_port()
    command = [sys.executable, *launcher] if launcher else [str(tmp_path / 'missing' / 'python')]
    backend, policies, _ = sam3_backend(tmp_path, port, command, sam3_start_timeout_s=1)
    started = []
    original = Sam3Launcher.start

    async def spy(self):
        started.append(self)
        await original(self)

    monkeypatch.setattr(Sam3Launcher, 'start', spy)
    with pytest.raises(error, match=message):
        session(backend)

    assert not policies and backend.sam3 is None and backend.policy is None
    process = started[0].process
    assert process is None or (process.returncode is not None and gone(process.pid))


def test_yam_tool_urdf_requires_the_mount_link(tmp_path):
    urdf = tmp_path / 'yam.urdf'
    urdf.write_text(YAM_URDF.replace('"gripper"', '"flange"'))
    with pytest.raises(ValueError, match="no link 'gripper'"):
        yam_tool_urdf(urdf, tmp_path)


def so101_observation(step=0):
    value = pb.Observation(instruction='Pick the cube', control_step=step)
    value.state['joint_position'].CopyFrom(tensor([0] * 5))
    value.state['gripper_position'].CopyFrom(tensor([40]))
    for name in ('head_image', 'left_image'):
        value.sensors.add(sensor_id=name, encoding=pb.RAW_RGB, width=2, height=2, data=bytes(12))
    return value


def test_so101_capx_keeps_its_six_value_contract(tmp_path):
    policies = []

    def policy_factory(**kwargs):
        policies.append(Policy(**kwargs))
        return policies[-1]

    backend = InspectCapxBackend({
        'robot_type': 'so101', 'sensors': ['head_image', 'left_image'],
        'joint_low': [-120, -104, -98, -102, -180], 'joint_high': [120, 104, 98, 102, 180],
        'joint_max_step': [1, 1, 1, 1, 2], 'gripper_open_value': 1,
        'robot_notes': 'Calibrated SO-101.', 'log_dir': str(tmp_path), 'urdf_path': 'so101.urdf',
        'capx_depth': False,
    }, policy_factory=policy_factory,
        kinematics_factory=lambda *args, **kwargs: SimpleNamespace(forward=lambda joints: np.eye(4)),
        server_check=lambda url: None)
    model = LocalModel.from_mapping(dict(
        name='grok-4.7-capx', model_type='llm', url='https://api.yhlxj.ai/v1', revision='rig-v1',
        endpoint='inprocess://inspect-agent', action_space='joint_position', action_dim=6,
        control_hz=10, max_horizon=150, backend_options={'robot_type': 'so101'}))
    asyncio.run(backend.start_session(model, {'run_id': 'r', 'robot_type': 'so101',
                                              'api_key': 'key'}))
    policy = policies[0]
    assert policy.kwargs['max_speed_frac'] == 0.1 and policy.kwargs['gripper_speed_frac'] == 0.5
    assert policy.kwargs['model'] == 'grok-4.7' and backend.observation_extras == ()
    policy.chunks.append(chunk([0.5, 0, 0, 0, 0, 43], control_hz=10))

    np.testing.assert_allclose(backend._infer(so101_observation()), [[0.5, 0, 0, 0, 0, 43]])
    assert policy.info.name == 'so101_capx' and policy.info.action_space.shape == (6,)
    np.testing.assert_allclose(policy.info.action_space.semantics.max_step, [1, 1, 1, 1, 2, 5])
    np.testing.assert_allclose(policy.observations[0].state['joint_pos'], [0, 0, 0, 0, 0, 40])
    assert 'No depth is available on this rig.' in policy.info.docs

    policy.chunks.append(chunk([0.5, 0, 0, 0, 0, 43], control_hz=10, request_stop=True,
                               stop_reason='GIVE_UP'))
    np.testing.assert_allclose(backend._infer(so101_observation(step=1)), [[0, 0, 0, 0, 0, 43]])
    assert backend.stop_reason == 'give_up'


def test_yam_capx_end_to_end_with_real_kinematics_and_policy(tmp_path):
    """Real placo IK and CapxPolicy: one scripted LLM turn moves both arms."""
    pytest.importorskip('placo')
    httpx = pytest.importorskip('httpx')
    from inspect_robots_capx import CapxPolicy
    from inspect_robots_capx.kinematics import ArmKinematics

    replies = [
        "import numpy as np\n"
        "down = np.array([0.0, 0.0, 1.0, 0.0])\n"
        "left = solve_ik(np.array([0.30, 0.0, 0.12]), down, arm='left')\n"
        "right = solve_ik(np.array([0.32, 0.02, 0.10]), down, arm='right')\n"
        "move_arms(left=left, right=right)\n"
        "close_gripper(arm='left')\n",
        "FINISH",
    ]
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {
            'role': 'assistant', 'content': replies.pop(0)}}]})

    urdf = tmp_path / 'yam.urdf'
    urdf.write_text(YAM_URDF.replace('<mesh filename="assets/gripper.stl"/>', '<box size="0.01 0.01 0.01"/>'))
    backend = InspectCapxBackend({
        'robot_type': 'yam', 'joint_low': JOINT_LOW * 2, 'joint_high': JOINT_HIGH * 2,
        'joint_max_step': [0.15] * 12, 'gripper_open_value': 1, 'sensors': list(SENSORS),
        'robot_notes': 'Dual YAM.', 'log_dir': str(tmp_path / 'logs'), 'urdf_path': str(urdf),
    }, policy_factory=lambda **kwargs: CapxPolicy(**kwargs, transport=httpx.MockTransport(handler)),
        kinematics_factory=ArmKinematics, server_check=lambda url: None)
    start(backend, model=yam_model(name='grok-4.7-capx'))

    plan = [backend._infer(yam_observation(depth=False))]
    while backend.pending is not None:
        plan.append(backend._infer(yam_observation(step=150 * len(plan), depth=False)))
    actions = np.vstack(plan)

    assert 'move_arms(left=..., right=...)' in requests[0]['messages'][0]['content']
    steps = np.abs(np.diff(np.vstack((np.r_[np.zeros(6), 1, np.zeros(6), 1], actions)), axis=0))
    assert steps[:, :6].max() <= 0.01 + 1e-6 and steps[:, 6].max() <= 1 / 30 + 1e-6
    final = actions[-1]
    assert final[6] == 0 and final[13] == 1  # left gripper closed after the move
    for side, start_index, target in (('left', 0, [0.30, 0.0, 0.12]), ('right', 7, [0.32, 0.02, 0.10])):
        pose = backend._yam_forward(side, final[start_index:start_index + 6].astype(np.float64))
        np.testing.assert_allclose(pose[:3, 3], target, atol=0.006)
        np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], atol=0.2)  # approach points down

    stop = backend._infer(yam_observation(step=10_000, joints=np.r_[final[:6], final[7:13]],
                                          grippers=(0.1, 1.0), depth=False))
    np.testing.assert_allclose(stop[0, [6, 13]], [0, 1])
    assert backend.stop_reason == 'done'
