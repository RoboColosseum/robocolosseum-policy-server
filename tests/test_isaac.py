"""Perceptron Isaac 0.5 worker with stand-in torch/LeRobot modules; loads no weights."""
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_model_workers import module, Tensor
from colosseum_policy_server import model_service as service
from colosseum_policy_server.isaac import CONTRACTS, IsaacRuntime

# PerceptronAI/Isaac-0.5-SO100_101 @ 296cb4d436c7f4cc914d6edf1a546a2d77cac1c5 and
# PerceptronAI/Isaac-0.5-YAM @ dc52f2565cf76a9dd8dff87a386c945b9e81bdfd, recorded_policy/config.json
FIXTURES = {robot: Path(__file__).parent / f'fixtures/isaac_{robot}_config.json' for robot in ('so101', 'yam')}


def checkpoint(path, robot):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text(FIXTURES[robot].read_text())
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        (path / name).write_text('{}')
    (path / 'hf_model').mkdir(exist_ok=True)
    return path


@pytest.fixture
def fake_lerobot(monkeypatch):
    calls = {'reset': 0, 'batches': []}
    module(monkeypatch, 'torch', from_numpy=Tensor, inference_mode=contextlib.nullcontext)

    def config_load(path, **kwargs):
        assert kwargs == {'local_files_only': True}
        calls['config'] = SimpleNamespace(path=path, device='cuda', num_inference_steps=10)
        return calls['config']
    module(monkeypatch, 'lerobot.configs', PreTrainedConfig=SimpleNamespace(from_pretrained=config_load))

    class Policy:
        @classmethod
        def from_pretrained(cls, path, *, config, local_files_only):
            assert local_files_only and config is calls['config']
            policy = cls()
            policy.config = config
            return policy
        def reset(self):
            calls['reset'] += 1
        def predict_action_chunk(self, batch):
            assert batch == 'processed'
            return 'normalized'
    module(monkeypatch, 'lerobot.policies.perceptron_isaac', PerceptronIsaacPolicy=Policy)

    def processors(config, **kwargs):
        calls['processors'] = kwargs
        def pre(batch):
            calls['batches'].append(batch)
            return 'processed'
        def post(value):
            assert value == 'normalized'
            state = calls['batches'][-1]['observation.state'].value  # (1, D), echoed as a 60-step chunk
            return Tensor(np.tile(state, (1, 60, 1)) + np.arange(60, dtype=np.float32)[None, :, None] / 100)
        return pre, post
    module(monkeypatch, 'lerobot.policies.factory', make_pre_post_processors=processors)
    return calls


def payload(robot):
    state = np.linspace(-1, 1, len(CONTRACTS[robot].names)).astype(np.float32)
    state[list(CONTRACTS[robot].grippers)] = .5
    return dict(state=state, instruction='stack the cups', **{
        key: np.full((8, 12, 3), 50 * (i + 1), np.uint8) for i, (key, _) in enumerate(CONTRACTS[robot].cameras)})


@pytest.mark.parametrize('robot', ['so101', 'yam'])
def test_loads_saved_processors_warms_up_and_serves_full_chunk(tmp_path, fake_lerobot, robot):
    path = checkpoint(tmp_path / robot, robot)
    worker = IsaacRuntime(path, robot_type=robot, device='cuda:1', num_steps=7)
    calls = fake_lerobot
    assert calls['config'].device == 'cuda:1' and calls['config'].num_inference_steps == 7
    assert calls['processors'] == {
        'pretrained_path': str(path),
        'preprocessor_overrides': {'device_processor': {'device': 'cuda:1'}},
        'postprocessor_overrides': {'device_processor': {'device': 'cpu'}}}
    # Warm-up ran once at load, with open grippers, before any request.
    assert calls['reset'] == 1 and len(calls['batches']) == 1
    warm = calls['batches'][0]['observation.state'].value[0]
    np.testing.assert_array_equal(warm[list(CONTRACTS[robot].grippers)], CONTRACTS[robot].gripper_high)

    data = payload(robot)
    result = worker.infer(data)['actions']
    assert result.shape == (60, len(data['state'])) and result.dtype == np.float32
    np.testing.assert_allclose(result[0], data['state'])
    assert calls['reset'] == 2  # every request starts from a cleared policy
    batch = calls['batches'][-1]
    assert batch['task'] == ['stack the cups']
    assert batch['observation.state'].value.shape == (1, len(data['state']))
    for i, (source, target) in enumerate(CONTRACTS[robot].cameras):
        image = batch[target].value
        assert image.shape == (1, 3, 8, 12) and image.dtype == np.float32
        np.testing.assert_allclose(image, 50 * (i + 1) / 255)
    assert set(batch) == {'task', 'observation.state', *(t for _, t in CONTRACTS[robot].cameras)}


def test_so101_wrist_fills_top_and_external_fills_side():
    # Model card sample: the wrist image goes in the policy's top slot. Request fields keep the
    # SO101 adapter's (external_cam, wrist_cam) order.
    assert CONTRACTS['so101'].cameras == (('external_cam', 'observation.images.side'),
                                          ('wrist_cam', 'observation.images.top'))


@pytest.mark.parametrize('robot', ['so101', 'yam'])
def test_return_steps_and_gripper_saturation(tmp_path, fake_lerobot, robot):
    worker = IsaacRuntime(checkpoint(tmp_path, robot), robot_type=robot, return_steps=16, warmup=False)
    high, grippers = CONTRACTS[robot].gripper_high, list(CONTRACTS[robot].grippers)
    actions = np.zeros((1, 60, len(CONTRACTS[robot].names)), np.float32)
    actions[0, 2, grippers[0]] = -.2
    actions[0, 3, grippers[-1]] = high + 3
    actions[0, 4, 1] = 123.0  # arm joints are never clipped
    worker.post = lambda value: Tensor(actions)
    result = worker.infer(payload(robot))['actions']
    assert result.shape == (16, len(CONTRACTS[robot].names))
    assert result[2, grippers[0]] == 0 and result[3, grippers[-1]] == high and result[4, 1] == 123.0


@pytest.mark.parametrize('bad', ['horizon', 'width', 'nan'])
def test_bad_model_output_rejected(tmp_path, fake_lerobot, bad):
    worker = IsaacRuntime(checkpoint(tmp_path, 'yam'), robot_type='yam', warmup=False)
    actions = {'horizon': np.zeros((1, 30, 14)), 'width': np.zeros((1, 60, 64)),
               'nan': np.full((1, 60, 14), np.nan)}[bad]
    worker.post = lambda value: Tensor(actions)
    with pytest.raises(ValueError, match='finite 60 x 14'):
        worker.infer(payload('yam'))


@pytest.mark.parametrize('bad', ['state', 'nan', 'instruction', 'image'])
def test_bad_inputs_rejected(tmp_path, fake_lerobot, bad):
    worker = IsaacRuntime(checkpoint(tmp_path, 'so101'), robot_type='so101', warmup=False)
    data = payload('so101')
    if bad == 'state': data['state'] = np.zeros(14, np.float32)
    elif bad == 'nan': data['state'][0] = np.nan
    elif bad == 'instruction': data['instruction'] = ' '
    else: data['wrist_cam'] = np.zeros((8, 12, 3), np.float32)
    with pytest.raises(ValueError):
        worker.infer(data)


@pytest.mark.parametrize('robot,field,value', [
    ('yam', 'type', 'pi05'), ('yam', 'chunk_size', 30), ('yam', 'n_obs_steps', 2),
    ('yam', 'action_feature_names', ['wrong']), ('so101', 'target_fps', 15.0),
    ('so101', 'joint_signs', [1, -1, 1, 1, 1, 1]), ('so101', 'input_features', {})])
def test_wrong_checkpoint_rejected_before_loading(tmp_path, fake_lerobot, robot, field, value):
    path = checkpoint(tmp_path, robot)
    raw = json.loads((path / 'config.json').read_text())
    raw[field] = value
    (path / 'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        IsaacRuntime(path, robot_type=robot)
    assert 'config' not in fake_lerobot  # nothing loaded


def test_checkpoint_for_other_robot_and_missing_files_rejected(tmp_path, fake_lerobot):
    with pytest.raises(ValueError, match='14-D'):
        IsaacRuntime(checkpoint(tmp_path / 'a', 'so101'), robot_type='yam')
    path = checkpoint(tmp_path / 'b', 'yam')
    (path / 'hf_model').rmdir()
    with pytest.raises(FileNotFoundError, match='hf_model'):
        IsaacRuntime(path, robot_type='yam')


def test_cli_validates_and_dispatches_isaac(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'so101', 'so101')
    base = ['isaac', '--checkpoint', str(path), '--port', '9207']
    args = service.parse_args([*base, '--robot-type', 'so101', '--isaac-return-steps', '30'])
    assert args.isaac_return_steps == 30 and args.checkpoint == path.resolve()
    for extra in (['--robot-type', 'franka'], ['--robot-type', 'yam'],
                  ['--robot-type', 'so101', '--isaac-return-steps', '61']):
        with pytest.raises(SystemExit):
            service.parse_args([*base, *extra])

    seen = {}
    class Runtime:
        def __init__(self, checkpoint, **kwargs):
            seen.update(kwargs, checkpoint=checkpoint)
    class Server:
        def __init__(self, address, handler): seen['address'] = address
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def serve_forever(self): seen['served'] = True
    import colosseum_policy_server.isaac as isaac
    monkeypatch.setattr(isaac, 'IsaacRuntime', Runtime)
    monkeypatch.setattr(service, 'HTTPServer', Server)
    service.main([*base, '--robot-type', 'so101', '--device', 'cuda:2', '--num-steps', '8'])
    assert seen == {'checkpoint': path.resolve(), 'robot_type': 'so101', 'device': 'cuda:2', 'num_steps': 8,
                    'return_steps': 60, 'address': ('127.0.0.1', 9207), 'served': True}
