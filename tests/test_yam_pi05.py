import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_model_workers import module, Tensor
from colosseum_policy_server.yam_pi05 import Pi05YAMRuntime
from colosseum_policy_server import model_service as service


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # NUSMAGIC/pi05-MolmoAct2-YAM @ 2f28d00ac28c543626f5a4f579a3da09bee4a4ed
    raw = json.loads((Path(__file__).parent / 'fixtures/pi05_yam_config.json').read_text())
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    for name in ('model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json',
                 'policy_preprocessor_step_3_normalizer_processor.safetensors',
                 'policy_postprocessor_step_0_unnormalizer_processor.safetensors'):
        (tmp_path / name).touch()
    calls = {}
    module(monkeypatch, 'torch', from_numpy=lambda value: value, inference_mode=contextlib.nullcontext)
    module(monkeypatch, 'safetensors.torch', load_file=lambda path: {'weight': 42})
    cfg = SimpleNamespace()
    def config_load(path, **kwargs):
        assert kwargs == {'local_files_only': True}
        return cfg
    module(monkeypatch, 'lerobot.policies.pi05.configuration_pi05',
           PI05Config=SimpleNamespace(from_pretrained=config_load))
    class Policy:
        def __init__(self, config):
            assert config.num_inference_steps == 7
            assert not config.compile_model and not config.gradient_checkpointing
        def _fix_pytorch_state_dict_keys(self, weights, config): return weights
        def _prepare_pretrained_state_dict(self, weights): return weights
        def load_state_dict(self, weights, *, strict):
            assert strict and weights == {'model.weight': 42}
            calls['loaded'] = True
        def to(self, device): return self
        def eval(self): return self
        def reset(self): calls['reset'] = calls.get('reset', 0) + 1
        def predict_action_chunk(self, batch):
            assert calls['loaded'] and batch == 'processed'
            return 'normalized'
    module(monkeypatch, 'lerobot.policies.pi05.modeling_pi05', PI05Policy=Policy)
    def processors(config, **kwargs):
        calls['processors'] = kwargs
        def pre(batch):
            calls['batch'] = batch
            return 'processed'
        def post(value):
            assert value == 'normalized'
            return Tensor(np.tile(calls['batch']['observation.state'], (1, 30, 1)))
        return pre, post
    module(monkeypatch, 'lerobot.policies.factory', make_pre_post_processors=processors)
    return Pi05YAMRuntime(tmp_path, tmp_path, num_steps=7), calls, tmp_path, Policy


def payload():
    return dict(state=np.arange(14, dtype=np.float32) / 20, instruction='pick up cup', **{
        key: np.full((8, 12, 3), i * 70, np.uint8)
        for i, key in enumerate(Pi05YAMRuntime.image_keys)})


def test_saved_processors_camera_order_and_full_absolute_chunk(runtime):
    worker, calls, path, _ = runtime
    for _ in range(2):
        result = worker.infer(payload())
        np.testing.assert_array_equal(result['actions'], np.tile(payload()['state'], (30, 1)))
    assert calls['reset'] == 2
    batch = calls['batch']
    assert [key for key in batch if key.startswith('observation.images.')] == list(worker.camera_map.values())
    for i, key in enumerate(worker.camera_map.values()):
        assert batch[key].shape == (3, 8, 12)
        np.testing.assert_allclose(batch[key], i * 70 / 255)
    assert batch['task'] == 'pick up cup'
    assert calls['processors']['pretrained_path'] == str(path)
    assert calls['processors']['preprocessor_overrides']['tokenizer_processor'] == {'tokenizer_name': str(path)}


@pytest.mark.parametrize('bad', ['width', 'horizon', 'nan', 'gripper'])
def test_bad_actions_rejected(runtime, bad):
    worker, _, _, _ = runtime
    actions = np.zeros((1, 30, 14))
    if bad == 'width': actions = np.zeros((1, 30, 32))
    elif bad == 'horizon': actions = np.zeros((1, 16, 14))
    elif bad == 'nan': actions[0, 0, 0] = np.nan
    else: actions[0, 3, 6] = -0.1
    worker.post = lambda value: Tensor(actions)
    if bad == 'gripper':  # saturated, not rejected; arm joints untouched
        actions[0, 3, 5] = 2.5
        result = worker.infer(payload())['actions']
        assert result[3, 6] == 0 and result[3, 5] == 2.5
        return
    with pytest.raises(ValueError): worker.infer(payload())


@pytest.mark.parametrize('bad', ['state', 'nan', 'gripper', 'instruction', 'image'])
def test_bad_inputs_rejected(runtime, bad):
    worker, _, _, _ = runtime
    data = payload()
    if bad == 'state': data['state'] = np.zeros(12)
    elif bad == 'nan': data['state'][0] = np.nan
    elif bad == 'gripper': data['state'][13] = 1.1
    elif bad == 'instruction': data['instruction'] = ' '
    else: data['top_cam'] = np.zeros((8, 12, 3), np.float32)
    with pytest.raises(ValueError): worker.infer(data)


@pytest.mark.parametrize('field,value', [('use_relative_actions', True), ('chunk_size', 16),
    ('action_feature_names', ['wrong']), ('type', 'groot')])
def test_wrong_checkpoint_rejected(runtime, field, value):
    _, _, path, _ = runtime
    raw = json.loads((path / 'config.json').read_text())
    raw[field] = value
    (path / 'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError): Pi05YAMRuntime(path, path, num_steps=7)


def test_camera_reordering_rejected(runtime):
    _, _, path, _ = runtime
    raw = json.loads((path / 'config.json').read_text())
    raw['input_features'] = dict(sorted(raw['input_features'].items()))
    (path / 'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='top/left/right'): Pi05YAMRuntime(path, path, num_steps=7)


def test_missing_stats_and_failed_weights_stop_loading(runtime, monkeypatch):
    _, _, path, policy = runtime
    def fail(*args, **kwargs): raise RuntimeError('weight shape mismatch')
    monkeypatch.setattr(policy, 'load_state_dict', fail)
    with pytest.raises(RuntimeError, match='weight shape mismatch'):
        Pi05YAMRuntime(path, path, num_steps=7)
    (path / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors').unlink()
    with pytest.raises(FileNotFoundError): Pi05YAMRuntime(path, path, num_steps=7)


def test_cli_yam_http_dispatch_and_franka_stats_requirement(runtime, monkeypatch):
    from colosseum_policy_server import yam_pi05
    worker, _, path, _ = runtime
    args = ['pi05_lerobot', '--robot-type', 'yam', '--checkpoint', str(path),
            '--tokenizer', str(path), '--port', '8204']
    assert service.parse_args(args).stats is None
    with pytest.raises(SystemExit): service.parse_args(args + ['--stats', str(path / 'config.json')])
    with pytest.raises(SystemExit): service.parse_args([*args, '--robot-type', 'franka'])
    monkeypatch.setattr(yam_pi05, 'Pi05YAMRuntime', lambda *a, **k: worker)
    served = []
    class Server:
        def __init__(self, address, handler): assert address == ('127.0.0.1', 8204)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def serve_forever(self): served.append(True)
    monkeypatch.setattr(service, 'HTTPServer', Server)
    service.main(args)
    assert served == [True]


def test_local_runtime_example_matches_checkpoint():
    from colosseum_policy_server.local_runtime import RuntimeConfig
    from colosseum_policy_server.backends.vla import load_model_adapter
    config = RuntimeConfig.from_yaml(Path(__file__).parents[1] / 'configs/local-runtime-yam-pi05.yaml.example')
    model = config.models['pi05-yam']
    assert model.revision == '2f28d00ac28c543626f5a4f579a3da09bee4a4ed'
    assert model.max_horizon == 30 and model.backend_options['robot_type'] == 'yam'
    load_model_adapter(model.backend_options['adapter'], {}).validate(model)
