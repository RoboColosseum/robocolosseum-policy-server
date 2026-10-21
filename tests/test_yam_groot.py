import contextlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_model_workers import module, Tensor
from colosseum_policy_server.yam_groot import GrootYAMRuntime
from colosseum_policy_server.model_service import parse_args


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = dict(type='groot', embodiment_tag='new_embodiment', use_relative_actions=False,
        n_obs_steps=1, chunk_size=16, n_action_steps=16,
        input_features={**{k: {'shape': [3,360,640]} for k in GrootYAMRuntime.camera_map.values()},
                        'observation.state': {'shape': [14]}}, output_features={'action': {'shape': [14]}})
    (tmp_path/'config.json').write_text(json.dumps(config))
    module(monkeypatch, 'torch', from_numpy=lambda x: x, inference_mode=contextlib.nullcontext)
    cfg = SimpleNamespace()
    module(monkeypatch, 'lerobot.policies.groot.configuration_groot',
           GrootConfig=SimpleNamespace(from_pretrained=lambda *a, **k: cfg))
    calls = {}
    class Policy:
        def to(self, device): return self
        def eval(self): return self
        def predict_action_chunk(self, batch):
            assert batch == 'processed'
            return 'normalized_actions'
    policy = Policy()
    def load(path, **kwargs):
        assert kwargs['local_files_only'] is True
        assert kwargs['config'].base_model_path == str(tmp_path)
        return policy
    module(monkeypatch, 'lerobot.policies.groot.modeling_groot',
           GrootPolicy=SimpleNamespace(from_pretrained=load))
    def processors(config, **kwargs):
        calls['processors'] = kwargs
        def pre(batch):
            calls['batch'] = batch
            return 'processed'
        def post(value):
            assert value == 'normalized_actions'
            return Tensor(np.tile(calls['batch']['observation.state'], (1,16,1)))
        return pre, post
    module(monkeypatch, 'lerobot.policies.factory', make_pre_post_processors=processors)
    return GrootYAMRuntime(tmp_path, base_model=tmp_path, processor=tmp_path), calls, tmp_path


def payload():
    return dict(state=np.arange(14,dtype=np.float32)/20, instruction='pick', **{
        k: np.full((8,12,3), i*50, np.uint8) for i,k in enumerate(GrootYAMRuntime.image_keys)})


def test_saved_processors_and_absolute_chunk(runtime):
    worker, calls, path = runtime
    result = worker.infer(payload())
    np.testing.assert_array_equal(result['actions'], np.tile(payload()['state'], (16,1)))
    for i, key in enumerate(worker.camera_map.values()):
        assert calls['batch'][key].shape == (3,8,12)
        np.testing.assert_allclose(calls['batch'][key], i*50/255)
    assert calls['batch']['task'] == 'pick'
    assert calls['processors']['preprocessor_overrides']['groot_n1_7_vlm_encode_v1']['model_name'] == str(path)
    assert 'groot_n1_7_pack_inputs_v1' not in calls['processors']['preprocessor_overrides']


def test_explicit_top_left_right_override_preserves_actions(runtime):
    _, calls, path = runtime
    original_config = (path / 'config.json').read_bytes()
    worker = GrootYAMRuntime(path, base_model=path, processor=path, camera_order='top-left-right')
    assert calls['processors']['preprocessor_overrides']['groot_n1_7_pack_inputs_v1'] == {
        'video_modality_keys': ['top', 'left', 'right']}
    result = worker.infer(payload())
    np.testing.assert_array_equal(result['actions'], np.tile(payload()['state'], (16, 1)))
    assert (path / 'config.json').read_bytes() == original_config
    with pytest.raises(ValueError, match='camera_order'):
        GrootYAMRuntime(path, base_model=path, processor=path, camera_order='wrong')


@pytest.mark.parametrize('bad', ['width','nan','gripper'])
def test_reject_invalid_model_output(runtime, bad):
    worker, _, _ = runtime
    values=np.zeros((1,16,14))
    if bad=='width': values=np.zeros((1,16,8))
    elif bad=='nan': values[0,0,0]=np.nan
    else: values[0,0,13]=2
    worker.post=lambda _:Tensor(values)
    if bad=='gripper':  # saturated, not rejected
        assert worker.infer(payload())['actions'][0,13]==1
        return
    with pytest.raises(ValueError):worker.infer(payload())


@pytest.mark.parametrize('field,value', [('use_relative_actions',True),('chunk_size',30),('embodiment_tag','droid')])
def test_reject_wrong_checkpoint(runtime, field,value):
    _,_,path=runtime
    raw=json.loads((path/'config.json').read_text());raw[field]=value
    (path/'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError):GrootYAMRuntime(path,base_model=path,processor=path)


def test_yam_groot_cli_requires_local_assets(tmp_path):
    args=['groot_n17','--robot-type','yam','--checkpoint',str(tmp_path),'--processor',str(tmp_path),'--port','8203']
    with pytest.raises(SystemExit):parse_args(args)
    assert parse_args(args+['--base-model',str(tmp_path)]).robot_type=='yam'


@pytest.mark.parametrize('camera_order', ['checkpoint', 'top-left-right'])
def test_cli_dispatches_yam_to_http_not_native_groot(runtime, monkeypatch, camera_order):
    from colosseum_policy_server import model_service as service, yam_groot
    worker, _, path = runtime
    calls = []
    def create(*args, **kwargs):
        assert kwargs['camera_order'] == camera_order
        return worker
    monkeypatch.setattr(yam_groot, 'GrootYAMRuntime', create)
    class Server:
        def __init__(self, address, handler): assert address == ('127.0.0.1', 8203)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def serve_forever(self): calls.append('http')
    monkeypatch.setattr(service, 'HTTPServer', Server)
    service.main(['groot_n17','--robot-type','yam','--checkpoint',str(path),
                  '--processor',str(path),'--base-model',str(path),'--port','8203',
                  '--groot-camera-order',camera_order])
    assert calls == ['http']


def test_camera_override_rejected_for_other_workers(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(['molmoact2', '--robot-type', 'yam', '--checkpoint', str(tmp_path),
                    '--port', '8202', '--groot-camera-order', 'top-left-right'])
