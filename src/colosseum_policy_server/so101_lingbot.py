"""LingBot-VLA v2 SO100/101 worker (RoboColosseum/LingBot-MolmoAct2-So101) behind HTTP /act.

The checkpoint is a Hugging Face export: weights, lingbotvla_cli.yaml and
finetune_assets/ sit at the snapshot root. State and actions are six absolute
values in the earlier LeRobot SO100/101 frame (the adapter converts), mapped by
the robot config onto model indices [0..4] (arm) and [28] (gripper) of the 55-D
layout. Training sampled up to three camera slots in random order; the external
view fills camera_0, the wrist view camera_1, and camera_2 is masked out.

Upstream builds the model in float32 and then merges every shard in memory,
about 2 x 25 GB of host RAM. Here the model is built in bfloat16 and shards are
loaded one at a time; the resulting weights equal upstream's .to(bfloat16).
"""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import yaml

ROBOT = 'so101_lingbot'
CHUNK = 50
CAMERAS = ('camera_0', 'camera_1')
MODEL_INDICES = [0, 1, 2, 3, 4, 28]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def robot_config(stats):
    def entry(prefix, start, end, action):
        value = {'origin_keys': [{prefix: {'start': start, 'end': end}}]}
        if action:
            value['subtract_state'] = False
        return value
    config = {}
    for group, prefix, action in [('states', 'observation.state', False), ('actions', 'action', True)]:
        config[group] = [{f'{prefix}.arm.position': entry(prefix, 0, 5, action)},
                         {f'{prefix}.effector.position': entry(prefix, 5, 6, action)}]
    config['images'] = [{f'observation.images.{camera}': {'origin_keys': f'observation.images.{source}'}}
                        for camera, source in zip(CAMERAS, ('external', 'wrist'))]
    config['norm_stats'] = str(stats)
    return config


def validate_assets(checkpoint, source_root, processor):
    checkpoint, source_root, processor = map(Path, (checkpoint, source_root, processor))
    stats = checkpoint / 'finetune_assets/norm_stats.json'
    manifest = checkpoint / 'finetune_assets/manifest.json'
    for path in (checkpoint / 'config.json', checkpoint / 'lingbotvla_cli.yaml', checkpoint / 'model.safetensors.index.json',
                 stats, manifest, source_root / 'deploy/lingbot_vla_v2_policy.py', processor / 'config.json',
                 processor / 'tokenizer_config.json', processor / 'preprocessor_config.json'):
        if not path.is_file():
            raise ValueError(f'LingBot SO101 missing local asset: {path}')
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    missing = sorted({name for name in index['weight_map'].values() if not (checkpoint / name).is_file()})
    if missing:
        raise ValueError(f'LingBot SO101 checkpoint is missing shards: {missing}')
    training = yaml.safe_load((checkpoint / 'lingbotvla_cli.yaml').read_text())
    norm_type = training['data'].get('norm_type') or []
    if (training['train'].get('chunk_size') != CHUNK or training['data'].get('cameras', [])[:2] != list(CAMERAS)
            or "{'arm.position': 'bounds_99_woclip'}" not in norm_type
            or "{'effector.position': 'bounds_99_woclip'}" not in norm_type):
        raise ValueError('LingBot SO101 requires a 50-step camera_N checkpoint with bounds_99_woclip statistics')
    assets = json.loads(manifest.read_text())
    if (assets.get('raw_action_to_model_indices') != MODEL_INDICES
            or assets.get('action_representation') != 'absolute_joint_position'):
        raise ValueError('LingBot SO101 requires absolute 6-D joints mapped to model indices [0..4, 28]')
    norm = json.loads(stats.read_text())['norm_stats']
    for key, size in [('observation.state.arm.position', 5), ('observation.state.effector.position', 1),
                      ('action.arm.position', 5), ('action.effector.position', 1)]:
        if any(len(norm[key][field]) != size for field in ('q01', 'q99')):
            raise ValueError(f'LingBot SO101 statistics {key} must have {size} values')
    return training, stats


def normalize_instruction(text):
    # Matches training's normalize_task_language, which upstream inference lacks.
    return re.sub(r'[.?!]+$', '', text.strip().lower()).strip()


class LingBotSO101Runtime:
    action_dim = 6
    image_keys = ('external_cam', 'wrist_cam')
    camera_map = dict(zip(image_keys, ('observation.images.external', 'observation.images.wrist')))

    def __init__(self, checkpoint, *, source_root, processor, device='cuda', dtype='bfloat16',
                 use_compile=False):
        checkpoint, source_root, processor = (Path(p).resolve() for p in (checkpoint, source_root, processor))
        training, stats = validate_assets(checkpoint, source_root, processor)
        if device != 'cuda':
            raise ValueError('Upstream LingBot currently requires --device cuda; select GPU via CUDA_VISIBLE_DEVICES')
        if dtype not in ('bfloat16', 'float32'):
            raise ValueError('Unsupported LingBot dtype')
        os.environ['QWEN3VL_PATH'] = str(processor)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ.setdefault('LINGBOT_ATTN_IMPL', 'sdpa')
        sys.path.insert(0, str(source_root))
        import torch
        from safetensors import safe_open
        upstream = importlib.import_module('deploy.lingbot_vla_v2_policy')
        build_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32

        class Server(upstream.LingbotVLAv2Server):
            def load_vla(self, path_to_pi_model):
                previous = torch.get_default_dtype()
                torch.set_default_dtype(build_dtype)
                try:
                    return super().load_vla(path_to_pi_model)
                finally:
                    torch.set_default_dtype(previous)

            def load_model_weights(self, path_to_pi_model, strict=True):
                expected = set(self.vla.state_dict())
                loaded, unexpected = set(), set()
                for shard in sorted(Path(path_to_pi_model).glob('*.safetensors')):
                    with safe_open(str(shard), framework='pt', device='cpu') as f:
                        weights = {key: f.get_tensor(key).to(build_dtype) for key in f.keys()}
                    unexpected |= set(weights) - expected
                    loaded |= set(weights) & expected
                    self.vla.load_state_dict(weights, strict=False)
                    del weights
                if strict and (unexpected or loaded != expected):
                    raise RuntimeError(f'LingBot SO101 weights: {len(expected - loaded)} missing, '
                                       f'{len(unexpected)} unexpected keys')

        # Upstream reads lingbotvla_cli.yaml three levels above the weights and
        # a relative robot-config path; stage both without touching the snapshot.
        with tempfile.TemporaryDirectory(prefix='lingbot-so101-') as staging:
            staging = Path(staging)
            weights = staging / 'run/checkpoints/step/hf_ckpt'
            weights.parent.mkdir(parents=True)
            weights.symlink_to(checkpoint, target_is_directory=True)
            (staging / 'run/lingbotvla_cli.yaml').write_text(yaml.safe_dump(training))
            path = staging / f'configs/robot_configs/{ROBOT}.yaml'
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(robot_config(stats)))
            with working_directory(staging):
                self.policy = Server(
                    path_to_pi_model=str(weights), robot_norm_path=str(stats), use_length=CHUNK,
                    chunk_ret=True, use_bf16=dtype == 'bfloat16', use_fp32=dtype == 'float32',
                    use_compile=use_compile)
                self.policy.reset(robo_name=ROBOT)
        # Spend the slow first call here, not in the trial.
        blank = np.zeros((480, 640, 3), np.uint8)
        self.infer({'external_cam': blank, 'wrist_cam': blank, 'instruction': 'warm up',
                    'state': np.array([0, 90, 90, 0, 0, 10], np.float32)})

    def infer(self, payload):
        state = np.asarray(payload['state'], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError('LingBot SO101 requires 6 finite state values')
        instruction = payload['instruction']
        if not isinstance(instruction, str) or not normalize_instruction(instruction):
            raise ValueError('Instruction must be nonempty text')
        observation = {'observation.state': state.copy(), 'task': normalize_instruction(instruction)}
        for source, target in self.camera_map.items():
            frame = np.asarray(payload[source])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or not frame.size:
                raise ValueError('LingBot SO101 requires nonempty uint8 RGB images')
            observation[target] = frame.copy()
        # chunk_ret=True computes a fresh full chunk on every call.
        result = self.policy.infer(observation)
        actions = np.asarray(result['action'], dtype=np.float32)
        if actions.shape != (CHUNK, 6) or not np.isfinite(actions).all():
            raise ValueError(f'LingBot SO101 must return finite {CHUNK} x 6 absolute actions')
        return {'actions': actions}
