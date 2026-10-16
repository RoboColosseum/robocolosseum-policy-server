"""LingBot-VLA v2 YAM worker behind the common local HTTP /act protocol."""
from contextlib import contextmanager
import importlib
import logging
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml

from .model_adapters.bounds import check_bounds

ROBOT = 'molmoact2_yam_abs'
log = logging.getLogger(__name__)


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def validate_assets(checkpoint, source_root, processor):
    checkpoint, source_root, processor = map(Path, (checkpoint, source_root, processor))
    root = checkpoint.parent.parent.parent
    robot_path = root / f'configs/robot_configs/{ROBOT}.yaml'
    stats = root / f'assets/norm_stats/{ROBOT}.json'
    for path in (checkpoint / 'config.json', root / 'lingbotvla_cli.yaml', robot_path, stats,
                 source_root / 'deploy/lingbot_vla_v2_policy.py', processor / 'config.json',
                 processor / 'tokenizer_config.json', processor / 'preprocessor_config.json'):
        if not path.is_file():
            raise ValueError(f'LingBot YAM missing local asset: {path}')
    if not list(checkpoint.glob('*.safetensors')):
        raise ValueError('LingBot checkpoint contains no safetensors weights')
    training = yaml.safe_load((root / 'lingbotvla_cli.yaml').read_text())
    robot = yaml.safe_load(robot_path.read_text())
    arm = [{'start': 0, 'end': 6}, {'start': 7, 'end': 13}]
    gripper = [{'start': 6, 'end': 7}, {'start': 13, 'end': 14}]
    expected = {}
    for group, prefix in [('states', 'observation.state'), ('actions', 'action')]:
        expected[group] = []
        for part, slices in [('arm', arm), ('effector', gripper)]:
            entry = {'origin_keys': [{prefix: s} for s in slices]}
            if group == 'actions':
                entry['subtract_state'] = False
            expected[group].append({f'{prefix}.{part}.position': entry})
    expected['images'] = [{f'observation.images.{camera}': {'origin_keys': f'observation.images.{key}'}}
                          for camera, key in [('camera_top', 'top'), ('camera_wrist_left', 'left'),
                                              ('camera_wrist_right', 'right')]]
    if any(robot.get(key) != value for key, value in expected.items()):
        raise ValueError('LingBot requires absolute YAM 14-D joint/gripper order and top/left/right cameras')
    if (training['train'].get('chunk_size') != 30
            or training['data'].get('cameras') != ['camera_top', 'camera_wrist_left', 'camera_wrist_right']):
        raise ValueError('LingBot YAM requires a 30-step three-camera checkpoint')
    return robot, stats


class LingBotYAMRuntime:
    action_dim = 14
    image_keys = ('top_cam', 'left_cam', 'right_cam')
    camera_map = dict(zip(image_keys, ('observation.images.top', 'observation.images.left',
                                     'observation.images.right')))

    def __init__(self, checkpoint, *, source_root, processor, device='cuda', dtype='bfloat16',
                 use_compile=False, return_steps=30):
        if not 1 <= return_steps <= 30:
            raise ValueError('LingBot return_steps must be in [1, 30]')
        self.return_steps = return_steps
        checkpoint, source_root, processor = (Path(p).resolve() for p in (checkpoint, source_root, processor))
        robot, stats = validate_assets(checkpoint, source_root, processor)
        if device != 'cuda':
            raise ValueError('Upstream LingBot currently requires --device cuda; select GPU via CUDA_VISIBLE_DEVICES')
        if dtype not in ('bfloat16', 'float32'):
            raise ValueError('Unsupported LingBot dtype')
        os.environ['QWEN3VL_PATH'] = str(processor)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, str(source_root))
        upstream = importlib.import_module('deploy.lingbot_vla_v2_policy')
        # Upstream reset reads a relative robot-config path. Stage only that
        # config; do not overwrite an upstream checkout or mutate the snapshot.
        with tempfile.TemporaryDirectory(prefix='lingbot-yam-') as staging:
            path = Path(staging) / f'configs/robot_configs/{ROBOT}.yaml'
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(robot))
            with working_directory(staging):
                self.policy = upstream.LingbotVLAv2Server(
                    path_to_pi_model=str(checkpoint), robot_norm_path=str(stats), use_length=30,
                    chunk_ret=True, use_bf16=dtype == 'bfloat16', use_fp32=dtype == 'float32',
                    use_compile=use_compile)
                self.policy.reset(robo_name=ROBOT)

    def infer(self, payload):
        state = np.asarray(payload['state'], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('LingBot YAM requires 14 finite state values')
        check_bounds(state[[6, 13]], 0, 1, ('left_gripper', 'right_gripper'),
                     'LingBot YAM gripper state outside [0, 1]')
        instruction = payload['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Instruction must be nonempty text')
        observation = {'observation.state': state.copy(), 'task': instruction}
        for source, target in self.camera_map.items():
            frame = np.asarray(payload[source])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or not frame.size:
                raise ValueError('LingBot YAM requires nonempty uint8 RGB images')
            observation[target] = frame.copy()
        # chunk_ret=True makes upstream infer compute a fresh full chunk on
        # every call rather than consuming a stale single-step action queue.
        result = self.policy.infer(observation)
        actions = np.asarray(result['action'], dtype=np.float32)
        if actions.shape != (30, 14) or not np.isfinite(actions).all():
            raise ValueError('LingBot YAM must return finite 30 x 14 absolute actions')
        grippers = actions[:, [6, 13]]
        outside = (grippers < 0) | (grippers > 1)
        if np.any(outside):
            actions = actions.copy()  # Do not mutate upstream's cached action chunk.
            for row, side in zip(*np.nonzero(outside)):
                original = float(grippers[row, side])
                clipped = float(np.clip(original, 0, 1))
                log.warning('LingBot YAM gripper clipped: action_index=%d %s_gripper '
                            'target=%.9g clipped=%.9g',
                            row, ('left', 'right')[side], original, clipped)
            actions[:, [6, 13]] = np.clip(grippers, 0, 1)
        check_bounds(actions[:, [6, 13]], 0, 1, ('left_gripper', 'right_gripper'),
                     'LingBot YAM action grippers outside [0, 1]')
        # Predict the full trained 30-step chunk; optionally execute only a prefix.
        return {'actions': actions[:self.return_steps]}
