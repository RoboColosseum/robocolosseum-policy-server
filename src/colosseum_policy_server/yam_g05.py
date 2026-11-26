"""G05 YAM HTTP bridge using upstream preprocessing and absolute action decoding.

Use the published training parts_meta/yam.yaml rather than guessing the
14-to-27-D padding/group layout.
"""
from copy import deepcopy
import logging
import os
from pathlib import Path
import runpy
import tempfile

import numpy as np
import yaml


log = logging.getLogger(__name__)


def validate_assets(checkpoint, source_root, processor, parts_meta=None):
    checkpoint, source_root, processor = map(lambda p: Path(p).resolve(),
                                            (checkpoint, source_root, processor))
    if parts_meta is not None:
        parts_meta = Path(parts_meta).resolve()
    else:
        relative = Path('configs/data/parts_meta/yam.yaml')
        parts_meta = checkpoint / relative
        if not parts_meta.is_file():
            parts_meta = source_root / relative
    for path in (checkpoint / 'model.pt', checkpoint / '.hydra/config.yaml',
                 checkpoint / 'dataset_stats.json', checkpoint / 'action_tokenizer.pt',
                 source_root / 'scripts/serve_policy.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not parts_meta.is_file():
        raise FileNotFoundError('Missing training parts_meta/yam.yaml; obtain the original recipe and '
                                'pass --g05-parts-meta (older checkpoint snapshots omit it): ' + str(parts_meta))
    if not processor.is_dir():
        raise ValueError('G05 YAM requires a local Qwen3.5 processor directory')
    config_text = (checkpoint / '.hydra/config.yaml').read_text()
    cfg = yaml.safe_load(config_text)
    data = cfg['data']
    if data.get('action_size') != 32 or data.get('obs_size') != 1:
        raise ValueError('G05 YAM requires 32 action steps and one observation')
    p = data['processors']['yam']
    expected_parts = [('left_arm', 6), ('left_gripper', 1), ('right_arm', 6), ('right_gripper', 1)]
    for kind in ('state', 'action'):
        if [(x['key'], x['raw_shape']) for x in p['shape_meta'][kind]] != expected_parts:
            raise ValueError('G05 YAM requires left/right six-joint arms and grippers')
    if [x['key'] for x in p['shape_meta']['images']] != ['head_rgb', 'left_wrist_rgb', 'right_wrist_rgb']:
        raise ValueError('G05 YAM requires head/left-wrist/right-wrist images')
    transforms = p['action_state_transforms']
    if len(transforms) != 1 or transforms[0].get('_target_') != (
            'g05.data_processor.transforms.relative_action.RelativeJointTransform') or (
            transforms[0].get('keys') != ['left_arm', 'right_arm']):
        raise ValueError('Expected G05 YAM relative arm transform, decoded by upstream postprocessing')
    if cfg['model']['model_arch'].get('action_dim') != 27:
        raise ValueError('Expected G05 YAM 27-D internal action layout')
    parts = yaml.safe_load(parts_meta.read_text())
    if not isinstance(parts, dict) or not all(k in parts for k in ('parts_meta', 'merge_spec')):
        raise ValueError('Training parts metadata must contain parts_meta and merge_spec')
    sizes, groups = parts['parts_meta'], parts['merge_spec']
    expected_groups = cfg['tokenizer']['vq_config']['parts_meta']
    if not isinstance(groups, dict) or list(groups) != list(expected_groups):
        raise ValueError('Training merge groups must match saved ActionCodec group order')
    for group, keys in groups.items():
        if (not isinstance(keys, list) or not keys or any(k not in sizes for k in keys)
                or any(type(sizes[k]) is not int or sizes[k] <= 0 for k in keys)
                or max(sizes[k] for k in keys) != expected_groups[group]):
            raise ValueError('Training group dimensions must match saved ActionCodec layout')
    for key, width in expected_parts:
        group = key.replace('_arm', '_control')
        if key not in groups[group] or sizes[key] < width:
            raise ValueError('Training parts layout must contain both YAM arms and grippers')
    return checkpoint, source_root, processor, parts_meta, config_text


def stage_config_text(config_text, parts_meta):
    cfg = yaml.safe_load(config_text.replace(
        'configs/data/parts_meta/yam.yaml', str(parts_meta)))
    # OmegaConf.update(..., merge=False) in upstream's sidecar loader replaces
    # interpolation aliases with partial dictionaries. Materialize only these
    # aliases before it patches ckpt_dir, retaining all saved tokenizer fields.
    model = cfg['model']
    if model.get('tokenizer') == '${tokenizer}':
        model['tokenizer'] = deepcopy(cfg['tokenizer'])
    arch = model['model_arch']
    if arch.get('AT_CONFIG') == '${model.tokenizer.vq_config}':
        arch['AT_CONFIG'] = deepcopy(model['tokenizer']['vq_config'])
    return yaml.safe_dump(cfg, sort_keys=False)


class G05YAMRuntime:
    action_dim = 14
    image_keys = ('top_cam', 'left_cam', 'right_cam')
    camera_map = dict(zip(image_keys, ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')))
    parts = (('left_arm', 0, 6), ('left_gripper', 6, 7), ('right_arm', 7, 13), ('right_gripper', 13, 14))

    def __init__(self, checkpoint, *, source_root, processor, parts_meta=None,
                 device='cuda', native_attention=False):
        checkpoint, source_root, processor, parts_meta, config_text = validate_assets(
            checkpoint, source_root, processor, parts_meta)
        # Resolve the saved config through upstream's own loader in a temporary
        # run directory. Never rewrite the downloaded training configuration.
        self._stage = tempfile.TemporaryDirectory(prefix='colosseum-g05-yam-')
        stage = Path(self._stage.name)
        (stage / '.hydra').mkdir()
        (stage / '.hydra/config.yaml').write_text(stage_config_text(config_text, parts_meta))
        for name in ('model.pt', 'dataset_stats.json', 'action_tokenizer.pt'):
            (stage / name).symlink_to(checkpoint / name)
        (stage / 'hf_processor').symlink_to(processor, target_is_directory=True)
        previous_cwd = Path.cwd()
        try:
            os.chdir(source_root)
            if not native_attention:
                from g05.models.g05.qwen35 import vision
                vision._flash_attn_varlen = None
                vision._flash_attn_backend = 'sdpa'
            upstream = runpy.run_path(str(source_root / 'scripts/serve_policy.py'))
            cfg = upstream['load_config_from_run_dir'](stage, str(stage / 'model.pt'),
                ['eval_embodiment=yam', 'model.use_torch_compile=false'])
            # Checkpoint's hf_processor_path is a relative path even when
            # pretrained_model_path is null, so set both consumers explicitly.
            cfg.model.model_arch.hf_processor_path = str(processor)
            cfg.model.processor.tokenizer_params.pretrained_model_name_or_path = str(processor)
            upstream['filter_embodiment'](cfg, 'yam')
            policy, self.processor = upstream['setup'](cfg, device=device)
            # Pinned G05 generate_action returns FM in `action` whenever
            # continuous_action is enabled, but inferencer still attaches AR
            # absence metadata. That metadata does not describe the FM tensor.
            self.continuous_output = getattr(policy, 'continuous_action', False) is True
            self._reported_ar_absence = False
            log.info('G05 YAM output branch: %s', 'continuous/FM' if self.continuous_output else 'discrete/AR')
            self.inferencer = upstream['PolicyInferencer'](policy, self.processor, device=device)
            self.build_obs = upstream['build_obs_dict']
        finally:
            os.chdir(previous_cwd)

    def infer(self, payload):
        state = np.asarray(payload['state'], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('G05 YAM requires 14 finite state values')
        if np.any(state[[6, 13]] < 0) or np.any(state[[6, 13]] > 1):
            raise ValueError('YAM gripper state must be normalized to [0, 1]')
        instruction = payload['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Instruction must be nonempty text')
        images = {}
        for source, target in self.camera_map.items():
            frame = np.asarray(payload[source])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or not frame.size:
                raise ValueError('G05 YAM requires nonempty uint8 RGB images')
            images[target] = np.ascontiguousarray(frame.transpose(2, 0, 1))
        raw = {'images': images, 'state': {key: state[start:end].copy() for key, start, end in self.parts},
               'task': instruction, 'frequency': 30, 'embodiment_type': 'yam'}
        # PolicyInferencer includes merger reversal, unnormalization and
        # RelativeJointTransform.backward. These are already absolute actions.
        results = self.inferencer.infer([self.build_obs(raw, self.processor)])
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError('G05 must return one postprocessed action dictionary')
        result = results[0]
        absent = set(result.get('_absent_keys') or ())
        if absent and self.continuous_output:
            if not self._reported_ar_absence:
                log.warning('G05 AR branch absent groups: %s; returned actions use continuous/FM. '
                            'Validating all four YAM output parts independently.', sorted(absent))
                self._reported_ar_absence = True
        elif absent - {'lower_body'}:
            # lower_body is not part of the YAM embodiment. Required arm/gripper
            # groups remain fatal for discrete output, including unknown groups.
            raise ValueError(f'G05 discrete/AR output has absent action groups: '
                             f'{sorted(absent)}; refusing partial robot commands')
        arrays = []
        for key, start, end in self.parts:
            value = result.get(key)
            if hasattr(value, 'detach'):
                value = value.detach().cpu().float().numpy()
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (1, 32, end - start) or not np.isfinite(array).all():
                raise ValueError(f'G05 {key} must be finite with shape (1, 32, {end-start})')
            arrays.append(array[0])
        actions = np.concatenate(arrays, axis=1)
        raw_actions = actions.copy()
        # Only saturate finite, postprocessed gripper targets. Keep all arm
        # joints unchanged and reject nonfinite/missing values above.
        for column, label in ((6, 'left_gripper'), (13, 'right_gripper')):
            values = actions[:, column]
            outside = (values < 0) | (values > 1)
            if np.any(outside):
                first = int(np.flatnonzero(outside)[0])
                log.warning('G05 clipped %s to [0, 1]: count=%d raw_min=%.6f '
                            'raw_max=%.6f first_action_index=%d target=%.6f clipped=%.6f',
                            label, int(outside.sum()), float(values.min()), float(values.max()),
                            first, float(values[first]), float(np.clip(values[first], 0, 1)))
                np.clip(values, 0, 1, out=values)
        return {'actions': actions, 'raw_actions': raw_actions,
                'raw_action_source': 'model_output_before_action_safety_clipping'}
