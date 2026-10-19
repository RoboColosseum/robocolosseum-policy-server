"""Local LeRobot pi05-MolmoAct2-YAM worker, using saved quantile processors."""
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Pi05YAMRuntime:
    action_dim = 14
    image_keys = ('top_cam', 'left_cam', 'right_cam')
    camera_map = dict(zip(image_keys, ('observation.images.top', 'observation.images.left',
                                     'observation.images.right')))
    action_names = [name for side in ('left', 'right') for name in
                    [*(f'{side}_joint_{i}.pos' for i in range(6)), f'{side}_gripper.pos']]

    def __init__(self, checkpoint, tokenizer, *, device='cuda', num_steps=10):
        checkpoint = Path(checkpoint)
        for name in ('config.json', 'model.safetensors', 'policy_preprocessor.json',
                     'policy_postprocessor.json',
                     'policy_preprocessor_step_3_normalizer_processor.safetensors',
                     'policy_postprocessor_step_0_unnormalizer_processor.safetensors'):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        if not Path(tokenizer).is_dir():
            raise ValueError('pi05 YAM requires a local PaliGemma tokenizer directory')
        raw = json.loads((checkpoint / 'config.json').read_text())
        if (raw.get('type') != 'pi05' or raw.get('use_relative_actions') is not False
                or raw.get('n_obs_steps') != 1 or raw.get('chunk_size') != 30
                or raw.get('n_action_steps') != 30 or raw.get('action_feature_names') != self.action_names):
            raise ValueError('Expected absolute 30-step pi05 YAM checkpoint with YAM joint order')
        inputs, outputs = raw['input_features'], raw['output_features']
        cameras = [key for key in inputs if key.startswith('observation.images.')]
        if (cameras != list(self.camera_map.values())
                or set(inputs) != {*self.camera_map.values(), 'observation.state'}
                or inputs['observation.state']['shape'] != [14]
                or outputs['action']['shape'] != [14]):
            raise ValueError('pi05 YAM requires top/left/right cameras and 14-D state/action')
        if type(num_steps) is not int or not 1 <= num_steps <= 10:
            raise ValueError('num_steps must be an integer in [1, 10]')

        import torch
        from safetensors.torch import load_file
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        config = PI05Config.from_pretrained(str(checkpoint), local_files_only=True)
        config.device = device
        config.num_inference_steps = num_steps
        config.compile_model = False
        config.gradient_checkpointing = False
        # The pinned upstream from_pretrained catches weight errors and returns
        # a potentially uninitialized policy. Apply its key conversion explicitly
        # and let strict loading errors propagate instead.
        self.policy = PI05Policy(config)
        weights = load_file(str(checkpoint / 'model.safetensors'))
        weights = self.policy._fix_pytorch_state_dict_keys(weights, config)
        weights = {key if key.startswith('model.') else f'model.{key}': value
                   for key, value in weights.items()}
        weights = self.policy._prepare_pretrained_state_dict(weights)
        self.policy.load_state_dict(weights, strict=True)
        del weights
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
            preprocessor_overrides={
                'tokenizer_processor': {'tokenizer_name': str(Path(tokenizer).resolve())},
                'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})

    def infer(self, payload):
        state = np.asarray(payload['state'], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('pi05 YAM requires 14 finite state values')
        if np.any(state[[6, 13]] < 0) or np.any(state[[6, 13]] > 1):
            raise ValueError('YAM gripper state must be normalized to [0, 1]')
        instruction = payload['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Instruction must be nonempty text')
        batch = {'observation.state': self.torch.from_numpy(state), 'task': instruction}
        for source, target in self.camera_map.items():
            frame = np.asarray(payload[source])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or not frame.size:
                raise ValueError('pi05 YAM requires nonempty uint8 RGB images')
            batch[target] = self.torch.from_numpy(
                np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32) / 255)
        # One observation per request, no action queue or cross-trial history.
        self.policy.reset()
        with self.torch.inference_mode():
            actions = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        actions = actions.detach().cpu().float().numpy()
        if actions.shape != (1, 30, 14) or not np.isfinite(actions).all():
            raise ValueError('pi05 YAM must return a finite 1 x 30 x 14 action tensor')
        # Only saturate finite gripper targets (same as G05/LingBot); arm joints are unchanged.
        grippers = actions[:, :, [6, 13]]
        if np.any(grippers < 0) or np.any(grippers > 1):
            log.warning('pi05 clipped grippers to [0, 1]: count=%d raw_min=%.6f raw_max=%.6f',
                        int(((grippers < 0) | (grippers > 1)).sum()), float(grippers.min()), float(grippers.max()))
            actions[:, :, [6, 13]] = np.clip(grippers, 0, 1)
        return {'actions': actions[0]}
