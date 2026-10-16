"""LeRobot GR00T N1.7 YAM worker; absolute joints and normalized grippers.

Loads local assets only. Use the checkpoint's saved pre/post processors so camera
packing, normalization and action unnormalization match training.
"""
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class GrootYAMRuntime:
    action_dim = 14
    image_keys = ('top_cam', 'left_cam', 'right_cam')
    camera_map = dict(zip(image_keys, ('observation.images.top', 'observation.images.left',
                                     'observation.images.right')))

    def __init__(self, checkpoint, *, base_model, processor, device='cuda', camera_order='checkpoint'):
        if camera_order not in {'checkpoint', 'top-left-right'}:
            raise ValueError('camera_order must be checkpoint or top-left-right')
        import torch
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        checkpoint = Path(checkpoint)
        raw = json.loads((checkpoint / 'config.json').read_text())
        if (raw.get('type') != 'groot' or raw.get('embodiment_tag') != 'new_embodiment'
                or raw.get('use_relative_actions', False) or raw.get('n_obs_steps') != 1
                or raw.get('chunk_size') != 16 or raw.get('n_action_steps') != 16):
            raise ValueError('Expected absolute 16-step LeRobot GR00T YAM checkpoint')
        inputs, outputs = raw['input_features'], raw['output_features']
        if set(inputs) != {*self.camera_map.values(), 'observation.state'}:
            raise ValueError('GR00T YAM checkpoint must contain top/left/right cameras and state')
        if inputs['observation.state']['shape'] != [14] or outputs['action']['shape'] != [14]:
            raise ValueError('GR00T YAM requires 14-D state and action')
        config = GrootConfig.from_pretrained(str(checkpoint), local_files_only=True)
        config.device = device
        config.base_model_path = str(Path(base_model).resolve())
        self.policy = GrootPolicy.from_pretrained(str(checkpoint), config=config,
                                                 local_files_only=True).to(device).eval()
        overrides = {
            'groot_n1_7_vlm_encode_v1': {'model_name': str(Path(processor).resolve()), 'device': device},
            'device_processor': {'device': device}}
        if camera_order == 'top-left-right':
            overrides['groot_n1_7_pack_inputs_v1'] = {'video_modality_keys': ['top', 'left', 'right']}
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
            preprocessor_overrides=overrides,
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        logging.getLogger(__name__).info('GR00T YAM camera order mode: %s', camera_order)

    def infer(self, payload):
        state = np.asarray(payload['state'], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('GR00T YAM requires 14 finite state values')
        if np.any(state[[6, 13]] < 0) or np.any(state[[6, 13]] > 1):
            raise ValueError('YAM gripper state must be normalized to [0, 1]')
        instruction = payload['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Instruction must be nonempty text')
        batch = {'observation.state': self.torch.from_numpy(state), 'task': instruction}
        for source, target in self.camera_map.items():
            frame = np.asarray(payload[source])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or not frame.size:
                raise ValueError('GR00T YAM requires nonempty uint8 RGB images')
            batch[target] = self.torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32) / 255)
        # No select_action queue: each /act call predicts and postprocesses a full chunk.
        with self.torch.inference_mode():
            actions = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        actions = actions.detach().cpu().numpy()
        if actions.shape != (1, 16, 14) or not np.isfinite(actions).all():
            raise ValueError('GR00T YAM must return a finite 1 x 16 x 14 action tensor')
        # Only saturate finite gripper targets (same as G05/LingBot); arm joints are unchanged.
        grippers = actions[:, :, [6, 13]]
        if np.any(grippers < 0) or np.any(grippers > 1):
            log.warning('GR00T clipped grippers to [0, 1]: count=%d raw_min=%.6f raw_max=%.6f',
                        int(((grippers < 0) | (grippers > 1)).sum()), float(grippers.min()), float(grippers.max()))
            actions[:, :, [6, 13]] = np.clip(grippers, 0, 1)
        return {'actions': actions[0].astype(np.float32)}
