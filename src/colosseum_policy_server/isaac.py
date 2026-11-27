"""Perceptron Isaac 0.5 worker for SO100/101 and bimanual YAM (LeRobot ``perceptron_isaac``).

Serves the ``recorded_policy`` directory of PerceptronAI/Isaac-0.5-SO100_101 or
PerceptronAI/Isaac-0.5-YAM through the checkpoint's saved pre/post processors. They
own quantile state/action normalization, image resizing (``stretch`` to the
checkpoint's ``image_size``), mHarmony rendering and action unnormalization. Actions
are absolute joint/gripper targets in the request's state layout, 60 per chunk.

SO101: state arrives in LeRobot's earlier SO100 frame, because the Policy Server
adapter converts it (``joint_frame: legacy``). That is the checkpoint's training
frame, so the saved processors keep their identity frame transform. Do not also
apply the model card's live-arm ``so_frame`` override: that would convert twice.
The wrist image fills ``observation.images.top`` and the external image
``observation.images.side``, as in the model card's sample.

YAM: top, left wrist and right wrist images. State/actions are left six joints,
left gripper, right six joints, right gripper; grippers in [0, 1].

Needs Perceptron's LeRobot fork (``isaac/lerobot`` with the ``perceptron_isaac``
extra, Python 3.12, CUDA 12.8) in the worker environment. Loads local files only.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
CHUNK = 60


@dataclass(frozen=True)
class IsaacContract:
    cameras: tuple[tuple[str, str], ...]  # (request field, checkpoint image feature)
    names: tuple[str, ...]  # state and action feature names, in wire order
    grippers: tuple[int, ...]
    gripper_high: float


CONTRACTS = {
    'so101': IsaacContract(
        cameras=(('external_cam', 'observation.images.side'), ('wrist_cam', 'observation.images.top')),
        names=('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
               'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'),
        grippers=(5,), gripper_high=100.0),
    'yam': IsaacContract(
        cameras=(('top_cam', 'observation.images.top'), ('left_cam', 'observation.images.left'),
                 ('right_cam', 'observation.images.right')),
        names=tuple(f'{side}_{joint}.pos' for side in ('left', 'right')
                    for joint in (*(f'joint_{i}' for i in range(6)), 'gripper')),
        grippers=(6, 13), gripper_high=1.0),
}


def validate_checkpoint(checkpoint, robot_type):
    """Check a local ``recorded_policy`` directory against the robot contract."""
    if robot_type not in CONTRACTS:
        raise ValueError('Isaac supports robot_type so101 or yam')
    checkpoint, contract = Path(checkpoint), CONTRACTS[robot_type]
    for name in ('config.json', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f'{checkpoint / name} is missing; pass the recorded_policy directory')
    if not (checkpoint / 'hf_model').is_dir():
        raise FileNotFoundError(f'{checkpoint / "hf_model"} is missing; download the full snapshot')
    raw = json.loads((checkpoint / 'config.json').read_text())
    inputs, outputs = raw.get('input_features') or {}, raw.get('output_features') or {}
    dim, cameras = len(contract.names), {feature for _, feature in contract.cameras}
    if (raw.get('type') != 'perceptron_isaac' or raw.get('n_obs_steps') != 1
            or raw.get('chunk_size') != CHUNK or raw.get('action_dim') != dim
            or raw.get('state_feature_names') != list(contract.names)
            or raw.get('action_feature_names') != list(contract.names)
            or set(inputs) != {*cameras, 'observation.state'}
            or inputs['observation.state'].get('shape') != [dim]
            or (outputs.get('action') or {}).get('shape') != [dim]):
        raise ValueError(f'Expected a Perceptron Isaac {robot_type} recorded_policy: {dim}-D state/action, '
                         f'cameras {sorted(cameras)}, one observation and {CHUNK}-step chunks')
    if raw.get('joint_signs') is not None or raw.get('joint_offsets') is not None:
        raise ValueError('Isaac checkpoint declares a joint frame transform; expected its training frame')
    if raw.get('target_fps') != 30:
        raise ValueError('Isaac checkpoint must be trained at target_fps 30')
    return raw


class IsaacRuntime:
    def __init__(self, checkpoint, *, robot_type, device='cuda', num_steps=10, return_steps=CHUNK, warmup=True):
        if type(num_steps) is not int or not 1 <= num_steps <= 10:
            raise ValueError('num_steps must be an integer in [1, 10]')
        if type(return_steps) is not int or not 1 <= return_steps <= CHUNK:
            raise ValueError(f'return_steps must be an integer in [1, {CHUNK}]')
        checkpoint = Path(checkpoint)
        validate_checkpoint(checkpoint, robot_type)
        contract = CONTRACTS[robot_type]
        self.robot_type, self.return_steps = robot_type, return_steps
        self.camera_map = dict(contract.cameras)
        self.image_keys = tuple(self.camera_map)
        self.action_dim = len(contract.names)
        self.grippers, self.gripper_high = list(contract.grippers), contract.gripper_high

        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy

        self.torch = torch
        config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
        config.device = device
        config.num_inference_steps = num_steps  # flow steps; the chunk stays 60
        self.policy = PerceptronIsaacPolicy.from_pretrained(str(checkpoint), config=config, local_files_only=True)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        if warmup:
            # The published fp32-storage packages load (and cast) the backbone on first use.
            # Do that before the worker listens, so the first trial does not wait minutes.
            start = time.monotonic()
            state = np.zeros(self.action_dim, np.float32)
            state[self.grippers] = self.gripper_high
            self.infer(dict(state=state, instruction='Hold the current position.',
                            **{key: np.zeros((480, 640, 3), np.uint8) for key in self.image_keys}))
            log.info('Isaac %s loaded and warmed up in %.1fs', robot_type, time.monotonic() - start)

    def infer(self, payload):
        torch = self.torch
        state = np.asarray(payload['state'], dtype=np.float32).reshape(-1)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError(f'Isaac {self.robot_type} requires {self.action_dim} finite state values')
        instruction = payload['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Instruction must be nonempty text')
        batch = {'observation.state': torch.from_numpy(state[None].copy()), 'task': [instruction]}
        for source, target in self.camera_map.items():
            frame = np.asarray(payload[source])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
                raise ValueError('Isaac requires nonempty uint8 HWC RGB images')
            # Model card input: float RGB in [0, 1], shape (1, 3, H, W).
            batch[target] = torch.from_numpy(
                np.ascontiguousarray(frame.transpose(2, 0, 1)[None], dtype=np.float32) / 255)
        # One observation per request: clear the policy's frame clock, flow-seed counter and queues.
        self.policy.reset()
        with torch.inference_mode():
            actions = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        actions = np.array(actions.detach().float().cpu().numpy(), dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.shape != (CHUNK, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f'Isaac must return a finite {CHUNK} x {self.action_dim} chunk, got {actions.shape}')
        actions = actions[:self.return_steps]
        # Saturate only gripper targets (as the pi05/G05/LingBot workers do); arm joints are unchanged.
        grippers = actions[:, self.grippers]
        outside = (grippers < 0) | (grippers > self.gripper_high)
        if outside.any():
            log.warning('Isaac clipped grippers to [0, %g]: count=%d raw_min=%.6f raw_max=%.6f', self.gripper_high,
                        int(outside.sum()), float(grippers.min()), float(grippers.max()))
            actions[:, self.grippers] = np.clip(grippers, 0, self.gripper_high)
        return {'actions': actions}
