"""Inference-side robot contracts. Hardware drivers live in the Client."""
from dataclasses import dataclass

import numpy as np


def canonical_robot(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('robot_type must be a nonempty string')
    return 'franka' if value == 'droid' else value


def session_robot(model, request, options):
    configured = model.backend_options.get('robot_type', options.get('robot_type'))
    # Older DROID clients did not send robot_type. Preserve that legacy path.
    robot = canonical_robot(request.get('robot_type') or configured or 'franka')
    if configured is not None and canonical_robot(configured) != robot:
        raise ValueError('Client robot_type does not match the configured model')
    return robot


@dataclass(frozen=True)
class JointRobotContract:
    name: str
    joint_count: int
    state_key: str = 'joint_position'
    gripper_indices: tuple[int, ...] = (7,)
    binary_gripper: bool = True
    gripper_high: float = 1.0
    gripper_max_step: float = 1.0
    state_unit: str = 'rad+normalized'

    @property
    def action_dim(self):
        return self.joint_count + len(self.gripper_indices)

    @property
    def joint_indices(self):
        return tuple(i for i in range(self.action_dim) if i not in self.gripper_indices)

    @property
    def labels(self):
        if self.name == 'yam':
            return tuple(label for side in ('left', 'right')
                         for label in (*[f'{side}_joint{i+1}' for i in range(6)], f'{side}_gripper'))
        if self.name == 'so101':
            return ('shoulder_pan', 'shoulder_lift', 'elbow_flex',
                    'wrist_flex', 'wrist_roll', 'gripper')
        return tuple(f'joint{i+1}' for i in range(self.joint_count)) + ('gripper',)

    def pack(self, joints, grippers):
        result = np.empty(self.action_dim, dtype=np.float64)
        result[list(self.joint_indices)] = joints
        result[list(self.gripper_indices)] = grippers
        return result

    def validate(self, model):
        if model.action_space != 'joint_position' or model.action_dim != self.action_dim:
            raise ValueError(f'{self.name} LLM requires {self.action_dim}-D joint_position')
        if self.name == 'yam' and model.control_hz != 30:
            raise ValueError('YAM requires control_hz: 30')


def llm_robot_contract(robot):
    # Add a verified contract here when its observation/action conventions are
    # implemented. Never infer embodiment from action dimension alone.
    if robot == 'yam':
        return JointRobotContract('yam', 12, gripper_indices=(6, 13), binary_gripper=False)
    if robot == 'so101':
        return JointRobotContract('so101', 5, gripper_indices=(5,),
                                  binary_gripper=False, gripper_high=100.0,
                                  gripper_max_step=5.0, state_unit='degrees+gripper_0_100')
    if robot != 'franka':
        raise ValueError(f'LLM robot adapter {robot!r} is not implemented')
    return JointRobotContract('franka', 7)
