"""YAM MolmoAct2/GR00T/pi05/G05 HTTP /act adapter (json_numpy protocol)."""
import asyncio
import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .observation import _rgb, _state
from .bounds import check_bounds
from .franka_vla import _finite_matrix, _positive_float
from ..numpy_wire import _json_array, _json_decode


class YAMVLAAdapter:
    supported_robot_types = ('yam',)

    def __init__(self, options):
        self.timeout = _positive_float(options.get('http_timeout_seconds', 30), 'http_timeout_seconds')

    def validate(self, model):
        if (model.backend_options.get('adapter') not in {'molmoact2_yam', 'groot_yam', 'pi05_yam', 'g05_yam', 'lingbot_yam'}
                or model.action_space != 'joint_position' or model.action_dim != 14
                or model.control_hz != 30):
            raise ValueError('YAM VLA requires 14-D joint_position at 30 Hz')
        steps = model.backend_options.get('num_steps', 10)
        if type(steps) is not int or not 1 <= steps <= 10:
            raise ValueError('num_steps must be an integer in [1, 10]')
        if not model.endpoint.startswith(('http://', 'https://')):
            raise ValueError('YAM VLA requires an HTTP endpoint')

    async def infer(self, model, observation):
        self.validate(model)
        if not observation.instruction.strip():
            raise ValueError('observation instruction is empty')
        images = {image.sensor_id: image for image in observation.sensors}
        if len(images) != len(observation.sensors):
            raise ValueError('Duplicate camera sensor IDs')
        joints = _state(observation, 'joint_position', (12,))
        grippers = _state(observation, 'gripper_position', (2,))
        check_bounds(grippers, 0, 1, ('left_gripper', 'right_gripper'),
                     'YAM grippers must be in [0, 1]')
        payload = {target: _json_array(_rgb(images.get(source), source)) for target, source in
                   [('top_cam', 'head_image'), ('left_cam', 'left_image'), ('right_cam', 'right_image')]}
        payload.update(state=_json_array(np.r_[joints[:6], grippers[:1], joints[6:], grippers[1:]]),
                       instruction=observation.instruction,
                       num_steps=model.backend_options.get('num_steps', 10))
        return await asyncio.to_thread(self._request, model, payload)

    def _request(self, model, payload):
        request = Request(model.endpoint.rstrip('/') + '/act',
                          data=json.dumps(payload, separators=(',', ':')).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            raise ValueError(f'Model service returned HTTP {exc.code}') from exc
        except (URLError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('Model service returned an invalid response') from exc
        if not isinstance(raw, Mapping) or 'error' in raw or 'actions' not in raw:
            raise ValueError('Model service returned no valid actions')
        actions = _finite_matrix(_json_decode(raw['actions']), 14)
        if len(actions) > model.max_horizon:
            raise ValueError('YAM action chunk exceeds max_horizon')
        check_bounds(actions[:, [6, 13]], 0, 1, ('left_gripper', 'right_gripper'),
                     'YAM action grippers must be in [0, 1]')
        if 'raw_actions' in raw:
            try:
                return self._with_evidence(raw, actions)
            except (ValueError, TypeError, KeyError, OverflowError):
                return actions  # Missing evidence fails recording, never valid control.
        return actions

    @staticmethod
    def _with_evidence(raw, actions):
        if 'raw_actions' in raw:
            original = np.asarray(_json_decode(raw['raw_actions']))
            if (original.ndim != 2 or original.shape[1] != 14
                    or not len(actions) <= len(original) <= 4096
                    or original.dtype not in (np.dtype('float32'), np.dtype('float64'))
                    or not np.isfinite(original).all()
                    or raw.get('raw_action_source') != 'model_output_before_action_safety_clipping'):
                raise ValueError('Invalid raw YAM action chunk/provenance')
            return dict(actions=actions, raw_actions=original,
                        raw_action_source=raw['raw_action_source'])
        return actions
