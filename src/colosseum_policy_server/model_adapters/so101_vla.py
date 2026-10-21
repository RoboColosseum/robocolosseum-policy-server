"""SO101 VLA adapters: HTTP /act workers and the upstream G05 chunk server.

State and actions are six values in LeRobot's so101_follower order
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper):
joint angles in degrees and gripper opening in [0, 100]. No hardware imports.

The Client uses LeRobot's current calibration frame. The registered generalist
SO100/101 checkpoints were trained on datasets in LeRobot's earlier frame, so
state is converted before inference and actions are converted back; see
https://huggingface.co/docs/lerobot/backwardcomp. A checkpoint fine-tuned on
current-frame data needs backend_options.joint_frame: current.
"""
import asyncio
import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .observation import _rgb, _state
from .franka_vla import _finite_matrix, _nonempty, _positive_float
from ..numpy_wire import _json_array, _json_decode, _msgpack_default, _msgpack_object

HTTP_ADAPTERS = frozenset({'molmoact2_so101', 'pi05_so101', 'groot_so101', 'lingbot_so101'})
ADAPTERS = HTTP_ADAPTERS | {'g05_so101'}
# Earlier LeRobot SO100/101 frame: model = sign * current + offset (degrees).
LEGACY_SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float32)
LEGACY_OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=np.float32)
JOINT_FRAMES = {'legacy': (LEGACY_SIGNS, LEGACY_OFFSETS),
                'current': (np.ones(6, np.float32), np.zeros(6, np.float32))}


class SO101VLAAdapter:
    supported_robot_types = ('so101',)

    def __init__(self, options):
        options = dict(options)
        self.external_sensor = _nonempty(options.get('external_sensor', 'head_image'), 'external_sensor')
        self.wrist_sensor = _nonempty(options.get('wrist_sensor', 'left_image'), 'wrist_sensor')
        if self.external_sensor == self.wrist_sensor:
            raise ValueError('external_sensor and wrist_sensor must differ')
        self.timeout = _positive_float(options.get('http_timeout_seconds', 30), 'http_timeout_seconds')

    def validate(self, model):
        adapter = model.backend_options.get('adapter')
        if adapter not in ADAPTERS or model.action_space != 'joint_position' or model.action_dim != 6:
            raise ValueError('SO101 VLA requires 6-D joint_position')
        steps = model.backend_options.get('num_steps', 10)
        if type(steps) is not int or not 1 <= steps <= 10:
            raise ValueError('num_steps must be an integer in [1, 10]')
        scheme = ('http://', 'https://') if adapter in HTTP_ADAPTERS else ('ws://',)
        if not model.endpoint.startswith(scheme):
            raise ValueError(f'{adapter} requires a {scheme[0]} endpoint')
        if model.backend_options.get('joint_frame', 'legacy') not in JOINT_FRAMES:
            raise ValueError('joint_frame must be legacy or current')
        if adapter == 'g05_so101':
            padded = model.backend_options.get('padded_cameras', ['wrist_left'])
            if not isinstance(padded, list) or any(not isinstance(name, str) or not name for name in padded):
                raise ValueError('padded_cameras must be a list of G05 camera keys')

    async def infer(self, model, observation):
        self.validate(model)
        if not observation.instruction.strip():
            raise ValueError('observation instruction is empty')
        images = {image.sensor_id: image for image in observation.sensors}
        if len(images) != len(observation.sensors):
            raise ValueError('Duplicate camera sensor IDs')
        external = _rgb(images.get(self.external_sensor), self.external_sensor)
        wrist = _rgb(images.get(self.wrist_sensor), self.wrist_sensor)
        signs, offsets = JOINT_FRAMES[model.backend_options.get('joint_frame', 'legacy')]
        state = signs * np.r_[_state(observation, 'joint_position', (5,)),
                              _state(observation, 'gripper_position', (1,))] + offsets
        if model.backend_options['adapter'] == 'g05_so101':
            actions = await self._g05(model, observation.instruction, external, wrist, state)
        else:
            payload = dict(external_cam=_json_array(external), wrist_cam=_json_array(wrist),
                           state=_json_array(state), instruction=observation.instruction,
                           num_steps=model.backend_options.get('num_steps', 10))
            actions = await asyncio.to_thread(self._request, model, payload)
        # Longer model chunks are replanned from a fresh observation after max_horizon steps.
        return ((_finite_matrix(actions, 6)[:model.max_horizon] - offsets) * signs).astype(np.float32)

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
        return _json_decode(raw['actions'])

    async def _g05(self, model, instruction, external, wrist, state):
        """One upstream connection per chunk: infer once, then drain its cached steps."""
        try:
            import msgpack
            import websockets
        except ImportError as exc:
            raise RuntimeError('install colosseum-policy-server[droid] for the G05 adapter') from exc

        def pack(value):
            return msgpack.packb(value, default=_msgpack_default)

        def unpack(value):
            return msgpack.unpackb(value, object_hook=_msgpack_object, raw=False, strict_map_key=False)

        images = {'exterior': external.transpose(2, 0, 1), 'wrist_right': wrist.transpose(2, 0, 1)}
        for name in model.backend_options.get('padded_cameras', ['wrist_left']):
            images.setdefault(name, np.zeros_like(images['exterior']))
        request = {'images': images, 'state': {'right_arm': state},
                   'task': instruction, 'embodiment_type': 'so100', 'frequency': float(model.control_hz)}
        steps = []
        async with websockets.connect(model.endpoint, compression=None, max_size=None,
                                      open_timeout=30, close_timeout=5) as socket:
            unpack(await asyncio.wait_for(socket.recv(), 30))  # {"action_steps": N}
            while True:
                await socket.send(pack(request))
                response = unpack(await asyncio.wait_for(socket.recv(), 240))
                action = response.get('action') if isinstance(response, Mapping) else None
                if not isinstance(action, Mapping) or 'error' in response or 'right_arm' not in action:
                    raise ValueError('G05 service returned no valid action')
                step = np.asarray(action['right_arm'], dtype=np.float32).reshape(-1)
                if step.shape != (6,):
                    raise ValueError('G05 SO100 action requires six values')
                steps.append(step)
                if response.get('need_obs', True) or len(steps) >= model.max_horizon:
                    return np.stack(steps)
                request = {}  # served from the chunk cached for this connection
