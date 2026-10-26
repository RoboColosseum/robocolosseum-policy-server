"""VLA session orchestration, independent of hardware operation.

Model adapter plugins declare supported_robot_types and implement
validate(model), infer(model, observation); optional session hooks are forwarded.
"""
from importlib import metadata

from ..model_adapters.franka_vla import FrankaVLAAdapter, _ADAPTERS
from ..model_adapters.robots import session_robot


def load_model_adapter(name, options):
    if name in {'molmoact2_yam', 'groot_yam', 'pi05_yam', 'g05_yam', 'lingbot_yam'}:
        from ..model_adapters.yam_vla import YAMVLAAdapter
        return YAMVLAAdapter(options)
    if name in {'molmoact2_so101', 'pi05_so101', 'groot_so101', 'g05_so101', 'lingbot_so101'}:
        from ..model_adapters.so101_vla import SO101VLAAdapter
        return SO101VLAAdapter(options)
    if name in _ADAPTERS:
        return FrankaVLAAdapter(options)
    entries = metadata.entry_points()
    entries = entries.select(group='colosseum_policy_server.model_adapters')
    matches = [entry for entry in entries if entry.name == name]
    if len(matches) != 1:
        raise ValueError(f'VLA model adapter {name!r} is not installed')
    return matches[0].load()(dict(options))


class VLABackend:
    def __init__(self, options):
        self.options = dict(options)
        self.active = self.model = None

    async def start_session(self, model, request):
        if self.active is not None:
            raise RuntimeError('Previous VLA session has not closed')
        robot = session_robot(model, request, self.options)
        adapter = load_model_adapter(model.backend_options.get('adapter'), self.options)
        if robot not in getattr(adapter, 'supported_robot_types', ()):
            raise ValueError(f'Model adapter does not support robot_type {robot!r}')
        if not callable(getattr(adapter, 'infer', None)) or not callable(getattr(adapter, 'validate', None)):
            raise ValueError('Model adapter requires validate and infer')
        adapter.validate(model)
        self.active, self.model = adapter, model
        start = getattr(adapter, 'start_session', None)
        if start is not None:
            await start(model, request)

    async def infer(self, model, observation):
        if self.active is None or model != self.model:
            raise RuntimeError('VLA model session is not prepared')
        return await self.active.infer(model, observation)

    async def end_session(self):
        active, self.active, self.model = self.active, None, None
        end = getattr(active, 'end_session', None)
        if end is not None:
            await end()


def create_backend(options):
    return VLABackend(options)
