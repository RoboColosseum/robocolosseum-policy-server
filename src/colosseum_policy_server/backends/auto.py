"""Dispatch registered model types without changing existing VLA adapters."""
from .vla import VLABackend
from .inspect_agent import InspectAgentBackend
from .inspect_capx import InspectCapxBackend, is_capx


class AutoBackend:
    def __init__(self, options):
        self.options, self.active = dict(options), None

    async def start_session(self, model, request):
        if self.active is not None:
            raise RuntimeError('Previous model session has not closed')
        shared = {key: value for key, value in self.options.items() if key not in {'llm', 'vla', 'capx'}}
        if is_capx(model):
            self.active = InspectCapxBackend(
                {**shared, **self.options.get('llm', {}), **self.options.get('capx', {})})
            await self.active.start_session(model, request)
        elif model.model_type == 'llm' or model.endpoint == 'inprocess://inspect-agent':
            self.active = InspectAgentBackend({**shared, **self.options.get('llm', {})})
            await self.active.start_session(model, request)
        else:
            self.active = VLABackend({**shared, **self.options.get('vla', {})})
            await self.active.start_session(model, request)

    async def infer(self, model, observation):
        if self.active is None:
            raise RuntimeError('Model session is not prepared')
        return await self.active.infer(model, observation)

    @property
    def control_mode(self):
        return getattr(self.active, 'control_mode', 'joint_pos')

    @property
    def observation_extras(self):
        return getattr(self.active, 'observation_extras', ())

    @property
    def internal_action_dim(self):
        return getattr(self.active, 'internal_action_dim', None)

    async def end_session(self):
        active, self.active = self.active, None
        if active is not None and hasattr(active, 'end_session'):
            await active.end_session()


def create_backend(options):
    return AutoBackend(options)
