"""Encode executable actions and optional original model output separately."""
from collections.abc import Mapping
import numpy as np
from . import colosseum_pb2 as pb

RAW_SOURCE = 'model_output_before_action_safety_clipping'
MAX_RAW_HORIZON = 4096


def make_action_plan(output, model, *, sequence, start_step):
    result = output if isinstance(output, Mapping) else {'actions': output}
    actions = np.asarray(result['actions'], dtype=np.float32)
    if (actions.ndim != 2 or not 1 <= len(actions) <= model.max_horizon
            or actions.shape[1] != model.action_dim or not np.isfinite(actions).all()):
        raise ValueError('Backend returned an invalid action array')
    plan = pb.ActionPlan(request_sequence=sequence, plan_id=sequence,
        start_step=start_step, valid_until_step=start_step + len(actions) - 1,
        control_hz=model.control_hz,
        actions=pb.Tensor(shape=actions.shape, dtype=pb.FLOAT32, data=actions.tobytes()))
    if 'raw_actions' in result:
        raw = np.ascontiguousarray(result['raw_actions'])
        if (raw.ndim != 2 or raw.shape[1] != model.action_dim
                or not len(actions) <= len(raw) <= MAX_RAW_HORIZON
                or raw.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or not np.isfinite(raw).all() or result.get('raw_action_source') != RAW_SOURCE):
            raise ValueError('Backend returned an invalid original action chunk/provenance')
        plan.raw_actions.CopyFrom(pb.Tensor(shape=raw.shape,
            dtype=pb.FLOAT64 if raw.dtype == np.float64 else pb.FLOAT32, data=raw.tobytes()))
        plan.raw_action_space = model.action_space
        plan.raw_action_source = RAW_SOURCE
    return plan
