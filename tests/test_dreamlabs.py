"""Dream wire mapping and playback checks without a cloud session or robot."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_policy_server.backends.dreamlabs import DreamLabsBackend, _chunk_from_reply
from colosseum_policy_server.local_runtime import RuntimeConfig


def test_example_session_and_playback(monkeypatch):
    config = RuntimeConfig.from_yaml(Path(__file__).parents[1] / 'configs/local-runtime-yam-dreamlabs.yaml.example')
    model = next(iter(config.models.values()))
    calls = []

    class Client:
        session_id = 'test'

        def policy(self, options):
            calls.append(options)

        def close(self):
            calls.append('closed')

    monkeypatch.setitem(sys.modules, 'dream', SimpleNamespace(DreamClient=Client))
    backend = DreamLabsBackend({'checkpoint': 'dreamlabs/dream-1-beta'})
    asyncio.run(backend.start_session(model, {}))
    assert calls == [{'checkpoint': 'dreamlabs/dream-1-beta', 'strict': False, 'action_hz': 20}]
    backend.chunk = np.repeat(np.arange(5, dtype=np.float32)[:, None], 14, axis=1)
    # A 20 Hz model is interpolated onto the Client's 30 Hz control steps.
    np.testing.assert_allclose(backend._targets()[:, 0], [0, 2 / 3, 4 / 3])
    backend.playing_since = 10
    backend._advance_plan(20)
    assert backend.plan_s == pytest.approx(0.2)
    asyncio.run(backend.end_session())
    assert calls[-1] == 'closed' and backend.client is None


def test_chunk_preserves_left_right_order_and_clips_grippers():
    left = [[1, 2, 3, 4, 5, 6, -0.01]]
    right = [[7, 8, 9, 10, 11, 12, 1.01]]
    result = _chunk_from_reply({'q_l_chunk': left, 'q_r_chunk': right})
    np.testing.assert_array_equal(result, [[1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 1]])


@pytest.mark.parametrize('left,right', [([], []), ([[0] * 6], [[0] * 6]),
    ([[0] * 7], [[0] * 7] * 2), ([[float('nan')] * 7], [[0] * 7])])
def test_invalid_cloud_chunks_are_rejected(left, right):
    with pytest.raises(ValueError):
        _chunk_from_reply({'q_l_chunk': left, 'q_r_chunk': right})
