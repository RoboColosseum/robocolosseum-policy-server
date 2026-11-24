"""Offline Dream recovery tests: fake SDK calls only, no cameras, CAN or remote API."""
import asyncio
import http.client
import ssl
import threading
import unittest
import pytest
from types import SimpleNamespace
from unittest.mock import patch

pytest.importorskip('dream.errors', reason='Install the dreamlabs extra for Dream recovery tests')

from dream.errors import (
    DreamActionError, DreamAuthError, DreamConfigError, DreamObservationError,
    DreamProtocolError, DreamSessionExpired, DreamTimeout,
)
from colosseum_policy_server.backends import dreamlabs as backend_module


def transport_error(cause=None):
    error = DreamProtocolError('transport failed')
    error.__cause__ = cause if cause is not None else ssl.SSLEOFError('TLS EOF')
    return error


class StopSignal:
    def __init__(self, stop_during_wait=False):
        self.stopped = False
        self.delays = []
        self.stop_during_wait = stop_during_wait

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, delay):
        self.delays.append(delay)
        if self.stop_during_wait:
            self.set()
        return self.stopped


class Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.closed = False
        self.session_id = 'offline-test'
        self.mode = 'sync'

    def get_action(self, *, commanded):
        self.calls.append(commanded)
        result = self.replies.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def set_observation(self, *args, **kwargs):
        pass

    def close(self):
        self.closed = True


def backend(client, signal=None):
    value = backend_module.DreamLabsBackend({'checkpoint': 'offline-test'})
    value.client = client
    if signal is not None:
        value._retry_stop = signal
    return value


class RetryTests(unittest.TestCase):
    def test_tls_recovers_after_three_retries_with_same_commanded_state(self):
        reply = object()
        client = Client([transport_error() for _ in range(3)] + [reply])
        stop = StopSignal()
        value = backend(client, stop)
        commanded = {'q_l_chunk': [1] * 7, 'q_r_chunk': [2] * 7}
        self.assertIs(value._get_action(commanded), reply)
        self.assertEqual(stop.delays, [.5, 1, 2])
        self.assertEqual(len(client.calls), 4)
        self.assertTrue(all(item is commanded for item in client.calls))

    def test_tls_exhaustion_preserves_final_exception(self):
        failures = [transport_error() for _ in range(4)]
        client, stop = Client(failures), StopSignal()
        with self.assertRaises(DreamProtocolError) as caught:
            backend(client, stop)._get_action(None)
        self.assertIs(caught.exception, failures[-1])
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(stop.delays, [.5, 1, 2])

    def test_expiry_recovers_after_three_retries(self):
        reply = object()
        client = Client([DreamSessionExpired('expired') for _ in range(3)] + [reply])
        stop = StopSignal()
        self.assertIs(backend(client, stop)._get_action(None), reply)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(stop.delays, [.5, 1, 2])

    def test_expiry_exhaustion_preserves_final_exception(self):
        failures = [DreamSessionExpired('expired') for _ in range(4)]
        client, stop = Client(failures), StopSignal()
        with self.assertRaises(DreamSessionExpired) as caught:
            backend(client, stop)._get_action(None)
        self.assertIs(caught.exception, failures[-1])
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(stop.delays, [.5, 1, 2])

    def test_error_families_have_separate_retry_limits(self):
        failures = []
        for _ in range(3):
            failures.extend([transport_error(), DreamSessionExpired('expired')])
        reply = object()
        client, stop = Client(failures + [reply]), StopSignal()
        self.assertIs(backend(client, stop)._get_action(None), reply)
        self.assertEqual(len(client.calls), 7)
        self.assertEqual(stop.delays, [.5, .5, 1, 1, 2, 2])

    def test_retry_counts_reset_for_each_new_request(self):
        first, second = object(), object()
        client = Client([transport_error(), first, transport_error(), second])
        stop = StopSignal()
        value = backend(client, stop)
        self.assertIs(value._get_action(None), first)
        self.assertIs(value._get_action(None), second)
        self.assertEqual(stop.delays, [.5, .5])

    def test_unrecoverable_errors_do_not_retry(self):
        for error in (
            DreamProtocolError('invalid JSON'),
            transport_error(ssl.SSLCertVerificationError('certificate invalid')),
            DreamAuthError('denied'), DreamConfigError('bad config'),
            DreamObservationError('bad observation'), DreamActionError('bad action'),
            DreamTimeout('request timeout'), ValueError('joint limit'),
        ):
            with self.subTest(error=type(error).__name__):
                client, stop = Client([error]), StopSignal()
                with self.assertRaises(type(error)) as caught:
                    backend(client, stop)._get_action(None)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(stop.delays, [])

    def test_wrapped_socket_disconnects_retry(self):
        for cause in (ConnectionResetError(), BrokenPipeError(), http.client.RemoteDisconnected(),
                      ssl.SSLZeroReturnError()):
            with self.subTest(cause=type(cause).__name__):
                error = transport_error(cause)
                wrapper = DreamProtocolError('wrapped')
                wrapper.__cause__ = error
                client, stop = Client([wrapper, 'fresh reply']), StopSignal()
                self.assertEqual(backend(client, stop)._get_action(None), 'fresh reply')
                self.assertEqual(len(client.calls), 2)

    def test_stop_during_backoff_prevents_next_attempt(self):
        error = transport_error()
        client = Client([error, object()])
        with self.assertRaises(DreamProtocolError) as caught:
            backend(client, StopSignal(stop_during_wait=True))._get_action(None)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(client.calls), 1)

    def test_already_stopped_worker_makes_no_request(self):
        client, stop = Client([]), StopSignal()
        stop.set()
        with self.assertRaises(asyncio.CancelledError):
            backend(client, stop)._get_action(None)
        self.assertEqual(client.calls, [])

    def test_exception_chain_cycle_is_not_retryable(self):
        error = DreamProtocolError('bad response')
        error.__cause__ = error
        self.assertFalse(backend_module._transient_transport_error(error))


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    def prepare_infer(self, client):
        value = backend(client)
        value.replies = 1  # Skip planner warm-up for cancellation tests.
        value._advance_plan = lambda now: None
        value._next_chunk_due = lambda now: True
        value.playing_since = None
        value.chunk = None
        value.asked_plan_s = value.plan_s = 0.0
        value.last_target = None
        return value

    async def test_outer_timeout_stops_backoff_and_collects_worker(self):
        client = Client([transport_error(), object()])
        value = self.prepare_infer(client)
        with patch.object(backend_module, '_model_inputs', return_value=({}, {})):
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(value.infer(SimpleNamespace(), SimpleNamespace()), .03)
        self.assertTrue(value._retry_stop.is_set())
        await asyncio.wait_for(value.end_session(), 1)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.closed)
        self.assertTrue(value.request.done())
        self.assertIsNone(value.last_target)

    async def test_cancelled_inflight_reply_is_not_installed_and_close_waits(self):
        started, release = threading.Event(), threading.Event()

        class BlockingClient(Client):
            def get_action(self, *, commanded):
                self.calls.append(commanded)
                started.set()
                if not release.wait(2):
                    raise RuntimeError('test worker was not released')
                return {'q_l_chunk': [[0] * 7], 'q_r_chunk': [[0] * 7]}

        client = BlockingClient([])
        value = self.prepare_infer(client)
        try:
            with patch.object(backend_module, '_model_inputs', return_value=({}, {})):
                task = asyncio.create_task(value.infer(SimpleNamespace(), SimpleNamespace()))
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(.005)
                self.assertTrue(started.is_set())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            close = asyncio.create_task(value.end_session())
            await asyncio.sleep(.01)
            self.assertFalse(client.closed)
            release.set()
            await asyncio.wait_for(close, 1)
            self.assertTrue(client.closed)
            self.assertIsNone(value.chunk)
            self.assertIsNone(value.last_target)
            self.assertTrue(value.request.cancelled())
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main()
