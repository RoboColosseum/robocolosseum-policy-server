from __future__ import annotations

import asyncio
import ssl
import threading
from pathlib import Path
from typing import Any, Coroutine

import numpy as np

from .sdk import AsyncColosseumPolicySDK, Observation, RobotConfiguration


class ColosseumPolicySDK:
    """Blocking policy SDK backed by an asyncio event loop in a background thread."""

    def __init__(
        self,
        *,
        router_url: str,
        token: str,
        server_id: str = "",
        policy_id: str = "",
        policy_revision: str = "",
        action_spaces: tuple[str, ...] = ("joint_position",),
        control_hz: int = 15,
        max_horizon: int = 16,
    ) -> None:
        self._initialize(
            AsyncColosseumPolicySDK(
                router_url=router_url,
                token=token,
                server_id=server_id,
                policy_id=policy_id,
                policy_revision=policy_revision,
                action_spaces=action_spaces,
                control_hz=control_hz,
                max_horizon=max_horizon,
            )
        )

    def _initialize(self, async_sdk: AsyncColosseumPolicySDK) -> None:
        self._async_sdk = async_sdk
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ColosseumPolicySDK":
        instance = cls.__new__(cls)
        instance._initialize(AsyncColosseumPolicySDK.from_yaml(path))
        return instance

    @property
    def router_url(self) -> str:
        return self._async_sdk.router_url

    @property
    def server_id(self) -> str:
        return self._async_sdk.server_id

    @property
    def policy_id(self) -> str:
        return self._async_sdk.policy_id

    @property
    def session_id(self) -> str:
        return self._async_sdk.session_id

    @property
    def robot(self) -> RobotConfiguration | None:
        return self._async_sdk.robot

    @property
    def selected_action_space(self) -> str:
        return self._async_sdk.selected_action_space

    @property
    def action_spaces(self) -> tuple[str, ...]:
        return self._async_sdk.action_spaces

    @property
    def control_hz(self) -> int:
        return self._async_sdk.control_hz

    @property
    def max_horizon(self) -> int:
        return self._async_sdk.max_horizon

    def start(self, *, ssl_context: ssl.SSLContext | None = None) -> "ColosseumPolicySDK":
        if self._thread is not None:
            raise RuntimeError("SDK is already started")
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        startup_errors: list[BaseException] = []

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_sdk.connect(ssl_context=ssl_context))
            except BaseException as exc:
                startup_errors.append(exc)
            finally:
                ready.set()
            if not startup_errors:
                loop.run_forever()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        thread = threading.Thread(target=run_loop, name="colosseum-policy-sdk", daemon=True)
        self._loop = loop
        self._thread = thread
        thread.start()
        ready.wait()
        if startup_errors:
            self._loop = None
            self._thread = None
            thread.join(timeout=5)
            raise startup_errors[0]
        return self

    def get_obs(self, *, timeout: float | None = None) -> Observation:
        return self._run(self._async_sdk.get_obs(timeout=timeout))

    def send_action(
        self,
        actions: np.ndarray,
        *,
        observation: Observation | None = None,
        plan_id: int | None = None,
        start_step: int | None = None,
        valid_until_step: int | None = None,
        control_hz: int | None = None,
        done: bool = False,
    ) -> None:
        self._run(
            self._async_sdk.send_action(
                actions,
                observation=observation,
                plan_id=plan_id,
                start_step=start_step,
                valid_until_step=valid_until_step,
                control_hz=control_hz,
                done=done,
            )
        )

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            self._run(self._async_sdk.close())
        finally:
            self._stop_loop()

    def _run(self, coroutine: Coroutine[Any, Any, Any]):
        loop = self._loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise RuntimeError("SDK is not started; use 'with ColosseumPolicySDK.from_yaml(...) as sdk'")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def _stop_loop(self) -> None:
        loop = self._loop
        thread = self._thread
        self._loop = None
        self._thread = None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> "ColosseumPolicySDK":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
