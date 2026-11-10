"""Serve a local-runtime model to the Router as a remote Policy Server.

The same YAML that ``colosseum-policy-local`` uses (backend, models, launchers) runs
remote evaluations: the runner dials out to the Router with a Policy Server token,
so the inference machine needs no public address or inbound port. The model is
loaded once and stays warm across trials; the connection re-registers after drops.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time

import numpy as np

from .local_runtime import LocalServiceSupervisor, RuntimeConfig, load_backend
from .sdk import AsyncColosseumPolicySDK, StaleObservationError


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


class RemotePolicyRunner:
    def __init__(self, config: RuntimeConfig, model_name: str, sdk: AsyncColosseumPolicySDK, *,
                 backend=None, supervisor=None) -> None:
        if model_name not in config.models:
            raise ValueError(f"model {model_name!r} is not in the runtime config: {', '.join(config.models)}")
        self.model = config.models[model_name]
        if self.model.model_type != "vla":
            raise ValueError("Remote runner serves VLA models; LLM agents need per-trial credentials")
        self.sdk = sdk
        self.backend = backend or load_backend(config.backend_name, config.backend_options)
        self.supervisor = supervisor or LocalServiceSupervisor(config)
        self.session = ""

    @classmethod
    def create(cls, config: RuntimeConfig, model_name: str, *, router_url: str, token: str, policy_id: str,
               ping_interval: float = 20, ping_timeout: float = 60, **kwargs) -> "RemotePolicyRunner":
        model = config.models.get(model_name)
        if model is None:
            raise ValueError(f"model {model_name!r} is not in the runtime config: {', '.join(config.models)}")
        sdk = AsyncColosseumPolicySDK(
            router_url=router_url, token=token, policy_id=policy_id, policy_revision=model.revision,
            action_spaces=(model.action_space,), control_hz=model.control_hz, max_horizon=model.max_horizon,
            ping_interval=ping_interval, ping_timeout=ping_timeout, reconnect=True)
        return cls(config, model_name, sdk, **kwargs)

    async def start(self) -> None:
        """Load the model before registering, so no trial is matched to a cold server."""
        started = time.monotonic()
        _log("model_loading", model=self.model.name)
        await self.supervisor.activate(self.model)
        _log("model_ready", model=self.model.name, seconds=round(time.monotonic() - started, 1))
        await self.sdk.connect()
        _log("registered", policy_id=self.sdk.policy_id, server_id=self.sdk.server_id)

    async def _end_session(self) -> None:
        end = getattr(self.backend, "end_session", None)
        if end is not None:
            await end()

    async def _begin_session(self, session_id: str) -> None:
        # Fresh adapter state per trial (history, agent memory).
        await self._end_session()
        robot = self.sdk.robot.robot_type if self.sdk.robot is not None else None
        start = getattr(self.backend, "start_session", None)
        if start is not None:
            await start(self.model, {"robot_type": robot} if robot else {})
        self.session = session_id
        _log("session", session=session_id, robot=robot)

    async def step(self) -> None:
        observation = await self.sdk.get_obs()
        if observation.session_id != self.session:
            await self._begin_session(observation.session_id)
        started = time.monotonic()
        budget = (observation.deadline_ms or 30000) / 1000
        try:
            actions = await asyncio.wait_for(self.backend.infer(self.model, observation.message), budget)
            actions = np.asarray(actions, dtype=np.float32)
            await self.sdk.send_action(actions, observation=observation, control_hz=self.model.control_hz)
        except StaleObservationError:
            _log("stale_observation", sequence=observation.sequence)
            return
        except Exception as exc:
            _log("inference_failed", sequence=observation.sequence, error=f"{type(exc).__name__}: {exc}")
            await self.sdk.send_error("INFERENCE_FAILED", "Remote inference failed", observation=observation)
            return
        _log("step", sequence=observation.sequence, control_step=observation.control_step,
             horizon=len(actions), infer_s=round(time.monotonic() - started, 3))

    async def run(self, stopped: asyncio.Event | None = None) -> None:
        stopped = stopped or asyncio.Event()
        await self.start()
        try:
            while not stopped.is_set():
                step = asyncio.ensure_future(self.step())
                stop = asyncio.ensure_future(stopped.wait())
                done, _ = await asyncio.wait({step, stop}, return_when=asyncio.FIRST_COMPLETED)
                if stop in done:
                    step.cancel()
                    break
                stop.cancel()
                step.result()
        finally:
            await self.sdk.close()
            await self._end_session()
            await self.supervisor.stop()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Serve a local-runtime model to the Router for remote evaluation")
    parser.add_argument("--config", required=True, help="local runtime YAML (same file as colosseum-policy-local)")
    parser.add_argument("--model", required=True, help="model name from the config")
    parser.add_argument("--router", required=True, help="Router URL, e.g. wss://router.example.com")
    parser.add_argument("--policy-id", required=True, help="policy ID of the deployment registered on the Router")
    parser.add_argument("--token-env", default="COLOSSEUM_POLICY_TOKEN",
                        help="environment variable holding the Policy Server token (default: COLOSSEUM_POLICY_TOKEN)")
    parser.add_argument("--ping-interval", type=float, default=20)
    parser.add_argument("--ping-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        parser.error(f"set the Policy Server token in ${args.token_env}")
    runner = RemotePolicyRunner.create(RuntimeConfig.from_yaml(args.config), args.model, router_url=args.router,
                                       token=token, policy_id=args.policy_id,
                                       ping_interval=args.ping_interval, ping_timeout=args.ping_timeout)

    async def run() -> None:
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            loop.add_signal_handler(sig, stopped.set)
        await runner.run(stopped)

    asyncio.run(run())


if __name__ == "__main__":
    main()
