"""Loopback Local Protocol runtime backed by installed policy plugins."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import metadata
import ipaddress
import json
from pathlib import Path
import socket
import time
import traceback
from typing import Any, Mapping
from urllib.parse import urlsplit

import numpy as np
import yaml

from . import colosseum_pb2 as pb
from .local_protocol import decode_control, encode_control
from .backends import LocalModel, PolicyBackend
from .backends.base import PolicyStopped


def load_backend(name: str, options: Mapping[str, Any]) -> PolicyBackend:
    entries = metadata.entry_points()
    entries = entries.select(group="colosseum_policy_server.backends") if hasattr(entries, "select") else entries.get("colosseum_policy_server.backends", [])
    matches = [entry for entry in entries if entry.name == name]
    if len(matches) != 1:
        raise ValueError(f"backend {name!r} is not installed")
    backend = matches[0].load()(dict(options))
    if not callable(getattr(backend, "infer", None)):
        raise ValueError("backend must define async infer")
    return backend


@dataclass(frozen=True)
class RuntimeConfig:
    host: str; port: int; backend_name: str; backend_options: Mapping[str, Any]; log_dir: Path; start_timeout_seconds: int; models: Mapping[str, LocalModel]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        backend = raw.get("backend", {}) if isinstance(raw, Mapping) else None
        if not isinstance(backend, Mapping) or not isinstance(backend.get("name", "auto"), str) or not isinstance(backend.get("options", {}), Mapping):
            raise ValueError("runtime backend must contain a string name and mapping options")
        host, port = str(raw.get("host", "127.0.0.1")), raw.get("port", 8000)
        try: is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError: is_loopback = host == "localhost"
        if not is_loopback or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("runtime host must be loopback and port must be valid")
        rows = raw.get("models", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("models must be nonempty")
        models = [LocalModel.from_mapping(row) for row in rows if isinstance(row, Mapping)]
        if len(models) != len(rows) or len({model.name for model in models}) != len(models):
            raise ValueError("models must be complete and uniquely named")
        timeout = raw.get("start_timeout_seconds", 180)
        if type(timeout) is not int or timeout < 1:
            raise ValueError("start_timeout_seconds must be positive")
        return cls(host, port, backend.get("name", "auto"), dict(backend.get("options", {})), Path(str(raw.get("log_dir", "local-policy-logs"))), timeout, {model.name: model for model in models})


class LocalServiceSupervisor:
    def __init__(self, config: RuntimeConfig): self.config, self.process, self.active, self.log = config, None, None, None
    @staticmethod
    async def _listening(endpoint: str) -> bool:
        parsed = urlsplit(endpoint)
        if parsed.hostname is None or parsed.port is None:
            return False
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(parsed.hostname, parsed.port), timeout=1)
        except (OSError, asyncio.TimeoutError, socket.gaierror):
            return False
        writer.close()
        await writer.wait_closed()
        return True
    async def activate(self, model: LocalModel) -> None:
        if not model.launcher: return
        await self.stop(); self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.config.log_dir / f"{model.name}-{time.strftime('%Y%m%d-%H%M%S')}.log").open("xb")
        self.process = await asyncio.create_subprocess_exec(*model.launcher, stdout=self.log, stderr=asyncio.subprocess.STDOUT, start_new_session=True); self.active = model
        deadline = time.monotonic() + self.config.start_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                raise RuntimeError(f"launcher exited with status {self.process.returncode}")
            if await self._listening(model.endpoint):
                return
            await asyncio.sleep(.25)
        raise RuntimeError("launcher did not make its loopback endpoint available before timeout")
    async def stop(self) -> None:
        process, self.process, self.active = self.process, None, None
        if process is not None and process.returncode is None: process.terminate(); await process.wait()
        if self.log is not None: self.log.close(); self.log = None


class LocalPolicyRuntime:
    progress_interval = 10.0
    observation_timeout_seconds = 90

    @staticmethod
    def _progress(event, run_id, model, **details):
        print(json.dumps(dict(event=event, run_id=run_id, model=model.name,
                              **details)), flush=True)

    async def _infer_with_progress(self, run_id, model, observation, deadline_ms):
        started = time.monotonic()
        details = dict(step=observation.control_step, deadline_ms=deadline_ms)
        self._progress('inference_start', run_id, model, **details)

        async def heartbeat():
            while True:
                await asyncio.sleep(self.progress_interval)
                self._progress('inference_waiting', run_id, model, **details,
                               elapsed_s=round(time.monotonic() - started, 2))

        reporter = asyncio.create_task(heartbeat())
        try:
            result = await asyncio.wait_for(self.backend.infer(model, observation), deadline_ms / 1000)
        except BaseException as exc:
            self._progress('inference_failed', run_id, model, **details,
                           elapsed_s=round(time.monotonic() - started, 2),
                           error_type=type(exc).__name__)
            raise
        else:
            self._progress('inference_returned', run_id, model, **details,
                           elapsed_s=round(time.monotonic() - started, 2))
            return result
        finally:
            reporter.cancel()
            try:
                await reporter
            except asyncio.CancelledError:
                pass

    def __init__(self, config: RuntimeConfig, backend: PolicyBackend | None = None, supervisor: LocalServiceSupervisor | None = None):
        self.config, self.backend, self.supervisor, self.lock = config, backend or load_backend(config.backend_name, config.backend_options), supervisor or LocalServiceSupervisor(config), asyncio.Lock()
    async def _error(self, ws, run_id: str, code: str, message: str) -> None:
        payload = pb.Error(code=code, message=message, retryable=False).SerializeToString()
        await ws.send(pb.RelayFrame(protocol_version=1, type=pb.ERROR, session_id=run_id, payload=payload).SerializeToString())
    async def handle(self, ws) -> None:
        run_id = ""
        try:
            request = decode_control(await asyncio.wait_for(ws.recv(), 30))
            if request.get("type") == "capabilities":
                await ws.send(encode_control({"type":"capabilities", "protocol_version":1, "verification_only":bool(getattr(self.backend, "verification_only", False))})); return
            if request.get("type") != "prepare" or not isinstance(request.get("run_id"), str) or not isinstance(request.get("model"), Mapping): raise ValueError("invalid preparation")
            try:
                local_options = json.loads(request.get('message') or '{}').get('client_options', {})
            except (ValueError, TypeError, AttributeError):
                local_options = {}
            if not isinstance(local_options, dict):
                raise ValueError('invalid local Client options')
            request = {**request, **{key: local_options[key]
                                   for key in ('auth_method', 'use_local_action_contract')
                                   if key in local_options}}
            run_id = request["run_id"]
            # Legacy profiles are opaque metadata; subfolder selection is deferred.
            requested_model = {key: value for key, value in request["model"].items() if key not in {"subfolder", "runtime_profile"}}
            if requested_model.get("model_type", "vla") == "vla":
                requested_model.pop("model_type", None)
                requested_model.pop("name", None)
            matches = [model for model in self.config.models.values() if model.public_spec() == requested_model]
            if len(matches) != 1: raise ValueError("unregistered model")
            model = matches[0]
            local_eef = model.model_type == 'llm' and model.backend_options.get('control_mode') == 'eef_abs_pose'
            if local_eef and request.get('use_local_action_contract') is not True:
                raise ValueError('EEF local action override requires Client use_local_action_contract: true')
            verification_only = bool(getattr(self.backend, "verification_only", False))
            if verification_only and request.get("test") is not True:
                raise ValueError("Loopback backend requires test: true")
            async with self.lock:
                self._progress('model_start', run_id, model)
                try:
                    try:
                        await self.supervisor.activate(model)
                        start = getattr(self.backend, "start_session", None)
                        if start is not None:
                            try:
                                await start(model, request)
                            finally:
                                request.pop("api_key", None)
                    except Exception:
                        print(traceback.format_exc(), flush=True)
                        await self._error(ws, run_id, "MODEL_START_FAILED", "Model startup failed")
                        return
                    ready = {"type":"ready", "run_id":run_id, "preparation_id":request.get("preparation_id", ""),
                             "model":dict(request["model"]), "loaded":not verification_only,
                             "verification_only":verification_only}
                    if local_eef:
                        ready['message'] = json.dumps({'effective_model': {
                            **dict(request['model']), 'action_space': 'eef_abs_pose'}})
                    await ws.send(encode_control(ready))
                    self._progress('model_ready', run_id, model)
                    await self._serve(ws, run_id, model)
                finally:
                    end = getattr(self.backend, "end_session", None)
                    if end is not None:
                        await end()
                    await self.supervisor.stop()
        except Exception: print(traceback.format_exc(), flush=True); await self._error(ws, run_id, "INVALID_REQUEST", "Invalid local policy request")
    async def _serve(self, ws, run_id: str, model: LocalModel) -> None:
        sequence = 1
        while True:
            started = time.monotonic()
            stage = 'receive'
            reason = 'Could not receive local observation'
            details = {'expected_sequence': sequence}
            try:
                # Readiness precedes operator approval and hardware construction.
                # Do not expire a prepared connection while the Client is still
                # setting up. recv() still ends on disconnect/cancellation.
                if sequence == 1:
                    raw = await ws.recv()
                else:
                    raw = await asyncio.wait_for(ws.recv(), self.observation_timeout_seconds)
                stage = 'decode_frame'
                reason = 'Malformed local observation frame'
                frame = pb.RelayFrame.FromString(raw)
                if frame.type == pb.SESSION_CLOSE and frame.session_id == run_id: return
                stage = 'validate_frame'
                details.update(received_type=frame.type, received_sequence=frame.sequence,
                               deadline_ms=frame.deadline_ms, session_matches=frame.session_id == run_id)
                violations = []
                if frame.type != pb.OBSERVATION: violations.append('expected OBSERVATION frame')
                if frame.session_id != run_id: violations.append('session mismatch')
                if frame.sequence != sequence:
                    violations.append(f'sequence mismatch: expected {sequence}, received {frame.sequence}')
                if frame.deadline_ms < 1: violations.append('deadline_ms must be positive')
                reason = 'Invalid local observation: ' + '; '.join(violations)
                if violations: raise ValueError(reason)
                stage = 'decode_observation'
                reason = 'Malformed local observation payload'
                observation = pb.Observation.FromString(frame.payload)
            except asyncio.TimeoutError:
                self._progress('observation_timeout', run_id, model, **details,
                               timeout_s=self.observation_timeout_seconds,
                               elapsed_s=round(time.monotonic() - started, 2))
                await self._error(ws, run_id, 'OBSERVATION_TIMEOUT',
                                  f'No observation received within {self.observation_timeout_seconds:g}s; '
                                  f'waiting for sequence {sequence}. This is not a model inference timeout.')
                return
            except Exception as exc:
                # Log protocol metadata, never images, observation bytes or credentials.
                self._progress('observation_rejected', run_id, model, **details,
                               stage=stage, error_type=type(exc).__name__, reason=reason,
                               elapsed_s=round(time.monotonic() - started, 2))
                await self._error(ws, run_id, 'INVALID_REQUEST', reason)
                return
            try: actions = np.asarray(await self._infer_with_progress(run_id, model, observation, frame.deadline_ms), dtype=np.float32)
            except asyncio.TimeoutError: await self._error(ws, run_id, "INFERENCE_TIMEOUT", "Local inference exceeded its deadline"); return
            except PolicyStopped as exc:
                await self._error(ws, run_id, "POLICY_STOPPED", f"Agent requested {exc.reason}; operator scoring is required")
                return
            except Exception: print(traceback.format_exc(), flush=True); await self._error(ws, run_id, "INFERENCE_FAILED", "Local inference failed"); return
            if actions.ndim != 2 or not 1 <= actions.shape[0] <= model.max_horizon or actions.shape[1] != model.action_dim or not np.isfinite(actions).all(): await self._error(ws, run_id, "INVALID_REQUEST", "Backend returned an invalid action array"); return
            plan = pb.ActionPlan(request_sequence=sequence, plan_id=sequence, start_step=observation.control_step, valid_until_step=observation.control_step + len(actions) - 1, control_hz=model.control_hz, actions=pb.Tensor(shape=actions.shape, dtype=pb.FLOAT32, data=actions.tobytes()))
            await ws.send(pb.RelayFrame(protocol_version=1, type=pb.ACTION_PLAN, session_id=run_id, sequence=sequence, payload=plan.SerializeToString()).SerializeToString()); sequence += 1
