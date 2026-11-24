"""Robot-independent contract for local policy backend plugins."""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

import numpy as np

from .. import colosseum_pb2 as pb


class PolicyBackend(Protocol):
    """Inference contract; optional async start_session(model, request) and
    end_session() hooks run under the runtime's exclusive session lock.

    end_session also runs after partial startup, errors, and cancellation.
    Stateless backends need only infer().
    """
    async def infer(self, model: "LocalModel", observation: pb.Observation) -> np.ndarray:
        """Return finite (horizon, action_dim) actions in the configured action space."""
        ...


class PolicyStopped(Exception):
    """An agent requested termination, not a verified task success."""

    def __init__(self, reason: str):
        if reason not in {"done", "give_up"}:
            raise ValueError("invalid policy stop reason")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class LocalModel:
    name: str; url: str; revision: str; action_space: str; action_dim: int; control_hz: int; max_horizon: int; endpoint: str
    launcher: tuple[str, ...] = (); backend_options: Mapping[str, Any] = field(default_factory=dict)

    model_type: str = "vla"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LocalModel":
        required = ("name", "url", "revision", "action_space", "action_dim", "control_hz", "max_horizon", "endpoint")
        if not all(key in value for key in required):
            raise ValueError("local model is incomplete")
        launcher, options = value.get("launcher", []), value.get("backend_options", {})
        if not isinstance(launcher, list) or not all(isinstance(arg, str) and arg for arg in launcher) or not isinstance(options, Mapping):
            raise ValueError("invalid model launcher or backend options")
        model = cls(str(value["name"]), str(value["url"]), str(value["revision"]), str(value["action_space"]), int(value["action_dim"]), int(value["control_hz"]), int(value["max_horizon"]), str(value["endpoint"]), tuple(launcher), dict(options), str(value.get("model_type", "vla")))
        endpoint = urlsplit(model.endpoint)
        # Cloud model identity travels in the existing string fields. This is
        # deliberately limited to the in-process agent, not arbitrary endpoints.
        identity = urlsplit(model.url)
        cloud = (
            identity.scheme == "llm" and identity.netloc in {"openai", "x-ai", "anthropic", "google"}
            and bool(identity.path.strip("/")) and not identity.query and not identity.fragment
            and model.endpoint == "inprocess://inspect-agent" and not model.launcher
            and bool(model.revision.strip())
        )
        try: is_loopback = endpoint.hostname is not None and ipaddress.ip_address(endpoint.hostname).is_loopback
        except ValueError: is_loopback = endpoint.hostname == "localhost"
        local_endpoint = endpoint.scheme in {"ws", "wss", "http", "tcp"} and is_loopback and endpoint.port is not None
        # A registered HF manifest can identify a cloud-agent configuration for
        # existing Routers that only accept HF source URLs and commit revisions.
        agent_manifest = (model.endpoint == "inprocess://inspect-agent" and not model.launcher
                          and isinstance(options.get("agent_model"), str) and bool(options["agent_model"]))
        legacy = (model.url.startswith("https://huggingface.co/") and len(model.revision) == 40
                  and (local_endpoint or agent_manifest))
        typed_cloud = (model.model_type == "llm" and identity.scheme == 'https' and bool(identity.hostname)
            and not any((identity.username, identity.password, identity.query, identity.fragment))
            and model.name.startswith(('gpt-', 'grok-', 'claude-', 'gemini-'))
            and model.endpoint == "inprocess://inspect-agent" and not model.launcher and bool(model.revision.strip()))
        valid_source = typed_cloud if model.model_type == "llm" else (cloud or legacy)
        if (model.model_type not in {"vla", "llm"} or not model.name or not valid_source or not model.action_space or model.action_dim < 1 or model.control_hz < 1 or model.max_horizon < 1):
            raise ValueError("invalid model contract")
        return model

    def public_spec(self) -> dict[str, Any]:
        result = {key: getattr(self, key) for key in ("url", "revision", "action_space", "action_dim", "control_hz", "max_horizon")}
        if self.model_type == "llm":
            result.update(model_type="llm", name=self.name)
        return result
