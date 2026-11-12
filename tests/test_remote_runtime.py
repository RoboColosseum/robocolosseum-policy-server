"""Remote runner: a local-runtime model served to the Router over an outbound connection."""
import asyncio

import numpy as np
import pytest
from websockets.asyncio.server import serve

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.local_runtime import LocalModel, RuntimeConfig
from colosseum_policy_server.remote_runtime import RemotePolicyRunner, main
from colosseum_policy_server.tensors import tensor_from_numpy, tensor_to_numpy


def config(tmp_path, **overrides):
    item = LocalModel.from_mapping({"name": "g05-yam", "url": "https://huggingface.co/example/model",
                                    "revision": "b" * 40, "action_space": "joint_position", "action_dim": 14,
                                    "control_hz": 30, "max_horizon": 32, "endpoint": "http://127.0.0.1:9100",
                                    **overrides})
    return RuntimeConfig("127.0.0.1", 8000, "unused", {}, tmp_path, 1, {item.name: item})


class Backend:
    def __init__(self):
        self.sessions, self.ended, self.calls = [], 0, 0

    async def start_session(self, model, request):
        self.sessions.append(request)

    async def end_session(self):
        self.ended += 1

    async def infer(self, model, observation):
        self.calls += 1
        if observation.instruction == "fail":
            raise RuntimeError("model crashed")
        assert observation.sensors[0].sensor_id == "head_image"  # wire message reaches the backend
        return np.full((32, 14), 0.5, np.float32)


class Supervisor:
    def __init__(self):
        self.events = []

    async def activate(self, model):
        self.events.append("activate")

    async def stop(self):
        self.events.append("stop")


def frame(kind, payload=None, *, session_id="", sequence=0, deadline_ms=0):
    return pb.RelayFrame(protocol_version=1, type=kind, session_id=session_id, sequence=sequence,
                         deadline_ms=deadline_ms,
                         payload=payload.SerializeToString() if payload is not None else b"").SerializeToString()


def yam_observation(instruction, step):
    message = pb.Observation(control_step=step, instruction=instruction)
    message.state["joint_position"].CopyFrom(tensor_from_numpy(np.zeros(12, np.float32)))
    message.sensors.add(sensor_id="head_image", encoding=pb.JPEG, width=640, height=480, data=b"jpeg")
    return message


@pytest.mark.asyncio
async def test_serves_a_trial_and_reports_inference_errors(tmp_path):
    seen = {}
    finished = asyncio.Event()

    async def router(ws):
        hello = pb.Hello.FromString(pb.RelayFrame.FromString(await ws.recv()).payload)
        seen["hello"] = (hello.role, hello.policy_id)
        await ws.send(frame(pb.REGISTERED, pb.Registered(peer_id="peer")))
        robot = pb.RobotSpec(robot_type="yam", joint_count=12, has_gripper=True, control_hz=30,
                             action_spaces=[pb.ActionSpaceSpec(name="joint_position", action_dim=14)])
        await ws.send(frame(pb.MATCHED, pb.MatchResult(session_id="s1", policy_id="acme", robot=robot), session_id="s1"))
        seen["ready"] = pb.SessionReady.FromString(pb.RelayFrame.FromString(await ws.recv()).payload)
        await ws.send(frame(pb.OBSERVATION, yam_observation("pick", 0), session_id="s1", sequence=1, deadline_ms=5000))
        reply = pb.RelayFrame.FromString(await ws.recv())
        seen["plan"] = (reply.type, reply.sequence, pb.ActionPlan.FromString(reply.payload))
        await ws.send(frame(pb.OBSERVATION, yam_observation("fail", 32), session_id="s1", sequence=2, deadline_ms=5000))
        reply = pb.RelayFrame.FromString(await ws.recv())
        seen["error"] = (reply.type, reply.sequence, pb.Error.FromString(reply.payload).code)
        finished.set()
        await ws.wait_closed()

    backend, supervisor = Backend(), Supervisor()
    async with serve(router, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        runner = RemotePolicyRunner.create(config(tmp_path), "g05-yam", router_url=f"ws://127.0.0.1:{port}",
                                           token="pol_t", policy_id="acme", backend=backend, supervisor=supervisor)
        stopped = asyncio.Event()
        task = asyncio.create_task(runner.run(stopped))
        await asyncio.wait_for(finished.wait(), 10)
        stopped.set()
        await asyncio.wait_for(task, 10)

    assert seen["hello"] == (pb.POLICY_SERVER, "acme")
    ready = seen["ready"]
    assert (ready.policy_revision, list(ready.action_spaces), ready.control_hz, ready.max_horizon) == (
        "b" * 40, ["joint_position"], 30, 32)
    kind, sequence, plan = seen["plan"]
    assert (kind, sequence, plan.request_sequence, plan.start_step, plan.valid_until_step, plan.control_hz) == (
        pb.ACTION_PLAN, 1, 1, 0, 31, 30)
    assert tensor_to_numpy(plan.actions).shape == (32, 14)
    assert seen["error"] == (pb.ERROR, 2, "INFERENCE_FAILED")
    assert backend.sessions == [{"robot_type": "yam"}] and backend.calls == 2
    assert supervisor.events == ["activate", "stop"]  # loaded once, before registering


def test_rejects_unknown_and_llm_models(tmp_path):
    with pytest.raises(ValueError, match="not in the runtime config"):
        RemotePolicyRunner.create(config(tmp_path), "missing", router_url="ws://x", token="t", policy_id="p",
                                  backend=Backend(), supervisor=Supervisor())


def test_cli_requires_token_in_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("COLOSSEUM_POLICY_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        main(["--config", "x.yaml", "--model", "m", "--router", "ws://x", "--policy-id", "p"])
    assert "COLOSSEUM_POLICY_TOKEN" in capsys.readouterr().err
