import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from colosseum_policy_server.local_runtime import LocalModel, LocalServiceSupervisor, RuntimeConfig

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="process-group and parent-death handling is Linux-only")

# A model service that spawns a worker, records both pids, then listens.
SERVICE = textwrap.dedent("""
    import os, signal, socket, subprocess, sys, time
    port, pidfile, ignore_term = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open(pidfile, "w").write(f"{os.getpid()} {worker.pid}")
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen()
    time.sleep(60)
""")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def service_config(tmp_path, ignore_term=False):
    script, pidfile, port = tmp_path / "service.py", tmp_path / "pids", free_port()
    script.write_text(SERVICE)
    model = LocalModel.from_mapping({
        "name": "svc", "url": "https://huggingface.co/example/model", "revision": "a" * 40,
        "action_space": "joint_position", "action_dim": 6, "control_hz": 30, "max_horizon": 2,
        "endpoint": f"tcp://127.0.0.1:{port}",
        "launcher": [sys.executable, str(script), str(port), str(pidfile), "1" if ignore_term else "0"]})
    return RuntimeConfig("127.0.0.1", 8000, "unused", {}, tmp_path / "logs", 20, {model.name: model}), model, pidfile


def alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def wait_dead(pids, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(0.05)
    return [pid for pid in pids if alive(pid)]


@pytest.mark.asyncio
async def test_stop_ends_the_service_and_its_workers(tmp_path):
    config, model, pidfile = service_config(tmp_path)
    supervisor = LocalServiceSupervisor(config)
    await supervisor.activate(model)
    pids = [int(pid) for pid in pidfile.read_text().split()]
    assert all(alive(pid) for pid in pids)
    await supervisor.stop()
    assert await asyncio.to_thread(wait_dead, pids) == []


@pytest.mark.asyncio
async def test_service_ignoring_sigterm_is_killed_after_grace(tmp_path):
    config, model, pidfile = service_config(tmp_path, ignore_term=True)
    supervisor = LocalServiceSupervisor(config)
    supervisor.stop_grace_seconds = 0.5
    await supervisor.activate(model)
    pids = [int(pid) for pid in pidfile.read_text().split()]
    started = time.monotonic()
    await supervisor.stop()
    assert time.monotonic() - started < 5
    assert await asyncio.to_thread(wait_dead, pids) == []


def test_hard_killed_server_takes_its_service_with_it(tmp_path):
    config, model, pidfile = service_config(tmp_path)
    server = tmp_path / "server.py"
    server.write_text(textwrap.dedent(f"""
        import asyncio
        from pathlib import Path
        from colosseum_policy_server.local_runtime import LocalModel, LocalServiceSupervisor, RuntimeConfig
        model = LocalModel.from_mapping({dict(name=model.name, url=model.url, revision=model.revision,
            action_space=model.action_space, action_dim=model.action_dim, control_hz=model.control_hz,
            max_horizon=model.max_horizon, endpoint=model.endpoint, launcher=list(model.launcher))!r})
        config = RuntimeConfig("127.0.0.1", 8000, "unused", {{}}, Path({str(config.log_dir)!r}), 20, {{model.name: model}})
        async def main():
            await LocalServiceSupervisor(config).activate(model)
            print("ready", flush=True)
            await asyncio.sleep(60)
        asyncio.run(main())
    """))
    process = subprocess.Popen([sys.executable, str(server)], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        pids = [int(pid) for pid in pidfile.read_text().split()]
        # SIGKILL skips every cleanup path in the server, like a crash.
        process.send_signal(signal.SIGKILL)
        process.wait()
        # The worker is a grandchild without the hook; only the service is guaranteed.
        assert wait_dead(pids[:1]) == []
    finally:
        process.kill()
        for pid in [int(pid) for pid in pidfile.read_text().split()] if pidfile.exists() else []:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
