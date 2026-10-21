"""CLI for the real loopback-only Local Protocol runtime."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import signal

from websockets.asyncio.server import serve

from .local_runtime import LocalPolicyRuntime, RuntimeConfig


async def run(config: RuntimeConfig) -> None:
    runtime = LocalPolicyRuntime(config)
    async with serve(runtime.handle, config.host, config.port, compression=None, max_size=32 * 1024 * 1024,
                     ping_interval=20, ping_timeout=60):
        print(f"Local Policy Server listening on ws://{config.host}:{config.port}; real model runtime only", flush=True)
        # A closed terminal (SIGHUP) or kill (SIGTERM) would otherwise exit
        # without stopping the model service; shut down as for Ctrl+C.
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            loop.add_signal_handler(sig, stopped.set)
        try:
            await stopped.wait()
        finally:
            await runtime.supervisor.stop()


def configured_runtime(path: str, robot: str | None = None) -> RuntimeConfig:
    config = RuntimeConfig.from_yaml(path)
    if robot is not None:
        config = replace(config, backend_name=robot)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve registered VLA and LLM models over Local Protocol v1")
    parser.add_argument("--config", default="configs/local-runtime.yaml", help="YAML local runtime configuration (default: configs/local-runtime.yaml)")
    parser.add_argument("--robot", metavar="ROBOT", help="Select an installed robot backend (for example droid or yam)")
    args = parser.parse_args()
    try:
        asyncio.run(run(configured_runtime(args.config, args.robot)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
