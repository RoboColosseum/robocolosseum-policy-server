"""Model workers used by the Local Policy launcher (no robot imports)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import runpy
import sys
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from .numpy_wire import _json_array, _json_decode, _msgpack_default, _msgpack_object

log = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 32 * 1024 * 1024


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["molmoact2", "pi05_lerobot", "groot_n17", "lap_3b", "g05", "lingbot_v2"])
    parser.add_argument("--robot-type", choices=["franka", "yam", "so101"], default="franka",
                        help="embodiment (default: franka); so101: molmoact2/pi05_lerobot/groot_n17/g05/lingbot_v2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--processor", type=Path)
    parser.add_argument("--base-model", type=Path, help="local GR00T N1.7 base snapshot for LeRobot YAM/SO101")
    parser.add_argument("--groot-camera-order", choices=["checkpoint", "top-left-right"], default="checkpoint",
                        help="GR00T YAM camera packing; top-left-right is an experimental override")
    parser.add_argument("--source-root", type=Path, help="installed G05 or LingBot upstream source checkout")
    parser.add_argument("--g05-parts-meta", type=Path, help="original training parts_meta/yam.yaml for G05 YAM")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="Molmo precision")
    parser.add_argument("--patch-molmo-bf16", action="store_true",
                        help="back up and patch the selected Molmo model source with deployed dtype fixes")
    parser.add_argument("--g05-native-attention", action="store_true", help="disable deployed G05 SDPA override")
    parser.add_argument("--lingbot-compile", action="store_true", help="enable optional LingBot torch.compile")
    parser.add_argument("--action-steps", type=int, default=30,
                        help="G05 SO101 steps served per inference; match the registered max_horizon")
    parser.add_argument("--g05-override", action="append", default=[], metavar="KEY=VALUE",
                        help="extra upstream G05 Hydra override for SO101 (repeatable)")
    args = parser.parse_args(argv)
    if args.model == "lingbot_v2" and args.robot_type not in {"yam", "so101"}:
        parser.error("lingbot_v2 currently supports --robot-type yam or so101 only")
    if args.lingbot_compile and args.model != "lingbot_v2":
        parser.error("--lingbot-compile requires lingbot_v2")
    if args.groot_camera_order != "checkpoint" and (args.model != "groot_n17" or args.robot_type != "yam"):
        parser.error("--groot-camera-order top-left-right requires groot_n17 --robot-type yam")
    if not 1 <= args.port <= 65535 or not 1 <= args.num_steps <= 10:
        parser.error("port must be valid and --num-steps must be in [1, 10]")
    if args.robot_type == "yam" and args.model not in {"molmoact2", "groot_n17", "pi05_lerobot", "g05", "lingbot_v2"}:
        parser.error("--robot-type yam is implemented only for molmoact2, groot_n17, pi05_lerobot, g05 and lingbot_v2")
    if args.robot_type == "so101" and args.model not in {"molmoact2", "pi05_lerobot", "groot_n17", "g05", "lingbot_v2"}:
        parser.error("--robot-type so101 is implemented only for molmoact2, pi05_lerobot, groot_n17, g05 and lingbot_v2")
    if not 1 <= args.action_steps <= 64 or any("=" not in item for item in args.g05_override):
        parser.error("--action-steps must be in [1, 64] and --g05-override must be KEY=VALUE")
    if args.g05_parts_meta is not None and (args.model != "g05" or args.robot_type != "yam"):
        parser.error("--g05-parts-meta requires g05 --robot-type yam")
    required = {"molmoact2": [], "pi05_lerobot": ["tokenizer", "stats"],
                "groot_n17": ["processor"], "lap_3b": ["tokenizer"], "g05": ["source_root"], "lingbot_v2": ["source_root", "processor"]}
    if args.model == "groot_n17" and args.robot_type == "yam":
        required["groot_n17"] = ["processor", "base_model"]
    if args.model == "g05" and args.robot_type == "yam":
        required["g05"] = ["source_root", "processor"]
    if args.model == "pi05_lerobot" and args.robot_type == "so101":
        required["pi05_lerobot"] = []  # tokenizer and statistics are bundled with the checkpoint
    if args.model == "groot_n17" and args.robot_type == "so101":
        required["groot_n17"] = ["base_model"]  # the VLM processor is bundled with the checkpoint
    if args.model == "pi05_lerobot" and args.robot_type == "yam":
        required["pi05_lerobot"] = ["tokenizer"]
        if args.stats is not None:
            parser.error("pi05 YAM uses saved checkpoint processors; omit --stats (DROID only)")
    for key in ["checkpoint", *required[args.model]]:
        path = getattr(args, key)
        if path is None or not path.exists():
            parser.error(f"--{key.replace('_', '-')} must point to an existing local asset")
    for key in ("checkpoint", "tokenizer", "stats", "processor", "source_root", "base_model", "g05_parts_meta"):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, path.resolve())
    if args.model == "g05" and args.robot_type == "yam":
        from .yam_g05 import validate_assets
        try:
            validate_assets(args.checkpoint, args.source_root, args.processor, args.g05_parts_meta)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    elif args.model == "g05":
        if not args.checkpoint.is_file() or not (args.source_root / "scripts/serve_policy.py").is_file():
            parser.error("G05 needs a checkpoint file and upstream scripts/serve_policy.py")
    elif not args.checkpoint.is_dir():
        parser.error("--checkpoint must be a local directory")
    if args.model == "lap_3b" and not args.tokenizer.is_file():
        parser.error("LAP --tokenizer must be the tokenizer.model file")
    if args.model == "pi05_lerobot" and args.robot_type != "so101" and not args.tokenizer.is_dir():
        parser.error("pi05 needs a local tokenizer directory")
    if args.model == "pi05_lerobot" and args.robot_type == "franka" and not args.stats.is_file():
        parser.error("pi05 needs a tokenizer directory and a statistics JSON file")
    if args.model == "groot_n17" and args.robot_type != "so101" and not args.processor.is_dir():
        parser.error("GR00T --processor must be a local processor directory")
    if args.model == "groot_n17" and args.robot_type in {"yam", "so101"} and not args.base_model.is_dir():
        parser.error("--base-model must be a local directory")
    if args.model == "lingbot_v2":
        if args.robot_type == "so101":
            from .so101_lingbot import validate_assets
        else:
            from .yam_lingbot import validate_assets
        try:
            validate_assets(args.checkpoint, args.source_root, args.processor)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    return args


def _infer(runtime, payload):
    if not isinstance(payload, Mapping):
        raise ValueError("request must be a mapping")
    result = runtime.infer(payload)
    if not isinstance(result, Mapping) or result.get("actions") is None:
        raise ValueError("model returned no actions")
    actions = np.asarray(result["actions"], dtype=np.float32)
    action_dim = getattr(runtime, "action_dim", 8)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != action_dim or not np.isfinite(actions).all():
        raise ValueError(f"model must return a finite N x {action_dim} action array")
    return dict(result, actions=actions)


def http_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value, default=_json_array).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in {"/act", "/healthz"}:
                self.reply(200, {"status": "ok"})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/act":
                self.reply(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid request length")
                self.connection.settimeout(30)
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                for key in (*getattr(runtime, "image_keys", ("external_cam", "wrist_cam")), "state"):
                    payload[key] = _json_decode(payload[key], max_values=MAX_REQUEST_BYTES)
            except Exception:
                log.exception("Invalid model request")
                self.reply(400, {"error": "Invalid model request"})
                return
            try:
                response = _infer(runtime, payload)
            except Exception:
                log.exception("Model inference failed")
                self.reply(500, {"error": "Model inference failed"})
                return
            self.reply(200, response)

        def log_message(self, format, *args):
            log.debug(format, *args)

    return Handler


def websocket_handler(runtime, metadata):
    import msgpack
    lock = asyncio.Lock()

    def pack(value):
        return msgpack.packb(value, default=_msgpack_default)

    async def handler(socket):
        await socket.send(pack(metadata))
        async for message in socket:
            try:
                payload = msgpack.unpackb(message, object_hook=_msgpack_object, raw=False, strict_map_key=False)
                async with lock:
                    response = await asyncio.to_thread(_infer, runtime, payload)
            except Exception:
                log.exception("pi05 inference failed")
                response = {"error": "Model inference failed"}
            await socket.send(pack(response))

    return handler


async def serve_pi05(runtime, host, port, num_steps):
    import websockets
    from .pi05_contract import LEROBOT_CHECKPOINT, LEROBOT_MODEL_SOURCE
    metadata = {"checkpoint": LEROBOT_CHECKPOINT, "model_source": LEROBOT_MODEL_SOURCE,
                "protocol": "openpi-droid-websocket-msgpack", "state_dim": 32, "action_dim": 32,
                "droid_output_dim": 8, "action_horizon": 15, "action_space": "joint_velocity",
                "gripper_action_space": "position", "right_wrist_image": "masked", "num_steps": num_steps}
    async with websockets.serve(websocket_handler(runtime, metadata), host, port,
                                compression=None, max_size=MAX_REQUEST_BYTES):
        await asyncio.Future()


def run_g05(args):
    """Execute the user's installed upstream; no G05 source is vendored."""
    if not args.g05_native_attention:
        from g05.models.g05.qwen35 import vision
        vision._flash_attn_varlen = None
        vision._flash_attn_backend = "sdpa"
    script = args.source_root / "scripts/serve_policy.py"
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.source_root)
        sys.argv = [str(script), "--ckpt_path", str(args.checkpoint), "--host", args.host,
                    "--port", str(args.port), "--device", args.device]
        if getattr(args, "robot_type", "franka") == "so101":
            # The SO100/101 run serves chunks; its upstream embodiment name is so100.
            sys.argv += ["--action_steps", str(args.action_steps), "eval_embodiment=so100", *args.g05_override]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


def main(argv=None):
    args = parse_args(argv)
    # Match the deployed offline loading behavior. No implicit Hub downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logging.basicConfig(level=logging.INFO)
    from . import model_loading
    if args.model == "molmoact2":
        runtime = model_loading.MolmoRuntime(args.checkpoint, device=args.device, num_steps=args.num_steps,
                                            dtype=args.dtype, patch_bf16=args.patch_molmo_bf16, robot_type=args.robot_type)
        # Single-threaded HTTP server serializes calls to the model, like the deployed lock.
        with HTTPServer((args.host, args.port), http_handler(runtime)) as server:
            server.serve_forever()
    elif args.model == "lingbot_v2":
        if args.robot_type == "so101":
            from .so101_lingbot import LingBotSO101Runtime as LingBotRuntime
        else:
            from .yam_lingbot import LingBotYAMRuntime as LingBotRuntime
        runtime = LingBotRuntime(args.checkpoint, source_root=args.source_root,
                                    processor=args.processor, device=args.device, dtype=args.dtype,
                                    use_compile=args.lingbot_compile)
        with HTTPServer((args.host, args.port), http_handler(runtime)) as server:
            server.serve_forever()
    elif args.model == "pi05_lerobot":
        if args.robot_type == "yam":
            from .yam_pi05 import Pi05YAMRuntime
            runtime = Pi05YAMRuntime(args.checkpoint, args.tokenizer,
                                    device=args.device, num_steps=args.num_steps)
            with HTTPServer((args.host, args.port), http_handler(runtime)) as server:
                server.serve_forever()
            return
        if args.robot_type == "so101":
            from .so101_pi05 import Pi05SO101Runtime
            runtime = Pi05SO101Runtime(args.checkpoint, device=args.device, num_steps=args.num_steps)
            with HTTPServer((args.host, args.port), http_handler(runtime)) as server:
                server.serve_forever()
            return
        runtime = model_loading.Pi05Runtime(args.checkpoint, args.tokenizer, args.stats,
                                           device=args.device, num_steps=args.num_steps)
        asyncio.run(serve_pi05(runtime, args.host, args.port, args.num_steps))
    elif args.model == "groot_n17":
        if args.robot_type == "yam":
            from .yam_groot import GrootYAMRuntime
            runtime = GrootYAMRuntime(args.checkpoint, base_model=args.base_model,
                                      processor=args.processor, device=args.device,
                                      camera_order=args.groot_camera_order)
            with HTTPServer((args.host, args.port), http_handler(runtime)) as server:
                server.serve_forever()
            return
        if args.robot_type == "so101":
            from .so101_groot import GrootSO101Runtime
            runtime = GrootSO101Runtime(args.checkpoint, base_model=args.base_model, device=args.device)
            with HTTPServer((args.host, args.port), http_handler(runtime)) as server:
                server.serve_forever()
            return
        policy = model_loading.load_groot(args.checkpoint, device=args.device, processor=args.processor)
        from gr00t.policy.server_client import PolicyServer
        with PolicyServer(policy=policy, host=args.host, port=args.port) as server:
            server.run()
    elif args.model == "lap_3b":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        policy = model_loading.load_lap(args.checkpoint, args.tokenizer)
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()
    elif args.robot_type == "yam":
        from .yam_g05 import G05YAMRuntime
        runtime = G05YAMRuntime(args.checkpoint, source_root=args.source_root, processor=args.processor,
                                parts_meta=args.g05_parts_meta, device=args.device,
                                native_attention=args.g05_native_attention)
        with HTTPServer((args.host, args.port), http_handler(runtime)) as server:
            server.serve_forever()
    else:
        run_g05(args)


if __name__ == "__main__":
    main()
