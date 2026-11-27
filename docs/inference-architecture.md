# Robot operation and inference

Hardware belongs to the Robot Client. Its `robot_type: franka` selects the
DROID hardware driver; YAM, SO101 and other robots use their own Client drivers.
The Policy Server never constructs those drivers.

The Policy Server defaults to `auto` inference when `backend.name` is absent:

- LLM: Inspect handles the prompt, conversation, provider API and tools. An
  inference-side robot contract supplies the embodiment name, joint count and
  action dimensions; configured rig bounds and gripper polarity remain required.
- VLA: `VLABackend` handles the session and selects `backend_options.adapter`.
  The model adapter owns observation encoding, service communication and action
  decoding. This is model/checkpoint-specific, not inferred from robot type.

Client adds `robot_type` to its direct Protobuf preparation. Adapter support and
model action dimensions/semantics are checked before reporting ready. An optional
`backend_options.robot_type` pins a model to a robot; it must match the Client.
`droid` is normalized to `franka`. Older Clients without the field retain the
legacy Franka default (or an explicitly configured robot).

Built-in VLA adapters: `molmoact2`, `pi05_lerobot`, `groot_n17`, `g05`, `lap_3b`
support Franka only. `molmoact2_yam`, `groot_yam`, `pi05_yam`, `g05_yam`, `lingbot_yam`
and `isaac_yam` support [YAM](yam.md); `molmoact2_so101`, `pi05_so101`, `groot_so101`,
`g05_so101`, `lingbot_so101` and `isaac_so101` support [SO101](so101.md). Their existing payloads, transports, and action
conversions remain unchanged. Inspect currently supports the Franka joint-position
contract only. Unsupported combinations fail preparation rather than borrowing
Franka semantics. A synthetic YAM plugin test verifies extensibility, not hardware support.

## Daily commands

Install Policy dependencies once with `uv sync --extra droid --extra agent`.
Copy `configs/local-runtime.yaml.example` to `configs/local-runtime.yaml`, fill
in the rig parameters and actual model catalog, then run from the server repo:

```bash
.venv/bin/colosseum-policy-local
```

After installing the Client's hardware dependencies and setting its config/API
key environment variables, run from the Client repo:

```bash
.venv/bin/colosseum-robot configs/robot.yaml
```

Add `--no-execute-action` to the Client for inference without executing returned
actions. Both VLA and LLM catalog entries must match Router assignments. VLA
weights/services still require setup; auto dispatch does not provision models.
Client and Policy Server need the updated Protobuf code; the new robot identity
field is Client-to-Policy-only and does not require a Router restart.

## Adding a VLA adapter

Register a factory under the `colosseum_policy_server.model_adapters` Python
entry-point group. The factory receives runtime options and returns an object
with:

- `supported_robot_types`: canonical robot names supported by this implementation.
- `validate(model)`: reject incompatible action contracts/checkpoint options.
- `async infer(model, observation)`: return finite `(horizon, action_dim)` actions.
- Optional `async start_session(model, request)` and `async end_session()` hooks.

Select it with `backend_options.adapter` in the registered model. Model-specific
settings are available through `model.backend_options`. Session cleanup runs after
partial startup and errors. No credentials should be logged by plugins.

Existing `backend.name: droid`, `--robot droid`, and `DroidBackend` imports are
compatibility aliases for the Franka VLA adapter. New mixed setups should omit
those overrides. The `droid` dependency extra also retains its existing name;
it installs inference transport dependencies, not robot control on the server.
