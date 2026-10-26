# Inspect Agent backend

`inspect_agent` runs Inspect Robots' LLM policy behind the existing binary
Protobuf Local Protocol. DROID remains on the Robot Client. No robot SDK,
local GPU, model weights, or model subprocess is needed on the Policy Server.
Images, state, and instructions go to the selected cloud API and incur
provider charges. Franka and YAM retain joint-position control. SO101 may use
seven-value EEF deltas internally while preserving its six-value public
joint-position contract; see the [SO101 guide](so101.md).

## Install and start

From the Policy Server checkout:

```bash
uv sync --locked --extra agent --extra droid
cp configs/local-runtime-agent.yaml.example configs/local-runtime.yaml
```

The optional extra pins Inspect core and Agent to the same Git commit,
`22a4da62776575723c0d6285315fc32cdc0cddf5`, which includes the SO101 EEF
agent semantics and Claude Code login. Both `uv` and `pip install -e '.[agent]'`
use the pinned source. Existing installations without this extra do not import
Inspect or install its HTTP dependencies.

Fill in `joint_low`, `joint_high`, `joint_max_step` (seven numbers each),
`gripper_open_value`, and meaningful `robot_notes` for the actual rig. The
example intentionally refuses preparation until bounds and polarity are set.
These limits validate each returned target; they are not collision checking
or evidence that a particular real-world trajectory is safe.

Start the server after configuring the rig:

```bash
.venv/bin/colosseum-policy-local
```

With no `backend.name`, the runtime automatically dispatches each assigned
model: LLMs use Inspect and VLAs use the VLA runtime with a registered model adapter. The default configuration path is
`configs/local-runtime.yaml`; `--config` can select another file. Install the
extras once as above; the direct executable keeps those installed dependencies.

Client sends its `robot_type` directly in preparation. Inference adapters check
robot compatibility; the Client alone owns hardware drivers. Existing built-in
model adapters and the LLM robot contract support Franka. Other robots require
implemented adapters; they never silently fall back to Franka.

Flat `backend.options` are shared; optional `llm` and `vla` mappings override
them for each type. Existing explicitly named backends and `--robot` overrides
remain supported. Leave `--robot` unset for automatic dispatch. To serve a
mixed evaluation, include both VLA and LLM entries in `models`, retaining each
VLA's adapter, endpoint and launcher. Automatic dispatch does not download or
configure unregistered models. Router assignments must match the catalog.

## Unified model descriptor

The public descriptor uses a single model name, not separate provider and
model-ID fields:

```yaml
model_type: llm
name: grok-4.7
url: https://api.x.ai/v1
revision: "4.7"
action_space: joint_position
action_dim: 8
control_hz: 15
max_horizon: 150
```

`name` is the exact API model name. `url` is the approved API base URL. An
internal adapter table selects Responses, Chat Completions, or Messages.
`revision` is a catalog version label (Grok uses `"4.7"`), not an API model selector.
It is still matched between Router and Policy Server and retained in evaluation
records; it does not pin the provider's weights.
Provider aliases may change; use snapshot names when providers expose them.

For `model_type: vla`, `url` retains its Hugging Face repository meaning and
`revision` must remain a 40-character commit. Omitting `model_type` preserves
legacy VLA behavior. Existing empty/default VLA descriptors retain the old
Protobuf dictionary. `model_type` and `name` are additive optional fields.
The local-only `endpoint: inprocess://inspect-agent` selects the agent runtime;
it is not sent by Router. VLA local endpoints continue to select model services.

## Client-owned API keys (BYOK)

No `llm_access_token` is used. Configure the Robot Client:

```yaml
policy_server_url: ws://127.0.0.1:8000  # or a trusted WSS endpoint
llm_api_keys:
  openai: env:OPENAI_API_KEY
  xai: env:XAI_API_KEY
  anthropic: env:ANTHROPIC_API_KEY
```

Direct key strings are also accepted. `env:VARIABLE` keeps keys outside YAML.
Unset/empty variables are omitted. Clients with no configured keys retain
VLA-only assignments. A Client with only an xAI key can receive Grok or VLA,
not GPT or Claude. All three keys allow all three providers, subject to the
registered policy's enabled state and normal pairing rules.

Router receives only `llm_api_urls` in the assignment request, never keys.
This is a capability declaration, not proof that a key is valid or an access
permission. An invalid key fails at the provider, billed to no other account.
Currently only local inference mode supports this flow; the outbound remote
Policy SDK route has no BYOK credential delivery.

Client sends only the assigned model's key in the direct Policy Server
`prepare.api_key` field. The key is not echoed in `ready`, stored in Router,
written in transcripts, or used for another session. Typed LLM sessions refuse
to fall back to the Policy Server host's own keys. The per-agent HTTP client
holds the key in memory until cleanup; Python does not guarantee secure memory
erasure. Only send keys to a trusted Policy Server.

Credential delivery requires `wss://` or literal loopback `ws://` (including
an SSH tunnel). The runtime listener itself stays on loopback; use the existing
SSH tunnel or a TLS proxy for a remote Robot Client.

| Name | API base | Wire |
|---|---|---|
| `gpt-6-astra` | `https://api.openai.com/v1` | Responses |
| `grok-4.7` | `https://api.x.ai/v1` | Chat Completions |
| `claude-opus-5-5` | `https://api.anthropic.com/v1` | Messages |

`effort: low` is the default. Do not disable thinking for Opus 5.5. `ready`
means the configured API client initialized, not that cloud weights were
loaded or that live API access has been verified. Account access and available
credit are checked by the provider on inference.

## Observation and action contract

`prepare` begins an isolated conversation. `start_session(model, request)` and
`end_session()` are optional async backend hooks; existing stateless VLA
backends continue to implement only `infer()`. The runtime holds its existing
exclusive lock through startup, inference, and cleanup.

The first observation binds the agent's tools to the configured rig and
resets its scene. Subsequent observations retain the conversation. Inputs:

- RAW_RGB sensors `head_image` and `left_image` by default; names configurable.
- `joint_position`: seven finite radians inside configured bounds.
- `gripper_position`: one normalized value in [0, 1].
- Nonempty `instruction` and strictly increasing `control_step` within a session.

The agent sees an eight-element `joint_pos` field and tools named `move_joints`,
`done`, and `give_up`. Inspect uses 1=open internally. The backend explicitly
converts RobotEnv polarity on input and output, and binarizes the returned
gripper to match the existing DROID Client.

The agent generates interpolated joint targets. The backend validates shape,
finite values, bounds, frequency, and per-step joint changes before emitting
an ordinary `ACTION_PLAN`. It returns at most `max_horizon` rows. Longer
motions are truncated to a prefix and reconsidered using the next measured
observation; `control_step` is the executed robot step, not the inference count.
This backend does not use Inspect's rollout/approver loop, so its own bounds
and step validation are mandatory. Actual execution and pacing stay in the
Robot Client.

Cloud calls can exceed the Client's default 30-second `deadline_ms`; adjust
that deadline according to measured latency. Increasing it does not make
LLM inference run at the 15 Hz robot control rate. On timeout the runtime sends
`INFERENCE_TIMEOUT`, discards the late result, and drains the synchronous HTTP
worker before admitting another session. Cleanup can wait for the provider
HTTP timeout (up to minutes); it never lets a late worker mutate a new trial.

## Stop requests and records

`done` and `give_up` latch a fixed hold target at the measured joint positions.
The gripper keeps its last executed binary command; if no command has been
executed, its measured position is thresholded into a binary command. The
backend returns this same target as a one-step `ACTION_PLAN` on each subsequent
observation, without further LLM calls. Measured absolute joint/gripper bounds
still apply. Holding does not reject target-minus-measured drift or move the
latched target with each observation. Normal agent-generated actions retain
their per-step joint limits. The Client's own action validation still applies.

The Client continues until the operator presses Enter or the task reaches
`max_steps`, then follows its normal finish and manual scoring flow. An agent's
self-report never marks success. A new session clears the hold state.

Each session writes a randomly named JSON record under `log_dir` with run ID,
public model identity, stop reason, stop step, stop detail, hold target, and
Inspect metadata. Inspect also writes
its image-free transcript and reported token usage there. Raw provider capture
is disabled. Records are local; they are not uploaded into Router eval logs.
Log files may still contain task and scene information.

## Validation

```bash
uv sync --locked --extra dev --extra agent --extra droid --extra demo
uv run --no-sync pytest -q
```

Agent tests use actual Inspect provider clients with `httpx.MockTransport`;
no API key with real privileges, cloud billing, robot, or downloaded model is
involved. They cover all three wires, Protobuf framing, conversation isolation,
stop requests, timeout cleanup, sensor/state conversion, and motion limits.
Passing these tests does not establish live API access or physical performance.
