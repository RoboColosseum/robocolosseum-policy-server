# Remote Inference Guide

Run your policy on your own GPU server and evaluate it on RoboColosseum robots.

## 1. How it works

Your model stays on your server (any cloud, any region). Your server makes an **outbound**
WebSocket connection to the RoboColosseum Router, so it needs no public IP address, inbound
port or VPN. When an evaluator starts a trial on a robot assigned to your policy, the Router
pairs the robot with your server and relays observations and action chunks between them.

```
Your GPU server ── outbound wss ──▶  RoboColosseum Router  ◀── wss ── Robot station
   (Policy SDK)                    (auth, pairing, relay)            (cameras, arms, operator)
```

Your model weights and code never leave your server. The robot receives only the action
chunks you return.

## 2. What you receive from us

| Item | Example | Notes |
|-|-|-|
| Router URL | `wss://<router-host>` | Provided at onboarding |
| Policy Server token | `pol_…` | Secret. Treat it like a password |
| Policy ID | `acme-yam-v1` | Names your deployment. Your server announces it when it connects |
| Robot and task | `yam`, task `pick-cup-v1` | Determines the observation/action contract (Section 6) |
| Test deployment | `acme-yam-v1-test` | Synthetic robot, not ranked. Use it before real-robot trials |

## 3. Install

```bash
git clone https://github.com/frodobots-org/colosseum-policy-server
cd colosseum-policy-server
uv sync --extra demo   # adds JPEG/PNG decoding and the example image viewer
```

## 4. Option A: serve a supported model without writing code

If your model already runs under `colosseum-policy-local` (G05, pi05, GR00T, LingBot,
MolmoAct2; see [local runtime](local-runtime.md)), the remote runner serves the same runtime
YAML to the Router:

```bash
export COLOSSEUM_POLICY_TOKEN=pol_xxxxxxxxxxxxxxxx
uv run colosseum-policy-remote --config configs/local-runtime.yaml --model g05-yam \
  --router wss://<router-host> --policy-id acme-yam-v1
```

The runner:

- loads the model **before** it registers, so no trial is matched to a cold server;
- keeps the model loaded across trials and resets its per-trial state at each new trial;
- reconnects after a Router restart or network drop;
- ends a trial immediately with `INFERENCE_FAILED` if inference raises, instead of letting
  the robot time out;
- prints one JSON line per chunk (`infer_s` is the model time).

Options: `--token-env` (default `COLOSSEUM_POLICY_TOKEN`), `--ping-interval` /
`--ping-timeout` (seconds, default 20 / 60). VLA models only; LLM agents need per-trial
credentials and use the local protocol.

## 5. Option B: connect your own model with the SDK

### 5.1 Check the connection

`configs/policy.yaml`:

```yaml
url: wss://<router-host>
token: pol_xxxxxxxxxxxxxxxx
policy_id: acme-yam-v1   # your deployment; when omitted, the Router uses the token name
robot: yam               # yam | so101 | franka | earth_rover_mini_plus
reconnect: true
# max_horizon: 32        # explicit keys override the robot preset
# ping_interval: 20      # keepalive, seconds; raise ping_timeout on a slow uplink
# ping_timeout: 60
```

`robot` fills `action_spaces`, `control_hz` and `max_horizon` for that robot (Section 6).

```bash
uv run --extra demo python examples/test_policy.py --no-enable-action
```

The script connects and waits. When a trial starts, it prints the task instruction, state and
image shapes.

### 5.2 Serve your model

```python
from colosseum_policy_server import ColosseumPolicySDK, StaleObservationError
from my_policy import load_model                  # your code

model = load_model()                              # load once, before connecting

with ColosseumPolicySDK.from_yaml("configs/policy.yaml") as sdk:
    print("registered; waiting for trials", flush=True)
    while True:
        obs = sdk.get_obs()                       # blocks until the robot sends an observation
        chunk = model.infer(                      # -> np.ndarray (horizon, 14) for YAM
            instruction=obs.instruction,
            joints=obs.state.joints,              # (12,)
            gripper=obs.state.gripper,            # (2,)
            head=obs.state.head_image, left=obs.state.left_image, right=obs.state.right_image)
        try:
            sdk.send_action(chunk, observation=obs)
        except StaleObservationError:
            pass                                  # the trial ended during inference
```

With `reconnect: true`, `get_obs()` keeps waiting across Router restarts and network drops;
a trial that was running when the connection dropped fails on the robot side. Without it, a
drop raises `SDKConnectionError` from `get_obs()`.

Images arrive as `RAW_RGB`, `JPEG` or `PNG`; `obs.state.*_image` and `ImageFrame.as_rgb()`
return RGB uint8 arrays for all three (JPEG/PNG need the `demo` extra).

To end a trial with an explicit error instead of letting it time out:
`sdk.send_error("MY_CODE", "message", observation=obs)`.

Keep the process running whenever trials are scheduled. If your server is not connected when
an evaluator presses Start, the trial fails immediately with `NO_POLICY_AVAILABLE`.

## 6. Robot contracts

### 6.1 YAM (bimanual, `robot: yam`)

| | |
|-|-|
| Control rate | **30 Hz** |
| Action space | `joint_position`, **14** values per step: `[left j1…j6 (rad), left gripper (0–1), right j1…j6 (rad), right gripper (0–1)]` |
| Chunk length | 1 … `max_horizon` steps (preset 50); 32 steps ≈ 1.07 s |
| State | `joint_position` float32 (12): left j1…j6, right j1…j6 (rad). `gripper_position` float32 (2): left, right (0 closed … 1 open). `cartesian_position` |
| Cameras | `head_image`, `left_image`, `right_image`, 640×480 RGB |
| Instruction | Natural-language task text in `obs.instruction` |

Non-finite values and grippers outside [0, 1] are rejected. The station limits joint motion
per step, so large jumps are rejected or clipped.

### 6.2 Earth Rover Mini+ (`robot: earth_rover_mini_plus`)

`base_velocity`: 2 values `[linear, angular]` in [−1, 1] (+angular turns left) at 10 Hz.
Cameras: `front_image` (always), `rear_image` (when available). State: `navigation_state`
float64 `[lat, lon, heading_deg, speed, battery, rpm1…rpm4]`; `goal_gps` `[lat, lon]` for
GPS goals, or a `goal_image` for photo goals. When the goal is reached, send an all-zero
chunk with `done=True`. The rover stops if no new chunk arrives within 1 s.

### 6.3 SO-101 and Franka

SO-101 (`robot: so101`): 5 joints + 1 gripper, `joint_position` 6-D, 30 Hz. Franka/DROID
(`robot: franka`): 7 joints + 1 gripper, `joint_position` 8-D at 15 Hz (`joint_velocity`
8-D and `cartesian_position` 7-D also exist). Confirm these with us before your first
real-robot trial.

## 7. Latency

A trial runs one chunk at a time: the robot executes the chunk, sends a new observation and
**waits** for your reply. The robot is idle for the whole round trip:

> round trip = observation upload + your inference time + chunk download

Measured on our Tokyo Router with a YAM station profile (3 cameras):

| Station image encoding | Upload per chunk | Network part of the round trip |
|-|-|-|
| RAW_RGB | 2.77 MB | ≈ 2.9 s |
| JPEG q85 | 0.13 MB | ≈ 0.7 s |

End to end with G05 on an NVIDIA A10G and JPEG: about 4 s between chunks (1.07 s execution,
2.27 s inference, about 0.7 s network).

Requirements:

1. **Inference per chunk shorter than the chunk duration.** For YAM, a 32-step chunk lasts
   1.07 s. Slower models still run, but the arm pauses between chunks.
2. **Host near the Router** (AWS ap-northeast-1, Tokyo, today). Every 100 ms of extra RTT is
   paid on every chunk.
3. **Warm up before you connect.** The first chunk of a trial must arrive within the station's
   deadline (30 s by default).
4. **Return the longest chunks your model supports** (up to `max_horizon`) to reduce round trips.

## 8. Protocol reference (other languages)

The schema is [`proto/colosseum.proto`](../proto/colosseum.proto).

- **Transport:** WebSocket over TLS, binary frames only, header
  `Authorization: Bearer <policy token>`, frames up to 32 MiB. Answer WebSocket pings.
- **Envelope:** each frame is a `RelayFrame{protocol_version=1, type, session_id, sequence,
  sent_at_ns, deadline_ms, payload}`; `payload` is the serialized message for `type`.

| Step | Direction | `type` | Payload | Notes |
|-|-|-|-|-|
| 1 | you → Router | `HELLO` | `Hello{role=POLICY_SERVER, policy_id}` | First frame. An invalid token closes the socket with code 4401 |
| 2 | Router → you | `REGISTERED` | `Registered{peer_id}` | An empty `policy_id` falls back to the token name |
| 3 | Router → you | `MATCHED` | `MatchResult{session_id, policy_id, robot: RobotSpec}` | A trial starts; `robot` has the type, joint count, control rate and action spaces |
| 4 | you → Router | `SESSION_READY` | `SessionReady{policy_id, policy_revision, action_spaces, control_hz, max_horizon}` | Same `session_id` |
| 5 | Router → you | `OBSERVATION` | `Observation{robot_time_ns, control_step, instruction, state, sensors}` | `sequence` starts at 1; `deadline_ms` is the station's budget for your reply |
| 6 | you → Router | `ACTION_PLAN` | `ActionPlan{request_sequence, plan_id, start_step, valid_until_step, control_hz, actions, done}` | Echo the observation `sequence` in `RelayFrame.sequence` **and** `request_sequence`. `start_step` must equal `control_step`; `valid_until_step ≥ start_step + horizon − 1` |
| – | Router → you | `RESET` | `ResetRequest{reason}` | Episode boundary: clear history state |
| – | Router → you | `SESSION_CLOSE` | – | Trial over. Stay connected for the next `MATCHED` |
| – | you → Router | `ERROR` | `Error{code, message}` | Ends the trial with your error |

- **Tensors:** `Tensor{shape, dtype, data}`, little-endian, row-major; actions are `FLOAT32`.
- **Images:** `Image{sensor_id, encoding, width, height, data}`; `encoding` is `RAW_RGB`
  (H×W×3 uint8), `JPEG` or `PNG`. Stations choose the encoding, so handle all three.
- **One request at a time:** the next observation comes only after the whole chunk ran.

## 9. Onboarding

1. We create your Policy token and deployments and send you the Router URL and token through
   a secure channel.
2. Connect to your **test deployment** (Section 5.1). Its synthetic robot sends observations
   with the real robot's contract (state shapes, action space, control rate).
3. Run your model against the test deployment; send us the per-chunk inference times.
4. We schedule real-robot trials. Keep your server connected during the window.
5. Results (success, partial success, videos) appear in the RoboColosseum dashboard.
   Test-deployment runs are not ranked.

## 10. Troubleshooting

| Symptom | Cause |
|-|-|
| Socket closed with code 4401 right after connecting | Wrong or revoked token |
| Connected, but trials never reach you | `policy_id` is not your deployment's ID (when omitted, the token name is used) |
| No `MATCHED` | No trial assigned to your policy now, or the evaluator started before you connected |
| `inference deadline exceeded` on the station | Inference plus network exceeded the station deadline |
| `StaleObservationError` | The trial ended while you were inferring; skip and wait for the next observation |
| `action horizon … exceeds max_horizon` / `requires action_dim 14` | Chunk shape does not match Section 6 |
| Drops during large uploads | Slow uplink: raise `ping_timeout` (Section 5.1) |

## 11. Security

- The token authenticates your server to the Router and is never forwarded to the robot.
- Traffic is TLS; your server opens no inbound ports.
- Only deployments bound to your token are matched to your server.
- To rotate a token, ask us; the old token is revoked.
