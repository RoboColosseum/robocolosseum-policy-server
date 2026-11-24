# Inspect CaP-X backend

`inspect_capx` runs Inspect Robots' CaP-X code-as-policy (`capx`) behind the
same Local Protocol as [`inspect_agent`](inspect-agent.md). Each turn, the
LLM writes a short Python program instead of calling tools. The program runs
in a sandbox, where it can call SAM3 segmentation, read the robot state, and
solve inverse kinematics. The backend sends the resulting joint targets to
the Client as normal joint-position action plans: six values on SO-101,
fourteen on bimanual YAM, where the LLM drives both arms (see
[Bimanual YAM](#bimanual-yam)).

SO-101 and bimanual YAM are supported. The Robot Client keeps ownership of
the hardware. The Policy Server receives only joint states, images and
(optionally) aligned depth, so it solves forward and inverse kinematics from a
local URDF.

## Model naming

A model whose name ends in `-capx` is routed to `inspect_capx` by the `auto`
backend. The Router registers three for SO-101: `claude-opus-5-5-capx`,
`gpt-6-astra-capx` and `grok-4.7-capx`. All other LLM names keep using
`inspect_agent`, so a single configuration can serve both, for example
`grok-4.7` (tool-calling agent) and `grok-4.7-capx` (code as policy). The
provider comes from the name prefix (`gpt-`, `grok-`, `claude-`). The model
sent to the provider API is `backend_options.api_model`, and defaults to the
name without `-capx`.

As with every model, `name`, `url`, `revision`, the action contract and
`max_horizon` must match the Router registration. The SO-101 CaP-X entries
use the `joint_position` space with `action_dim: 6`. This holds even when the
Client's `llm_control_mode` is `eef_delta_pose`, because CaP-X solves its own
IK on the Policy Server. The YAM CaP-X entries use `joint_position` with
`action_dim: 14` and `control_hz: 30`, like the YAM joint agents.

## Install

The `capx` extra pins Inspect Robots core, the Agent plugin and the CaP-X
plugin (with its Placo kinematics) to one Git commit of
`supernoooo/inspect-robots` (branch `yam-capx`, which adds the multi-arm
profile YAM needs on top of `so101-capx`). Install it instead of the `agent`
extra:

```bash
uv sync --locked --extra capx --extra droid   # SO-101
uv sync --locked --extra capx                 # YAM
```

uv fetches Inspect Robots from GitHub at the pinned commit; there is no
separate `pip install`.

The tool-calling LLM agents keep working with this extra in joint mode
(`llm_control_mode: joint_pos`). SO-101 EEF agent mode needs the `agent`
extra.

Two more pieces are needed:

1. The SO-101 URDF with its meshes, from TheRobotStudio's SO-ARM100 repository
   at a pinned commit (about 19 MB). The URDF alone is not enough, because it
   references the meshes in `assets/`:

   ```bash
   git clone -q --filter=blob:none --no-checkout https://github.com/TheRobotStudio/SO-ARM100.git /tmp/so-arm100
   git -C /tmp/so-arm100 sparse-checkout set Simulation/SO101
   git -C /tmp/so-arm100 checkout -q 5f6d2b876a53a4872e405b991dd925556c9e38a4
   cp -r /tmp/so-arm100/Simulation/SO101 /path/to/SO101
   sha256sum /path/to/SO101/so101_new_calib.urdf
   # 3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c
   ```

   Set `capx.urdf_path` to `/path/to/SO101/so101_new_calib.urdf`.
2. The SAM3 segmentation server. A copy lives in this repository under
   [`servers/sam3`](../servers/sam3/README.md). It needs a GPU and its own
   environment. Run these commands from the root of this repository (the
   Policy Server checkout, not the Client). Installing SAM3 and downloading its
   weights means accepting Meta's SAM License:

   ```bash
   uv venv servers/sam3/.venv --python 3.10
   uv pip install --python servers/sam3/.venv/bin/python -e servers/sam3
   servers/sam3/.venv/bin/python -m sam3_server --port 8114
   ```

   Run the last command once by hand: the first start downloads `sam3.pt`
   (3.45 GB). Stop it with Ctrl-C after it logs `Uvicorn running`.

With `sam3_launcher` set (see [Start SAM3 with each session](#start-sam3-with-each-session)),
the Policy Server starts SAM3 itself for each CaP-X session. Without it, start
SAM3 yourself before the Policy Server: the backend checks that the SAM3 port
is reachable when a session starts and fails preparation if it is closed.

## Configure

Start from an SO-101 runtime configuration that already serves LLMs through
`inspect_agent` in joint mode. Its `llm` options (`joint_low`, `joint_high`,
`joint_max_step`, `sensors`, `robot_notes`, credentials) are shared with
CaP-X. Add a `capx` mapping under `backend.options` and one model entry per
CaP-X name, registered on the Router with the same values:

```yaml
backend:
  name: auto
  options:
    llm:
      # ... the existing inspect_agent rig options ...
      max_llm_calls: 30   # shared by an LLM and its CaP-X version
    capx:
      urdf_path: /path/to/so101_new_calib.urdf
      sam3_url: http://127.0.0.1:8114
      capx_camera: head_image
      capx_depth: true
models:
- name: grok-4.7-capx
  model_type: llm
  url: https://api.yhlxj.ai/v1
  revision: rig-v1
  action_space: joint_position
  action_dim: 6
  control_hz: 10
  max_horizon: 150
  endpoint: inprocess://inspect-agent
  backend_options:
    robot_type: so101
```

The `capx` mapping accepts these settings:

| Key | Default | Meaning |
|---|---|---|
| `urdf_path` | required | SO-101 URDF with a `gripper_frame_link` |
| `sam3_url` | `http://127.0.0.1:8114` | SAM3 server |
| `sam3_launcher` | none | Command that starts SAM3 for each session; see below |
| `sam3_start_timeout_s` | `600` | Longest wait for a launched SAM3 to listen |
| `capx_camera` | first `sensors` entry | Camera that `segment()` and depth refer to |
| `capx_depth` | `true` | Ask the Client for depth aligned to `capx_camera` |
| `prior_learnings` | none | Optional Markdown notes appended to the prompt |
| `max_llm_calls` | `30` | LLM turns per episode before giving up (failed code counts) |
| `max_speed_frac` | `0.1` | Per-step speed: `0.1` or more moves at the full `joint_max_step`, lower values scale it down |
| `gripper_speed_frac` | `0.5` | Same, for the gripper |
| `image_horizon` | `2` | Number of recent turns whose images stay in the prompt |
| `ik_position_tolerance_m` | `0.005` | Largest accepted IK position residual |
| `ik_orientation_weight` | `0.1` | Starting orientation weight; it is relaxed when a pose is unreachable |

`max_llm_calls` and `max_speed_frac` default to the `inspect_agent` values, and
a value set in the shared `llm` mapping applies to both backends. Set them in
the `capx` mapping only to give CaP-X a different budget on purpose.

Every provider, Claude included, uses the API key that the Client sends
(`llm_api_keys`), through `llm_endpoint` (`router` or `official`). CaP-X does
not accept account logins. The wire format defaults to Responses for OpenAI
and Chat Completions for xAI and Anthropic. Set `capx_wire` to override it.

Start the server:

```bash
.venv/bin/colosseum-policy-local --config configs/local-runtime-so101-capx.yaml
```

## Depth

With `capx_depth: true`, the ready message asks the Client for
`observation_extras: ["depth"]`. An SO-101 Client then enables the depth
stream of `capx_camera` (it must be a RealSense camera), aligns it to the
color image, and sends two extra state tensors with each observation:

- `<camera>_depth_m`: an H x W float32 depth map in metres, at the image size
  sent to the policy;
- `<camera>_intrinsics`: the 3 x 3 color intrinsics, scaled to that size.

Generated code can then lift a segmentation mask to a 3D point in the camera
frame. Depth goes only to the Policy Server. The Client does not record or
upload it. Set `capx_depth: false` if the camera has no depth stream.

## Action plans and stopping

Generated code can return more targets than the Router `max_horizon` allows.
The backend sends the first `max_horizon` targets and keeps the rest for the
next steps, so the LLM is not called again until they have all been sent.
Every target is checked against the joint bounds and per-step limits before
it is sent.

When the policy reports success (FINISH) or gives up (GIVE_UP), the backend
holds the measured joints with the last commanded gripper, as `inspect_agent`
does after a stop request. The session log in `log_dir` records
`policy: "capx"` and the full transcript, including the generated code.

These checks bound each returned target. They are not collision checking or
evidence that a trajectory is safe on a particular rig.

## Bimanual YAM

On YAM, CaP-X drives both arms through the CaP-X plugin's multi-arm profile.
The 14 action dimensions keep the YAM joint contract: left joints 1-6
(radians), left gripper (0 closed, 1 open), right joints 1-6, right gripper.
The generated code names the arm in every arm and gripper helper:

| Helper | Meaning |
|---|---|
| `solve_ik(position, quaternion_wxyz, arm)` | Six radians that put that arm's grasp point at the pose |
| `move_to_joints(joints, arm)` | Move one arm; the other arm and both grippers hold |
| `move_arms(left=..., right=...)` | Move both arms together; they start and finish at the same time |
| `open_gripper(arm)`, `close_gripper(arm)` | Ramp one gripper; everything else holds |

Queued calls run one after another. The state shows `joint_pos` (the 14
values, with the last commanded gripper values), `gripper_measured`, and each
arm's measured tool pose as `<arm>_eef_pos` and `<arm>_eef_quat_wxyz`.

### YAM kinematics

The Policy Server needs i2rt's URDF of one YAM arm, next to its `assets/`
meshes, for example `robot_models/arm/yam/v1/yam.urdf` from an installed
`i2rt` package or checkout. Both arms use the same URDF. At session start the
backend writes a temporary copy with a fixed `grasp_site` link on the
`gripper` mount: the midpoint of the LINEAR_4310 fingertips, 0.14465 m along
the approach axis. This is the tool point of Inspect Robots' native YAM CaP-X
pipeline, so its prompts and rig notes carry over. The tool's local +z is the
approach axis and the jaws open along its local +y. At zero joints the tool
points along the base +x axis, and straight down is close to
`quaternion_wxyz` `[0, 0, 1, 0]`.

IK uses the configured joint limits, intersected with the URDF ranges. It
starts from the arm's queued joints. When that start cannot reach the pose,
or misses the orientation by more than 10 degrees, IK retries from a raised
"ready" pose (`ik_ready_seed`) and keeps the better solution. Position always
wins over orientation. A residual above `ik_position_tolerance_m` raises
inside the generated code.

Each arm reports and solves poses in its own base frame (metres; x forward
out of the base, y left, z up). No calibration relates the two frames, and
the prompt says so. To let both arms share coordinates, measure each base
origin and heading in one rig frame and set `arm_base_poses`. The prompt then
describes that shared frame instead.

### YAM configuration

Start from [`configs/local-runtime-yam-capx.yaml.example`](../configs/local-runtime-yam-capx.yaml.example).
The shared `llm` mapping keeps the YAM rig options (`joint_low`, `joint_high`,
`joint_max_step`, `sensors`, `robot_notes`). The `capx` mapping adds:

| Key | Default | Meaning |
|---|---|---|
| `urdf_path` | required | i2rt YAM arm URDF with its meshes |
| `capx_camera` | first `sensors` entry (`head_image`) | Camera that `segment()` and depth refer to |
| `capx_depth` | `true` | Ask the Client for depth aligned to `capx_camera` |
| `joint_speed_rad_s` | `0.3` | Speed of queued capx joint motion (0.01 rad per 30 Hz step) |
| `gripper_speed_per_s` | `1.0` | Gripper ramp speed in normalized units per second |
| `gripper_max_step` | `0.15` | Largest gripper change per step; match the Client |
| `arm_base_poses` | none | `{left: [x, y, z, yaw_deg], right: [...]}` of both bases in one rig frame |
| `ik_orientation_weight` | `1.0` | Starting orientation weight (six joints reach most orientations) |
| `ik_position_tolerance_m` | `0.005` | Largest accepted IK position residual |
| `ik_ready_seed` | `[0, 1.05, 1.05, -0.52, 0, 0]` | IK retry start, radians |
| `tool_offset_xyz`, `tool_offset_rpy` | LINEAR_4310 grasp point | Tool frame relative to the mount link |
| `tool_mount_frame` | `gripper` | URDF link the tool frame is fixed to |

On YAM, `max_speed_frac` and `gripper_speed_frac` do not apply to CaP-X: the
shared `llm` value tunes the tool-calling agent, where `0.1` allows the rig's
full `joint_max_step` (0.15 rad, or 4.5 rad/s at 30 Hz). CaP-X uses
`joint_speed_rad_s` and `gripper_speed_per_s` instead. Every emitted step is
still checked against `joint_max_step` and `gripper_max_step`.

When a gripper closes on an object, the measured opening stays wider than the
command. CaP-X plans from the last commanded gripper values, so a later arm
move does not loosen the grasp, and a stop holds the measured joints with
those commanded grippers. `gripper_measured` shows the real opening, so code
can check the grasp.

The YAM Client streams aligned depth from the RealSense camera named by
`adapter_config.depth_camera` (default `head_image`), which must match
`capx_camera`. The two arms are not collision-checked against each other or
the table. Keep the arms apart in `robot_notes`, and supervise every trial.

### Run on YAM

The YAM URDF and meshes ship with the I2RT package that the YAM Client
installs. With the Client checked out next to this repository, set
`urdf_path` to
`/home/<user>/colosseum-client/.venv/lib/python3.12/site-packages/i2rt/robot_models/arm/yam/v1/yam.urdf`.

```bash
uv sync --locked --extra capx
cp configs/local-runtime-yam-capx.yaml.example configs/local-runtime-yam-capx.yaml
# fill in the joint bounds, robot_notes and urdf_path
uv run --locked --extra capx colosseum-policy-local --config configs/local-runtime-yam-capx.yaml
```

The example sets `sam3_launcher`, so SAM3 starts with each CaP-X session and
needs no terminal of its own.

To serve the CaP-X models next to your local VLAs and LLM agents, add the
`capx` mapping and the three `-capx` model entries to your existing YAM config
(for example `local/local-runtime-yam-agent-joint.yaml`) instead. The Client
needs no new install; it enables depth when the ready message asks for it.

## Start SAM3 with each session

Set `capx.sam3_launcher` to the command that starts SAM3:

```yaml
capx:
  sam3_url: http://127.0.0.1:8114
  sam3_launcher: [servers/sam3/.venv/bin/python, -m, sam3_server, --port, "8114"]
```

When a CaP-X session starts and nothing listens at `sam3_url`, the backend
runs this command and waits until SAM3 listens (up to `sam3_start_timeout_s`;
loading the cached weights takes well under a minute). When the session ends,
it stops SAM3, so the GPU is free for a VLA that runs next. Like a VLA worker,
SAM3 also stops if the Policy Server exits or is killed. Its output goes to
`sam3-<time>.log` in the agent `log_dir`.

If SAM3 is already listening, for example because you started it by hand, the
backend uses it and leaves it running. A relative program path is resolved
from the directory the Policy Server starts in, normally the repository root.
If the command has `--port`, it must match the `sam3_url` port.
