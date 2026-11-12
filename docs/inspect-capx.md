# Inspect CaP-X backend

`inspect_capx` runs Inspect Robots' CaP-X code-as-policy (`capx`) behind the
same Local Protocol as [`inspect_agent`](inspect-agent.md). Each turn, the
LLM writes a short Python program instead of calling tools. The program runs
in a sandbox, where it can call SAM3 segmentation, read the robot state, and
solve SO-101 inverse kinematics. The backend sends the resulting joint
targets to the Client as normal six-value joint-position action plans.

Only SO-101 is supported. The Robot Client keeps ownership of the hardware.
The Policy Server receives only joint states, images and (optionally) aligned
depth, so it solves forward and inverse kinematics from a local URDF.

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
`max_horizon` must match the Router registration. The CaP-X entries use the
`joint_position` space with `action_dim: 6`. This holds even when the Client's
`llm_control_mode` is `eef_delta_pose`, because CaP-X solves its own IK on the
Policy Server.

## Install

The `capx` extra pins Inspect Robots core, the Agent plugin and the CaP-X
plugin (with its Placo kinematics) to one Git commit of
`supernoooo/inspect-robots` (branch `so101-capx`). Install it instead of the
`agent` extra:

```bash
uv sync --locked --extra capx --extra droid
```

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
   environment:

   ```bash
   uv venv servers/sam3/.venv --python 3.10
   uv pip install --python servers/sam3/.venv/bin/python -e servers/sam3
   servers/sam3/.venv/bin/python -m sam3_server --port 8114
   ```

The backend checks that the SAM3 port is reachable when a session starts. It
fails preparation if the port is closed.

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
