# MolmoAct2-BimanualYAM

For Dream-1 cloud inference on YAM, see [Dream-1 YAM](dreamlabs-yam.md).

The `molmoact2_yam` VLA adapter connects Colosseum's local protocol to the
[upstream YAM HTTP /act protocol](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam/host_server_yam.py).
It requires `robot_type: yam`, 14-D absolute `joint_position` at 30 Hz, three
RGB cameras, twelve joint positions and two normalized grippers. It does not
implement pi0.5 YAM. For LLM control, see the Inspect Agent section below.

## Model worker on the GPU host

Use a separate model environment with the dependencies from the
[official MolmoAct2 repository](https://github.com/allenai/molmoact2/tree/6070080a20321b4f498ab30f28e1d09ac465edb7).
Install this `colosseum-policy-server` package into that environment too.
Download a pinned `allenai/MolmoAct2-BimanualYAM` snapshot, including its remote
model Python files, processor/tokenizer assets and `norm_stats.json`. Point the
worker at that **local snapshot directory**. It loads offline, with
`trust_remote_code=True`; it does not download weights for you.

```bash
colosseum-policy-model molmoact2 --robot-type yam \
  --checkpoint /absolute/path/to/MolmoAct2-BimanualYAM-snapshot \
  --dtype float32 --port 8202
```

For bfloat16, use `--dtype bfloat16 --patch-molmo-bf16`. The patch backs up and
modifies only the selected snapshot's model source, and fails if it cannot
match the known upstream source patterns. Newer checkpoints may need an updated
patch; float32 avoids that patch requirement. Actual GPU/model loading has not
been validated by this change.

YAM inference uses `norm_tag=yam_dual_molmoact2`, images `[top, left, right]`,
continuous actions, no depth reasoning and no CUDA graph, following the
upstream example. Do not select `--robot-type yam` with DROID weights.

Alternatively, run the official `examples/yam/host_server_yam.py` service and
point the adapter at it. The default upstream script resolves a Hub snapshot
without pinning a revision, so make sure the worker's actual weights match the
revision registered on your Router. Our local worker makes that snapshot
selection explicit.

## Colosseum local server

From the `colosseum-policy-server` checkout:

```bash
uv sync
cp configs/local-runtime-yam.yaml.example configs/local-runtime-yam.yaml
# Edit model revision/contract to match your Router and actual snapshot.
uv run colosseum-policy-local --config configs/local-runtime-yam.yaml
```

The example uses the built-in `vla` backend; do not pass `--robot yam` (that
legacy option selects a backend entry point, not this model adapter).
A worker can be started separately, or via the optional absolute-path
`launcher` in the example configuration. The model service is loopback HTTP;
the Colosseum service is loopback WebSocket on port 8000. Forward port 8000 to
the robot host if needed.

The adapter converts the Client's split vectors into
`[left six joints, left gripper, right six joints, right gripper]` and sends
`top_cam`, `left_cam`, `right_cam`, `state`, `instruction`, `num_steps` using
json_numpy-compatible encoding. Actions stay in that same 14-value order.
Malformed dimensions, missing/duplicate cameras, nonfinite values, grippers
outside [0,1], and chunks longer than the configured horizon are rejected.

Tests cover mock-model loading, norm tag and camera ordering, real loopback HTTP
serialization, and local VLA routing. They do not prove GPU inference, measured
latency, motor operation or task success. See the
[Client YAM guide](../../colosseum-client/docs/yam.md) for hardware configuration
and shutdown behavior.

## Inspect Agent: Grok, Opus, GPT and Gemini

The Colosseum Inspect Agent adapter supports the same YAM Client with
`grok-4.7`, `claude-opus-5-5`, `gpt-6-astra` and `gemini-4-hydrogen` model
identifiers. Grok, Opus and GPT reuse the existing Chat Completions, Anthropic
Messages and Responses transports. The example configs call Gemini through the
Antigravity CLI (`agy`), which must be installed on this host; see
[Gemini through the Antigravity CLI](inspect-agent.md#gemini-through-the-antigravity-cli).
Gemini is policyId `gemini-4-hydrogen-yam` on the Router, which
sends the model name `gemini-4-hydrogen`; see
[inspect-agent.md](inspect-agent.md#client-owned-api-keys-byok). This is adapter support; model availability, credentials and
relay compatibility must be checked against your actual provider.

```bash
uv sync --extra agent
cp configs/local-runtime-yam-agent.yaml.example configs/local-runtime-yam-agent.yaml
# Fill the twelve-joint limits, rig notes, and registered model identities.
uv run --no-sync colosseum-policy-local --config configs/local-runtime-yam-agent.yaml
```

Each entry uses 14-D `joint_position` at 30 Hz. `max_horizon: 150` caps a
returned trajectory at five seconds of nominal control time, not the LLM's
request frequency. Router YAM deployments must use the same contract; existing
8-D Franka model deployments cannot be reused as-is. This change does not
register or deploy Router entries.

The Client supplies API keys via its existing `llm_api_keys` configuration.
No model worker/GPU or `inspect-robots-yam` hardware plugin is needed for this
LLM path: the Colosseum Client owns I2RT and cameras. The Policy Server binds
Inspect's agent to the YAM action/state description.

Joint labels are `left_joint1` through `left_joint6`, then `left_gripper`,
`right_joint1` through `right_joint6`, then `right_gripper`. Both grippers are
continuous [0,1], 1=open. Bounds and per-step limits are packed in this order;
settings arrays remain twelve joint values, left then right. YAM requires
`gripper_open_value: 1` and all three camera roles.

When the agent requests done/give_up, measured joints form a fixed hold target
and each gripper retains its last executed command (or its measured opening
if no command has executed). Subsequent hold requests make no more LLM calls.
Operator scoring is still required; an agent stop does not establish success.
This powered hold during a trial differs from Client teardown, which releases
torque through I2RT close.

Offline tests exercise all three provider wire formats with the actual pinned
Inspect agent and mocked HTTP replies, including Protobuf action plans, both
arm indices, continuous grippers and stop handling. No live provider call or
physical YAM execution was performed.

## MolmoAct2 vs GR00T YAM deployment

The additional `groot_yam` adapter uses the same three-camera `/act` wire format.
The registered checkpoint is **LeRobot format**, not an Isaac-GR00T native
checkpoint: `NUSMAGIC/GR00T-N1.7-MolmoAct2-YAM` at
`5ba9c403bdb962b88c39f843da536eb8370b61b2`. Its saved configuration declares
`new_embodiment`, absolute 14-D actions, 16-step chunks and
`observation.images.top/left/right`. The worker uses the checkpoint's saved
pre/postprocessors, including normalization and action unnormalization.
Do not load this snapshot through the Franka GR00T native worker.

Use separate GPU environments for MolmoAct2 and GR00T. Keep the lightweight
local Policy Server in its own environment. For the LeRobot GR00T environment,
the upstream source inspected for this integration is
`huggingface/lerobot@8c920c4270460851cedd2737657584586d3dc66f`:

```bash
# From the Policy Server repository; native/CUDA dependencies may need rig-specific setup.
uv venv --python 3.12 .venv-groot-yam
uv pip install --python .venv-groot-yam/bin/python \
  'lerobot[groot] @ git+https://github.com/huggingface/lerobot.git@8c920c4270460851cedd2737657584586d3dc66f'
uv pip install --python .venv-groot-yam/bin/python -e .
```

Provision local snapshots before starting the offline worker:

- YAM fine-tune above, including `config.json`, `model.safetensors`, both
  processor JSON files and both processor state safetensors.
- `nvidia/GR00T-N1.7-3B` base model with its required local/cached dependencies.
- `Qwen/Qwen3-VL-2B-Instruct` tokenizer/processor assets matching the checkpoint.

The worker disables Hub downloads; missing base/backbone assets fail at startup.
GPU installation, memory requirements and live checkpoint loading have **not**
been validated here. The loader/processor calls are covered with mocked models.

```bash
# In its GPU environment, with actual local asset paths:
.venv-groot-yam/bin/colosseum-policy-model groot_n17 --robot-type yam \
  --checkpoint /absolute/path/to/GR00T-N1.7-MolmoAct2-YAM \
  --base-model /absolute/path/to/GR00T-N1.7-3B \
  --processor /absolute/path/to/Qwen3-VL-2B-Instruct --port 8203
```

First test each worker separately, without opening CAN or cameras (the model
receives black images and synthetic state; this is protocol/inference validation,
not a task-performance test):

```bash
uv run --no-sync python scripts/check_yam_worker.py \
  --endpoint http://127.0.0.1:8202 --max-horizon 30
uv run --no-sync python scripts/check_yam_worker.py \
  --endpoint http://127.0.0.1:8203 --max-horizon 16
```

For Router A/B evaluation:

```bash
cp -n configs/local-runtime-yam-vla-pair.yaml.example configs/local-runtime-yam-vla-pair.yaml
uv run --no-sync colosseum-policy-local --config configs/local-runtime-yam-vla-pair.yaml
```

If both workers do not fit in GPU memory, use the config's `launcher` lists to
start them on demand. Fill absolute environment and asset paths for **both**
entries, stop manually launched workers first, and invoke the model executable
directly. Avoid `bash`/`uv run` wrappers which may leave GPU child processes alive.
The supervisor only owns the process it launches; it does not stop separately
started services. Model cold starts may require increasing
`start_timeout_seconds` (e.g. 600) and Client `prepare_timeout`.

Copy your existing working YAM Client config to a separate VLA config and set
`llm_api_keys: {}` at the top level, so Router assigns VLA candidates. Do not
resume an existing LLM assignment. Keep your measured CAN, camera and gripper
settings. Start with:

```bash
uv run --no-sync colosseum-robot configs/robot.yam-vla.yaml --no-execute-action
```

Do not loosen Client `joint_max_step` merely to silence rejection. For execution,
confirm each model's absolute joint commands, gripper polarity and target changes
on the rig first. A configured 30 Hz is nominal; capture and inference can reduce
actual frequency. For pi05 YAM, use the dedicated setup below.

## pi05 YAM

`NUSMAGIC/pi05-MolmoAct2-YAM` at
`2f28d00ac28c543626f5a4f579a3da09bee4a4ed` is a LeRobot pi05 fine-tune:
absolute 14-D actions, 30-step chunks at 30 Hz, continuous grippers in [0, 1].
State/action order is left six joints, left gripper, right six joints, right
gripper. This uses `adapter: pi05_yam`, HTTP `/act`, and the YAM embodiment;
the existing Franka/DROID pi05 WebSocket/velocity worker is a different contract.

The worker loads both saved processor pipelines and their safetensors statistics.
These perform QUANTILES state/action normalization, state-to-language preparation,
PaliGemma tokenization, and action unnormalization. Do not supply DROID `--stats`.
Camera features retain checkpoint order **top, left, right**. RGB uint8 input is
converted to CHW float [0, 1], then the upstream policy applies its resize/padding
and [-1, 1] transform. The checkpoint declares 640x360 training images; the Client
uses 640x480, so verify actual camera framing against training before hardware use.
The worker does not invent a crop to resolve that difference.

Use the inspected LeRobot revision (the strict loader uses its key conversion
helpers):

```bash
uv venv --python 3.12 .venv-pi05-yam
uv pip install --python .venv-pi05-yam/bin/python \
  'lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@8c920c4270460851cedd2737657584586d3dc66f'
uv pip install --python .venv-pi05-yam/bin/python -e .
```

Provision the full checkpoint snapshot, including `model.safetensors`,
`config.json`, both processor JSON files and both processor statistics files.
Also provision a local `google/paligemma-3b-pt-224` tokenizer directory (access to
that upstream repository may require accepting its terms). Worker startup is
offline. It constructs the policy and loads weights strictly; unreadable or
incompatible weights raise an error rather than serving an uninitialized model.
Compilation and gradient checkpointing are disabled for this inference worker.

```bash
.venv-pi05-yam/bin/colosseum-policy-model pi05_lerobot --robot-type yam \
  --checkpoint /absolute/path/to/pi05-MolmoAct2-YAM \
  --tokenizer /absolute/path/to/paligemma-3b-pt-224 \
  --num-steps 10 --port 8204
```

`--num-steps` sets diffusion iterations, not the returned action horizon, which
remains 30. The worker resets policy history for each one-observation request,
predicts a complete chunk, then postprocesses it once. Invalid/nonfinite actions
or grippers outside [0, 1] are rejected, not silently clipped.

Use `configs/local-runtime-yam-pi05.yaml.example` as the Local Policy config, or
copy its model entry into a multi-model config. Its commented launcher uses the
model environment executable directly. Router-backed evaluation additionally
requires a matching Router model registration/runtime profile; this code change
does not register or deploy it.

Test the worker without opening robot hardware:

```bash
.venv/bin/python scripts/check_yam_worker.py --endpoint http://127.0.0.1:8204 --max-horizon 30
```

Repository tests cover mocked loading/processors, validation and HTTP/WebSocket
relay routing. Real checkpoint/GPU inference, installation compatibility and
physical task performance are not validated by those tests. The synthetic worker
check above is also not a hardware or task-performance test.

Sources: [checkpoint config](https://huggingface.co/NUSMAGIC/pi05-MolmoAct2-YAM/blob/2f28d00ac28c543626f5a4f579a3da09bee4a4ed/config.json),
[saved preprocessor](https://huggingface.co/NUSMAGIC/pi05-MolmoAct2-YAM/blob/2f28d00ac28c543626f5a4f579a3da09bee4a4ed/policy_preprocessor.json),
[LeRobot policy](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/policies/pi05/modeling_pi05.py).

### GR00T camera-order comparison

The default `--groot-camera-order checkpoint` preserves the saved processor.
For the pinned YAM checkpoint, `video_modality_keys: null` makes the inspected
LeRobot packer sort camera keys as left, right, top. To test an explicit
**top, left, right** order, append this to the GR00T YAM worker command:

```bash
--groot-camera-order top-left-right
```

For a Local Policy `launcher:` list, append these two arguments:

```yaml
      - --groot-camera-order
      - top-left-right
```

Restart the worker after changing the command. Startup logs show
`GR00T YAM camera order mode: top-left-right`. Remove the arguments or use
`--groot-camera-order checkpoint` to restore the saved behavior. This overrides
only the packer's `video_modality_keys` in memory; checkpoint files, image names,
state/action order and normalization are unchanged. The training-time camera
order is not established by this option; treat it as an A/B variant.

## G05 YAM

`RoboColosseum/G05-MolmoAct2-YAM` at
`685add3f748f823c49889f092f6c3cceabcae062` uses GalaxeaVLA, not LeRobot's
PI05/GR00T policy loader. The bridge exposes the same HTTP `/act` YAM protocol:
14-D absolute joint positions and continuous [0, 1] grippers, with **32-step**
chunks at nominal 30 Hz. Cameras map top/left/right to
`head_rgb/left_wrist_rgb/right_wrist_rgb` as CHW uint8; state maps to named
`left_arm`, `left_gripper`, `right_arm`, `right_gripper` groups without inversion.

Internally the saved recipe uses `RelativeJointTransform` for both arms, a 27-D
grouped/padded layout, stepwise normalization and ActionCodec. The bridge calls
the upstream `PolicyInferencer`, including its postprocessor, so actions are
unnormalized and converted back to absolute joints exactly once. Missing action
parts, unexpected dimensions and nonfinite values fail. Finite postprocessed
gripper targets are clipped to [0, 1] with a warning containing the side, count,
raw range and first affected chunk index. Arm joints are not clipped. Input
gripper state still must be within [0, 1].
The pinned upstream returns FM actions when `continuous_action` is enabled,
but attaches AR absence metadata even to FM output. The bridge logs that AR
metadata separately and validates every returned YAM part. For discrete output,
absent arm/gripper or unknown groups still fail; only the unused `lower_body`
group may be absent. No missing YAM commands are filled or synthesized.
The saved Hydra config selects Qwen3.5, even though the model README describes
Qwen2.5; use the saved architecture configuration.

The training `configs/data/parts_meta/yam.yaml` is published at checkpoint
revision `26c91270ce33325a29990b3ff7ea7a4b47bd4ec1`. That revision adds the metadata
and updates the README; the model weights and saved model configuration are
unchanged from the registered revision above. A copy is included in this repo at
`configs/data/parts_meta/yam.yaml`. It maps each 6-D arm into a 9-D control slot,
with 1-D grippers and an absent 7-D lower-body slot.

Supply `--g05-parts-meta` explicitly when using an older snapshot. Otherwise,
the worker checks the checkpoint's `configs/data/parts_meta/yam.yaml`, then the
same path in the upstream source checkout. Missing metadata fails before loading
weights. Use `OpenGalaxea/GalaxeaVLA@89f2322b4ad016e192437adc1a2c253b05bab246`
for the upstream serving implementation inspected here.

In a GalaxeaVLA environment matching that revision, install this Policy Server
package (`uv pip install --python /path/to/GalaxeaVLA/.venv/bin/python -e .`).
Provision the downloaded checkpoint directory, original parts metadata and local
Qwen3.5 processor assets, then run:

```bash
/path/to/GalaxeaVLA/.venv/bin/colosseum-policy-model g05 --robot-type yam \
  --checkpoint /absolute/path/to/G05-MolmoAct2-YAM \
  --source-root /absolute/path/to/GalaxeaVLA \
  --processor /absolute/path/to/qwen3_5_2b_base_processor \
  --g05-parts-meta /absolute/path/to/colosseum-policy-server/configs/data/parts_meta/yam.yaml \
  --port 8205
```

The bridge stages config/asset links in a temporary directory and uses the
upstream loader/setup helpers. It retains the checkpoint's discrete/continuous
head choices, disables compilation, and uses the same SDPA vision fallback as
our existing G05 worker (`--g05-native-attention` opts out). It does not modify
checkpoint files. Use `configs/local-runtime-yam-g05.yaml.example` for Local
Policy; Router-backed evaluation needs matching Router registration.

The matching processor directory is published under
`OpenGalaxea/G05/qwen3_5_2b_base_processor` at revision
`e312be81e90c56a55bcb26b57429bd39a335b449`. Download it separately with an HF
account authorized to access the base repository:

```bash
hf download OpenGalaxea/G05 --revision e312be81e90c56a55bcb26b57429bd39a335b449 \
  --include 'qwen3_5_2b_base_processor/*' --local-dir models/G05-base-assets
```

Confirm the intended inference head with the author: the saved config enables both
`discrete_action` and `continuous_action` and returns continuous actions, whereas
the model card reports token cross-entropy training. The bridge currently retains
the saved choices rather than guessing a head override.

Tests use mocked upstream model/processor objects and synthetic network traffic.
They do not establish checkpoint loading, dependency compatibility, GPU behavior,
or physical robot performance.

### Download the YAM checkpoints

Install the Hugging Face CLI in a download environment, then download **all**
files (including `.hydra` for G05 and processor statistics for pi05):

```bash
hf download RoboColosseum/G05-MolmoAct2-YAM \
  --revision 26c91270ce33325a29990b3ff7ea7a4b47bd4ec1 \
  --local-dir models/G05-MolmoAct2-YAM
hf download NUSMAGIC/pi05-MolmoAct2-YAM \
  --revision 2f28d00ac28c543626f5a4f579a3da09bee4a4ed \
  --local-dir models/pi05-MolmoAct2-YAM
```

The snapshots are approximately 11.95 GB and 9.35 GB respectively. Tokenizer /
processor assets are separate; downloading model weights alone does not make a
worker runnable.

Sources: [G05 snapshot](https://huggingface.co/RoboColosseum/G05-MolmoAct2-YAM/tree/685add3f748f823c49889f092f6c3cceabcae062),
[saved data recipe](https://huggingface.co/RoboColosseum/G05-MolmoAct2-YAM/blob/685add3f748f823c49889f092f6c3cceabcae062/data_yam_full.yaml),
[upstream serving](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/scripts/serve_policy.py).

Training metadata source: [published YAM parts layout](https://huggingface.co/RoboColosseum/G05-MolmoAct2-YAM/blob/26c91270ce33325a29990b3ff7ea7a4b47bd4ec1/configs/data/parts_meta/yam.yaml).

## Isaac 0.5 YAM

[`PerceptronAI/Isaac-0.5-YAM`](https://huggingface.co/PerceptronAI/Isaac-0.5-YAM) at
`dc52f2565cf76a9dd8dff87a386c945b9e81bdfd` is Isaac 0.5, a 36B sparse VLM,
fine-tuned on bimanual YAM demonstrations from xDoF ABC and MolmoAct2. It is a
LeRobot `perceptron_isaac` policy: 14-D absolute joint positions in the usual
YAM order, continuous [0, 1] grippers, **60-step** chunks at 30 Hz. Use
`adapter: isaac_yam` with HTTP `/act`. Cameras map top/left/right to the
policy's `top`/`left`/`right` image features.

The worker loads the checkpoint's saved pre/post processors, which do the
quantile normalization, image resizing and action unnormalization. The
checkpoint declares 640x360 images with `image_preprocessing: stretch`, so the
Client's 640x480 frames are stretched to 640x360 by the model's own processor;
the worker does not crop. Grippers outside [0, 1] are saturated with a warning,
as for pi05/G05/LingBot. Each request resets the policy, so requests are
independent.

The worker needs Perceptron's pinned LeRobot fork (Python 3.12, CUDA 12.8,
Linux x86_64), not the upstream LeRobot used by the other workers:

```bash
git clone --branch main --recurse-submodules https://github.com/perceptron-ai-inc/isaac.git
cd isaac/lerobot
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -c constraints/perceptron_isaac-cu128.txt \
  --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  '.[perceptron_isaac,training]'
uv pip install --no-deps -e /path/to/colosseum-policy-server
hf download PerceptronAI/Isaac-0.5-YAM --revision dc52f2565cf76a9dd8dff87a386c945b9e81bdfd \
  --local-dir /path/to/Isaac-0.5-YAM
```

```bash
colosseum-policy-model isaac --robot-type yam \
  --checkpoint /path/to/Isaac-0.5-YAM/recorded_policy --device cuda --port 9217
```

`--checkpoint` is the `recorded_policy` directory. The package is 143 GB of
fp32 weights. Isaac loads them on first use, so the worker runs one synthetic
inference before it opens its port; give the launcher a long
`start_timeout_seconds`. In bfloat16 it needs most of an 80 GB GPU. Perceptron
reports about 0.95 s per YAM chunk on one H100. `--num-steps` (default 10) sets
the flow steps, not the chunk length. If the Router registers a `max_horizon`
below 60, pass `--isaac-return-steps` with that value: the YAM adapter rejects
longer chunks.

Use [local-runtime-yam-isaac.yaml.example](../configs/local-runtime-yam-isaac.yaml.example)
as the runtime config, either with its launcher or with a relay to a remote GPU
host, then check the worker without the robot:

```bash
.venv/bin/python scripts/check_yam_worker.py --endpoint http://127.0.0.1:9217 --max-horizon 60
```

Tests use stand-in torch/LeRobot modules and the published `config.json`. They
do not load the weights or establish performance on a physical rig.
