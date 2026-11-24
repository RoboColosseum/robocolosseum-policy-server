# Dream-1 YAM in Colosseum

The `dreamlabs` backend uses `dreamlabs-client` to call Dream's cloud API.
It is selected by the existing `auto` backend through
`backend_options.backend: dreamlabs`. Hardware execution stays in the Client;
no Client source changes or separate GPU worker are required.

## Install and configure

Install the optional dependency group in the Policy Server environment:

```bash
python -m pip install -e '.[dreamlabs]'
```

The tested SDK version is 1.1.3. Supply credentials through `DREAM_API_KEY`
or the SDK's `~/.dreamlabs/credentials`; never put credentials in Git.
Copy `configs/local-runtime-yam-dreamlabs.yaml.example` to a local config.
Its model identity reflects the existing station registration; confirm it
matches your Router registration before using it elsewhere. The zero revision
is a registration identifier, not a downloaded or pinned model checkpoint.

```bash
colosseum-policy-local --config configs/local-runtime-yam-dreamlabs.yaml
```

For a mixed-model runtime, copy the `backend.options.dreamlabs` block and
the Dream model entry into the existing `auto` configuration. There is no
new backend entry point and no model worker launcher.

## YAM contract

The model uses 14-D absolute joint targets at a Client rate of 30 Hz:
left six joints, left gripper, right six joints, right gripper.
Observations contain twelve joint positions and two grippers in [0, 1].
Camera IDs `head_image`, `left_image`, `right_image` map to Dream's
`top`, `left`, `right`. RGB images are center-cropped to 16:9 and encoded
as 640 x 360 JPEGs. Empty instructions, invalid state and malformed action
chunks fail before execution.

Dream plan rows play at 20 Hz through interpolation; the backend returns
three Client steps per call. Its initial planner hold asks three extra
times at 2.4-second intervals. Network latency can extend this startup.
Do not change playback settings without checking the model contract.

## Timeouts and recovery

The deployed SDK request timeout is 30 seconds. The existing station uses
a 60000 ms Client inference deadline; configure that separately in the
Client's local config, rather than changing the Client default.
TLS/socket disconnects and expired sessions each allow three extra attempts
with 0.5, 1 and 2 second backoff. Authentication, malformed requests, SDK
timeouts and cloud capacity errors are not automatically retried here.

Cancellation stops further retries and discards late results. SDK threads
already running cannot be interrupted; session shutdown waits for them
before closing. A local server being ready does not establish Dream cloud
capacity. `NO_CAPACITY` requires cloud availability, not a local restart.

## Offline verification

```bash
python -m pip install -e '.[dev,dreamlabs]'
python -m pytest tests/test_dreamlabs.py tests/test_dreamlabs_recovery.py -q
```

These tests use synthetic observations and fake SDK calls. They do not
start a robot session or verify cloud availability.
