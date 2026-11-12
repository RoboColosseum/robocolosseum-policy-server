## Test data with a real robot identity

```yaml
robot_type: franka # or yam
test: true        # dummy data/actions; false uses the hardware adapter
track: 1          # 1 Open, 2 Fine-tuning
```

`test` must be a YAML boolean. The normal Client command and evaluation workflow
are shared. Dummy local action dimensions follow the Router-assigned model contract;
this is a transport fixture, not a validated YAM hardware interface. Model loading
is still simulated by `colosseum-policy-test-backend`.

Router persists an immutable test flag per assignment, exposes it in Open and
Fine-tuning reviews, and excludes synthetic results from formal ranking, difficulty
fitting and Fine-tuning summaries. Test attempts do not consume formal trial quotas.
Test-only deployments are unavailable to `test: false` clients. Ready messages for
simulation cannot start an assignment marked real. Changing mode during a pending
assignment requires completing or aborting that assignment first.

Web review is available at `/web/evals` on the Router, with Test badges under
the original robot identity. Historical `robot_type: test` records are retained,
but Legacy test is no longer offered in the robot selector.
The old robot_type test syntax remains accepted for compatibility. New configs
should always use a real robot_type with a separate test flag.

# Local simulation with the deployed Router

Restart the Policy Server after updating its source. From `colosseum-policy-server`:

```bash
uv run colosseum-policy-test-backend
```

From `colosseum-client`:

```bash
uv run colosseum-robot configs/robot.yaml
```

The Client example defaults to `robot_type: franka`, `test: true`, `track: 1`,
`router_url: wss://robocolosseum-api.onrender.com`, and `policy_server_url: ws://localhost:8000`.
No cameras or robot are needed. No `SSL_CERT_FILE` is needed; unset it if previously
exported for a different local certificate. Cloud TLS verification stays enabled.

## Router task menu

Client fetches `/api/eval/tasks?robot_id=franka&test=true`; the task names are not embedded in Client.
The deployed test catalog uses the five Franka tasks from `RoboColosseum_Plan_260902.xlsx`:

1. Close laptop
2. Place the mug on the coaster
3. Open the drawer
4. Pour the water of the cup into the bowl
5. Put the towel into the basket

These are separate synthetic task records, not finalized hardware/scoring protocols.
For `track: 1`, Client asks for a prompt instead.

## Shared evaluation lifecycle

All robot types run the same Client evaluation workflow. `test` selects the dummy
robot adapter. Router assigns the models; Policy Server validates their identity,
simulates download (3s), load (2s), warm-up (1s), then sends `simulation_ready`.
Client enables inference immediately and reports readiness to Router in the background,
explicitly retaining `verification_only=true`, `loaded=false`, and `state=simulation`.
This is a completed simulation trial, not a claim of real model loading.

Each observation comes from the dummy adapter. After 0.5s the server returns an
ActionPlan echoing joint and gripper positions. The assigned action space must be
`joint_position`, and its dimension must equal the combined state length. Other
state orders can be selected with repeated `--state-field` options. Client records
frames, validates and applies
actions in memory. Open Track runs A, asks for partial success, then prompts to restore
the scene and runs B, asks for partial success and A/B/tie preference, and uploads
videos/results. Fine-tuning follows the same single-trial evaluation workflow as other
robot types. Resume/abort are supported by the normal Client.

Results are stored on Router under robot `test`; no automatic abort or local-only
score path is used. The prepared config uses `max_trial_steps: 3` for short Open Track
trials; this is a config choice, not different evaluation logic. Fine-tuning step limits
come from the task. The adapter supplies `head_image` when cameras are omitted.

Preparation has a total `prepare_timeout` (default 1800 seconds), which progress
cannot extend. Each inference has `deadline_ms` (default 30000). Data sent during
preparation is rejected with `NOT_READY` and connection closure. Failures stop the
trial and preserve recording/resume information. Router reporting is asynchronous;
completion waits for all reported steps to be persisted before scoring or switching.

Optional slow-service testing (normal users do not need these flags):

```bash
uv run colosseum-policy-test-backend --download-seconds 20 --load-seconds 10 --warmup-seconds 5 --inference-seconds 2
```

All delays are synthetic; no weights are downloaded or loaded. The test deployment's
revision is synthetic, not a claim about an actual Hugging Face commit. Receipts and
synthetic action records are written to `local-receipts.jsonl`.

## Wire protocol

Both WS and WSS use binary Protobuf only, from the shared `proto/colosseum.proto`:
`RelayFrame(type=LOCAL_CONTROL)` wraps `LocalControl` for capabilities, prepare,
received, progress, simulation_ready and errors. Observations and action plans use
`OBSERVATION` and `ACTION_PLAN` frames. Console/JSONL diagnostics are not wire JSON.
Client marks simulation lifecycle reports explicitly; no real weights are loaded. Physical robot delivery checks retain
receipt-only behavior and cannot request simulated action output.

## Optional local TLS

Supply both `--certfile /path/to/fullchain.pem --keyfile /path/to/privkey.pem` to the
Policy Server, and use `wss://` in Client. Use a certificate trusted by Client.
The default localhost WS setup above does not need certificates.

## Optional isolated Router

For development without the cloud Router, run from `colosseum-router`:

```bash
uv run python scripts/run_communication_router.py
```

This seeds the same five tasks in a fresh temporary database. It prints the generated
Client config and local Router certificate paths. Point the Policy Server to its
printed policy port (default 18444), and run Client with that generated config and
`SSL_CERT_FILE` pointing to the local Router certificate. This extra certificate is
only needed for this optional self-signed Router, not the deployed Router.
