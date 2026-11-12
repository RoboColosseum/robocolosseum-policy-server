# GeNIE on the Earth Rover Mini+

The first benchmark policy of the Earth Rover track. GeNIE ([project page](https://clear-nus.github.io/genie/),
[paper](https://arxiv.org/abs/2506.17960)) is a navigation system from the NUS CLeAR Lab that won the Earth
Rover Challenge at ICRA 2025. In Colosseum it runs as policy `genie-rover`.

It is an Earth Rovers SDK program, so it runs behind `colosseum-rover-sdk-bridge`. It also runs against a real
Earth Rovers SDK.

## What is GeNIE's and what is ours

GeNIE's public code is [GENIE-SAMTP](https://github.com/jiaming-ai/GENIE-SAMTP). It is installed from git at a
pinned commit, and its functions are called as released:

- **SAM-TP**: SAM 2.1 Hiera-tiny with a learned traversability prompt, using the authors' checkpoint (`checkpoint_2.pt`).
- **BEV path planner**:
  - ground-plane projection;
  - the seeded polynomial path bank;
  - high-cost filtering;
  - adaptive k-means path clusters, with the group closest to the goal direction;
  - footprint costs and the top-k path average.

The live system that won the challenge has not been published. These parts are reconstructed from the paper:

| Part | Here |
|---|---|
| Goal term in the path cost, `C = sum f + beta * sum g` | The released planner scores traversability only. `planning.py` adds `g` as each point's angular distance from the goal direction, divided by 180 degrees, with `beta = 1` (the value the paper reports). |
| Waypoint controller: "align heading with the next waypoint, then advance 1 m" | `controller.py`. The rover turns in place when the waypoint is more than 35 degrees off, and otherwise drives with gentle steering. It re-plans on every new frame. |
| VLM recovery when no path survives | Not released. The rover backs up, then turns 45 degrees toward the goal, alternating sides. |
| Localization | GPS gives the goal bearing. Heading comes from the compass, plus an offset learned from straight GPS segments. |
| Mini+ camera | 92.6 degree horizontal FOV, 0.14 m above the ground, level (`camera.py`; estimates, no calibration exists). |

The path bank is sampled once instead of per frame. It is seeded and independent of the goal, and sampling it took 2 s per frame.

## Fine-tuned on the track dataset

`finetune/` fine-tunes SAM-TP on the Earth Rover track's dataset (the Wuhan level-1 drives), with labels made from
the drives: the path the human took, detected obstacles, and the clearance the drivers kept from them. Policy
`genie-rover-ft` runs GeNIE with that checkpoint (`--checkpoint weights/genie-samtp-wuhan-v2.pt`); `genie-rover` is
the authors' weights, unchanged. Method, results and where the weights are: [`finetune/README.md`](finetune/README.md).

## Setup

```bash
cd baselines/genie_rover
uv sync --extra dev
.venv/bin/genie-rover-weights      # 409 MB from the authors' Google Drive, checked against its SHA-256
.venv/bin/python -m pytest -q      # no torch, checkpoint or network needed
```

Device choice is automatic: CUDA, then Apple MPS, then CPU. Timings on an M2 Pro (MPS):

- SAM-TP: 0.13 s per frame.
- Full plan: 0.25–0.35 s per frame.

## Run a trial

```bash
colosseum-rover-sdk-bridge --config configs/policy.yaml --cwd baselines/genie_rover \
  --run "baselines/genie_rover/.venv/bin/genie-rover --yes-i-want-the-rover-to-move --base-url http://127.0.0.1:8000 --max-seconds 170"
```

Without `--yes-i-want-the-rover-to-move`, the program plans and logs but sends no commands. Each run writes
`runs/<time>/log.jsonl` (one line per control tick) and one picture per plan (the camera frame with SAM-TP
traversability, beside the BEV plan). `genie-rover-replay IMAGES... --output DIR` plans on saved frames without
a rover.

## Timing through Colosseum

The bridge receives a new observation every 2–3 s and answers it at once with the command that is current at
that moment. The Client then holds that command until the next answer, so the rover never sees a command
set between two observations. Two rules follow:

- An advance holds its command until the next plan replaces it.
- An in-place turn lasts until at least one observation has been answered with it. Each answered turn command
  turns the rover about 45 degrees.

A command reaches the rover one observation after the frame it was planned on. At 1.1 m/s the rover therefore
moves about 5 m between seeing something and reacting to it.

## First results (27 October 2026)

Task `rover-wuhan-mission-1-cp2-v1` (Wuhan mission-1, checkpoint 1 to checkpoint 2, about 28 m, 180 s), on
`luke-notch-quirk`, operated from La Plata, with GeNIE on an M2 Pro:

| Trial | Score | Distance to goal | Notes |
|---|---|---|---|
| `FT-23689d1733ecd234` | 2 | 31.7 m to 15.0 m | Advances were dropped between observations (fixed in `advance_s`). |
| `FT-717948f460957348` | 2 | 28.2 m to 15.0 m | In-place turns shorter than the observation gap never reached the rover (fixed: a turn now lasts until an observation samples it). |
| `FT-2ffdccd82d72c839` | 3 | 26.8 m to 6.4 m | Called `/checkpoint-reached` at 7.9 m after 110 s and stopped. At 74–76 s it came within centimetres of a stone step at 1.2 m/s, so contact can't be ruled out and the score is not 4. |

The weights come from the authors' Google Drive. GENIE-SAMTP has no license file, so ask the authors before
redistributing the checkpoint.
