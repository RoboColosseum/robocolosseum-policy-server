# Fine-tuning GeNIE's SAM-TP on the track dataset

GeNIE's only learned part is SAM-TP, the model that says which pixels are drivable. The planner and controller
are fixed. These scripts fine-tune SAM-TP from the authors' `checkpoint_2.pt` on the Earth Rover track's dataset,
[`RoboColosseum/earth-rover-mini-plus-wuhan-level-1`](https://huggingface.co/datasets/RoboColosseum/earth-rover-mini-plus-wuhan-level-1)
(107 tele-operated drives of Wuhan mission-1, two Mini+ rovers). BIT-3453.

The dataset has camera frames and driving commands, not drivable-ground masks, so the labels are made from the
drives themselves.

## Labels

| Step | Script | What it does |
|---|---|---|
| Frames and driven path | `prepare.py` | Samples frames at 2 Hz while the rover moves forward. Each frame is converted back to the runtime 512×288 geometry, since the dataset stretched 1024×576 to 640×480. The script integrates wheel odometry over the next 5 m: speed is 0.00446 m/s per rpm, fitted against GPS, and yaw rate comes from the left/right wheel difference, fitted against the camera's own rotation. It then draws a rover-wide strip with GeNIE's camera model. Episodes with `episode % 7 == 3` are held out. |
| Original model | `teacher.py` | Runs the authors' SAM-TP exactly as GeNIE calls it, and keeps its drivability per pixel. |
| Masks | `label.py` | Grounding DINO boxes (curb, step, stairs, wall, hedge, bollard, bench, person, bicycle and so on) are turned into SAM 2.1 large masks and marked not drivable. The ground SAM 2.1 large grows from the driven strip is marked drivable, and everything above the horizon is not drivable. Where the original model is confident ground is not drivable it stays so; the rest is unknown. Overexposed frames are skipped. |
| Clearance margin | `refine.py` | Measures how close the human drivers came to near obstacles (within 1.8 m, on the ground plane). Ground inside that clearance that no one drove on gets a lower target, rising from the edge of the obstacle to fully drivable at the margin. |

Only the near field is used for the margin. With the camera 0.14 m above the ground, a row 16 px below the horizon
is already 2 m away, so obstacles near the horizon cannot be placed reliably.

## Training

`train.py` uses the same forward pass as GeNIE: SAM2 transforms, the learned prompt, a single mask, and logits
resized to the frame. It trains every part, as the authors' `freezeNoneNone` configuration does, with the image
encoder at a lower learning rate. The loss is:
- on known pixels, 20 × focal + 1 × dice (the authors' weights, here with soft targets);
- on unknown pixels, cross-entropy towards the original model, so it keeps its answer where the labels say
  nothing.

The run uses bf16 and a batch of 8 on one A10G (24 GB). An epoch of 5,153 frames takes 4.8 minutes.

## Running it

Two environments: the training one has GENIE-SAMTP's own `sam2` package; the labelling one has `transformers`
(Grounding DINO, SAM 2.1 large), kept apart because GENIE-SAMTP's `sam2` builds a prompt-less encoder by default.

```bash
python prepare.py --data lerobot --out prepared
python teacher.py prepared checkpoint_2.pt prepared/teacher
python label.py prepared prepared/teacher prepared/labels          # labelling env, ~75 min on an A10G
python refine.py prepared prepared/labels prepared/targets-v2 75 0.05   # margin = 75th percentile of human clearance, target 0.05 at the obstacle
python train.py --data prepared --labels prepared/targets-v2 --teacher prepared/teacher --ckpt checkpoint_2.pt --out runs/ft2 \
  --epochs 6 --batch 8 --lr 3e-5 --vision-lr 6e-6 --distill 0.3
python eval_planner.py prepared prepared/targets checkpoint_2.pt runs/ft2/best.pt runs/ft2/planner_eval.json 600
python sheet.py prepared prepared/labels sheet.jpg                  # contact sheet to review labels
```

## Results (29 October 2026)

Labels: 5,991 frames (5,153 training, 838 held out from 15 episodes). Human clearance from near obstacles over
1,020 frames: 25th percentile 0.14 m, median 0.36 m, 75th percentile 0.68 m.

Variant 1: margin = median clearance (0.36 m), target 0.2 at the obstacle edge, 8 epochs, learning rate 1e-5
(image encoder 2e-6), best epoch 4.

| Held out | Original | Variant 1 |
|---|---|---|
| IoU against the labels | 0.9925 | 0.9965 |
| Obstacle pixels called drivable | 0.06 % | 0.01 % |
| Driven-path pixels called drivable | 99.32 % | 99.43 % |
| Mean drivability inside the margin (target 0.64) | 0.92 | 0.80 |
| Agreement with the original model where labels are unknown | — | 97.5 % |

GeNIE's planner on 590 held-out frames, with checkpoint 2 as the goal:

| | Original | Variant 1 |
|---|---|---|
| Planned heading within 15° of the human's | 77.6 % | 77.8 % |
| Paths within 0.3 m of an obstacle | 0.7 % | 0.5 % |
| Paths within 0.5 m of an obstacle | 3.9 % | 4.1 % |
| Plans that trigger a recovery | 0.2 % | 0.0 % |
| Near-obstacle frames (82): path unchanged / farther / closer | | 78 / 1 / 3 |

The original SAM-TP already segments this plaza well. Variant 1 learns the clearance only part of the way, and
GeNIE's path choice barely changes.

Variant 2, the one in use: margin = 75th percentile of human clearance (0.68 m), target 0.05 at the obstacle edge,
6 epochs, learning rate 3e-5 (image encoder 6e-6), distillation weight 0.3, best epoch 3. Held out: IoU 0.9966,
obstacle pixels called drivable 0.01 %, driven-path pixels called drivable 99.40 %, mean drivability inside the
margin 0.77 (original 0.94, target 0.60).

| GeNIE's planner, 590 held-out frames | Original | Variant 1 | Variant 2 |
|---|---|---|---|
| Planned heading within 15° of the human's | 77.6 % | 77.8 % | 78.1 % |
| Median path clearance from near obstacles | 0.72 m | 0.72 m | 0.76 m |
| Paths within 0.5 m of an obstacle | 3.9 % | 4.1 % | 3.6 % |
| Plans that trigger a recovery | 0.2 % | 0.0 % | 0.0 % |
| Near-obstacle frames (82): path farther / closer / unchanged | | 1 / 3 / 78 | 15 / 4 / 63 |

Next to the hedge wall the fine-tuned model shades the strip of paving along the curb as less drivable, and
GeNIE's path moves away from the wall. These are offline numbers on 15 held-out drives; trials on the rover decide.

## Where things are

- Weights: [`RoboColosseum/genie-samtp-wuhan-level-1`](https://huggingface.co/RoboColosseum/genie-samtp-wuhan-level-1)
  (private; the base checkpoint has no published licence), files `genie-samtp-wuhan-v2.pt` and `-v1.pt`.
- Labels, logs and evaluations: `s3://autonav-training-outputs-487525999771/robocolosseum/genie-samtp-wuhan-level-1/2026-10-29/`
  (`artifacts.tar`, 410 MB, plus both checkpoints).
- Compute: one g5.2xlarge in Tokyo (Singapore offers no g5/g6), about 8 hours including both variants, about US$15.
- Policy: `genie-rover-ft`, deployment `ft-rover-wuhan-mission-1-cp2-genie-ft`.

```bash
hf download RoboColosseum/genie-samtp-wuhan-level-1 genie-samtp-wuhan-v2.pt --local-dir weights
genie-rover --checkpoint weights/genie-samtp-wuhan-v2.pt ...
```
