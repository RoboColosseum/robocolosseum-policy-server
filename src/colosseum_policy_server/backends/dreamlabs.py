"""Dream Labs backend: a Dream model on a bimanual YAM through the public `dreamlabs-client` library.

How a trial runs:
- start_session() opens one model session; end_session() closes it.
- The first infer() holds the arms still for under 8 s while it asks the model again, so the first rows the arms run
  already follow the plan of the model's server-side task planner, whose first plan takes 3.5 to 6.5 s.
- Every infer() sends the Client's observation and returns the targets for its next `rows_per_call` loop steps.
  The model's rows are played at `plan_hz`, the rate Dream Labs' own YAM stations play them at, whatever the
  Client's rate: each target is interpolated between the rows due at that step.
- The dream client keeps the model's frame window (9 frames at 5 fps) and fetches the next chunk in the background
  while the arms play the current one, so a call waits on the network only when the next chunk has not landed yet.

The API key is read by the dream client itself, from `DREAM_API_KEY` or `~/.dreamlabs/credentials`.
"""
from __future__ import annotations

import asyncio
import base64
import time

import numpy as np

from .base import LocalModel
from .. import colosseum_pb2 as pb
from ..model_adapters.bounds import check_bounds
from ..model_adapters.observation import _rgb, _state

# Fixed by the model's inputs: frames are center-cropped to 16:9, never squished, then sized to FRAME_SIZE.
FRAME_SIZE = (640, 360)
JPEG_QUALITY = 85
CAMERAS = {"head_image": "top", "left_image": "left", "right_image": "right"}
# Fixed by the model's YAM wire: a YAM, joint targets, grippers 0 closed to 1 open.
YAM = {"robot.arm": "yam", "robot.exec_head": "joint", "robot.max_joint_vel_rad_s": 0.0, "robot.gripper_open": 1.0}

# Defaults of the backend options of the same name in lower case (configs/local-runtime-yam-dreamlabs.yaml.example).
# Model rows per second, as Dream Labs' YAM stations play them, ramping linearly from one row to the next.
PLAN_HZ = 20
# Targets per call: at 30 Hz, one observation every 0.1 s, twice the model's 5 fps frame window.
ROWS_PER_CALL = 3
# The dream client paces from its last 5 round trips and a session's first is slow, so until 5 replies are in, the
# next chunk is asked for as each lands, at most once per WARMUP_GAP_S: the API allows one request per 0.7 s.
WARMUP_REPLIES = 5
WARMUP_GAP_S = 1.0
# The task planner's first plan lands 3.5 to 6.5 s into a session and no reply says when, so a trial's first call
# asks HOLD_ASKS more times, HOLD_PERIOD_S apart, the arms still, and plays the reply asked for 7.2 s after the first.
HOLD_ASKS = 3
HOLD_PERIOD_S = 2.4


def create_backend(options):
    """Return the backend for the `dreamlabs` entry point."""
    return DreamLabsBackend(options)


def _jpeg(rgb: np.ndarray) -> str:
    """Return an RGB frame center-cropped to 16:9 and sized to FRAME_SIZE, as base64 JPEG."""
    import cv2

    height, width = rgb.shape[:2]
    crop_h, crop_w = min(height, round(width * 9 / 16)), min(width, round(height * 16 / 9))
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    frame = rgb[top:top + crop_h, left:left + crop_w]
    if (crop_w, crop_h) != FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    # The model decodes with OpenCV, which reads BGR.
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _model_inputs(observation: pb.Observation):
    """Return the model's frames and state for a Client observation; raise ValueError on a malformed one."""
    if not observation.instruction.strip():
        raise ValueError("observation instruction is empty")
    images = {image.sensor_id: image for image in observation.sensors}
    if len(images) != len(observation.sensors):
        raise ValueError("Duplicate camera sensor IDs")
    joints = _state(observation, "joint_position", (12,))  # left 6, right 6
    grippers = _state(observation, "gripper_position", (2,))  # left, right
    check_bounds(grippers, 0, 1, ("left_gripper", "right_gripper"), "YAM grippers must be in [0, 1]")
    frames = {f"image_window.{camera}": _jpeg(_rgb(images.get(source), source)) for source, camera in CAMERAS.items()}
    state = {"prompt": observation.instruction, **YAM,
             "state.arm_l.joint_pos": joints[:6].tolist(), "state.arm_l.gripper_pos": grippers[:1].tolist(),
             "state.arm_r.joint_pos": joints[6:].tolist(), "state.arm_r.gripper_pos": grippers[1:].tolist()}
    return frames, state


def _chunk_from_reply(reply) -> np.ndarray:
    """Return a get_action() reply as one (H, 14) chunk in Client order; raise ValueError on a malformed one."""
    left, right = np.asarray(reply["q_l_chunk"], np.float32), np.asarray(reply["q_r_chunk"], np.float32)
    if left.ndim != 2 or left.shape != right.shape or left.shape[1] != 7 or not len(left):
        raise ValueError(f"dream returned chunks of shapes {left.shape} and {right.shape}, not matching (H, 7)")
    chunk = np.concatenate([left, right], axis=1)  # left 6, left gripper, right 6, right gripper
    if not np.isfinite(chunk).all():
        raise ValueError("dream returned non-finite actions")
    chunk[:, [6, 13]] = np.clip(chunk[:, [6, 13]], 0, 1)  # gripper rows can pass [0, 1] by a hair
    return chunk


class DreamLabsBackend:
    """One trial's dream client session and the chunk the arms are playing.

    Plan time (`plan_s`) is the model's clock: row k of a chunk is due k / plan_hz s of plan time after the
    observation it answers. It follows the wall clock while the arms play and stands still while they wait.
    """

    def __init__(self, options):
        """Read the checkpoint and the playback options; the session opens in start_session()."""
        self.checkpoint = options["checkpoint"]
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError("dreamlabs backend requires a checkpoint, e.g. dreamlabs/dream-1-beta")
        self.plan_hz = options.get("plan_hz", PLAN_HZ)
        self.rows_per_call = options.get("rows_per_call", ROWS_PER_CALL)
        self.warmup_replies = options.get("warmup_replies", WARMUP_REPLIES)
        self.warmup_gap_s = options.get("warmup_gap_s", WARMUP_GAP_S)
        self.hold_asks = options.get("hold_asks", HOLD_ASKS)
        self.hold_period_s = options.get("hold_period_s", HOLD_PERIOD_S)
        self.client = None

    async def start_session(self, model: LocalModel, request) -> None:
        """Check the model is a 14-D YAM joint model and open the trial's dream client session."""
        if model.action_space != "joint_position" or model.action_dim != 14:
            raise ValueError("dreamlabs backend requires 14-D YAM joint_position")
        if model.max_horizon < self.rows_per_call:
            raise ValueError(f"dreamlabs backend requires max_horizon of at least {self.rows_per_call}")
        import dream

        self.control_hz = model.control_hz
        # The most plan time one call may add, so a stalled Client resumes the plan where its arms stopped.
        self.max_advance_s = 2 * self.rows_per_call / model.control_hz
        self.client = dream.DreamClient()
        # action_hz times the client's prefetch in rows of the plan.
        self.client.policy({"checkpoint": self.checkpoint, "strict": False, "action_hz": self.plan_hz})
        self.plan_s = 0.0
        self.playing_since = None  # wall time the arms started the last call's targets; None while they wait
        self.played_s = 0.0
        self.chunk = None  # (H, 14) rows in Client order
        self.chunk_plan_s = 0.0  # plan time of the chunk's row 0
        self.takeover_row = 0.0  # the chunk's row the arms took it over at
        self.request = None  # the get_action() in flight
        self.asked_at = 0.0  # wall time of the request's observation
        self.asked_plan_s = 0.0  # plan time of the request's observation
        self.replies = 0
        self.last_target = None  # what the arms were last commanded

    async def infer(self, model: LocalModel, observation: pb.Observation) -> np.ndarray:
        """Return the targets of the Client's next rows_per_call steps, requesting the next chunk when it is due."""
        now = time.monotonic()
        self._advance_plan(now)
        self._install_reply()
        self.client.set_observation(*_model_inputs(observation), timestamp=now)
        if self.replies == 0 and self.request is None:
            await self._hold_for_planner()
        due = self._next_chunk_due(now)
        if self.request is None and due:
            self._ask(now)
        targets = await self._wait_for_targets()
        self.last_target = targets[-1]
        return targets

    async def end_session(self) -> None:
        """Wait for the request in flight, print the playback rate and close the session."""
        if self.client is None:
            return
        if self.request is not None:
            await asyncio.wait({self.request})  # the client takes one call at a time, close() included
        if self.played_s:
            print(f"dreamlabs: played {self.plan_s * self.plan_hz:.0f} plan rows in {self.played_s:.1f} s of play, "
                  f"{self.plan_s * self.plan_hz / self.played_s:.1f} rows/s", flush=True)
        print(f"dreamlabs: closing session {self.client.session_id}", flush=True)
        self.client.close()
        self.client = None

    def _advance_plan(self, now: float) -> None:
        """Move plan time by the wall time the arms played the last call's targets."""
        if self.playing_since is None:
            return
        self.plan_s += min(now - self.playing_since, self.max_advance_s)
        self.played_s += now - self.playing_since

    async def _hold_for_planner(self) -> None:
        """Ask the model hold_asks more times with the first observation, hold_period_s apart; the arms stay still."""
        started = time.monotonic()
        for _ in range(self.hold_asks):
            self._ask(time.monotonic())
            await asyncio.wait({self.request})
            self.request.result()  # this reply is dropped, but a failed call still raises
            self.request = None
            self.replies += 1
            await asyncio.sleep(max(0.0, self.asked_at + self.hold_period_s - time.monotonic()))
        self._ask(time.monotonic())
        await asyncio.wait({self.request})
        print(f"dreamlabs: held the first chunk {time.monotonic() - started:.1f} s for the planner", flush=True)

    def _next_chunk_due(self, now: float) -> bool:
        """Whether to request the next chunk: none yet, a session's first replies, or the dream client's pacing."""
        warming_up = (self.replies < self.warmup_replies and self.client.mode == "sync"
                      and now - self.asked_at >= self.warmup_gap_s)
        if self.chunk is None or warming_up:
            return True
        row = (self.plan_s - self.chunk_plan_s) * self.plan_hz
        return self.client.refill_due(int(row - self.takeover_row), len(self.chunk) - int(row))

    def _ask(self, now: float) -> None:
        """Request the next chunk in the background for the latest observation."""
        # The new chunk takes over from the row after each arm's last commanded joints.
        last = self.last_target
        commanded = None if last is None else {"q_l_chunk": last[:7].tolist(), "q_r_chunk": last[7:].tolist()}
        self.request = asyncio.create_task(asyncio.to_thread(self.client.get_action, commanded=commanded))
        self.asked_at, self.asked_plan_s = now, self.plan_s

    async def _wait_for_targets(self) -> np.ndarray:
        """Return the call's targets, holding the arms until a chunk with rows still due has landed."""
        while (targets := self._targets()) is None:
            self.playing_since = None
            if self.request is None:  # the last reply landed too late to use
                self._ask(time.monotonic())
            await asyncio.wait({self.request})
            self._install_reply()
        self.playing_since = time.monotonic()
        return targets

    def _targets(self) -> np.ndarray | None:
        """Return the call's targets, interpolated between the rows due then; None once the chunk has run out."""
        if self.chunk is None:
            return None
        steps_s = np.arange(self.rows_per_call) / self.control_hz
        rows = (self.plan_s - self.chunk_plan_s + steps_s) * self.plan_hz  # fractional plan rows, one per loop step
        rows = rows[rows <= len(self.chunk) - 1]
        if not len(rows):
            return None
        below = np.floor(rows).astype(int)
        above = np.minimum(below + 1, len(self.chunk) - 1)
        share = (rows - below)[:, None]
        return self.chunk[below] + share * (self.chunk[above] - self.chunk[below])

    def _install_reply(self) -> None:
        """Install the landed reply's chunk, skipping the rows the arms played since its observation."""
        if self.request is None or not self.request.done():
            return
        reply, self.request = self.request.result(), None
        self.replies += 1
        chunk = _chunk_from_reply(reply)
        takeover_row = (self.plan_s - self.asked_plan_s) * self.plan_hz
        if takeover_row > len(chunk) - 1:
            print(f"dreamlabs: discarding a chunk that landed after all {len(chunk)} of its rows were due", flush=True)
            return
        self.chunk, self.chunk_plan_s, self.takeover_row = chunk, self.asked_plan_s, takeover_row
