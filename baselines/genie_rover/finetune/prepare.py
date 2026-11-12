"""Frames and driven-path footprints from the Wuhan level-1 LeRobot dataset, for SAM-TP fine-tuning (BIT-3406).

For every sampled frame:
  - the front image, back in the runtime geometry (the dataset stretched 1024x576 to 640x480; GeNIE sees 512x288);
  - the path the human drove next (wheel odometry: speed from the mean wheel rpm, yaw rate from the left/right
    wheel difference, scaled against the camera's own rotation), drawn as a rover-wide ground strip with GeNIE's
    camera model (92.6 deg HFOV, 0.14 m high, level, 0.11 m ahead of the rover centre);
  - metadata for the offline planner check (the human's heading 1 m along the path, the goal direction).

    python prepare.py --data lerobot --out prepared
"""
import argparse, glob, json, math, os

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

FPS = 10
W, H = 512, 288
HFOV = 92.6
CAM_H, CAM_FWD = 0.14, 0.11
M_PER_RPM = 0.00446          # fitted against GPS over 4 s windows (3,229 windows, median error 0.17 m/s)
ROVER_WIDTH = 0.40
PATH_M, HORIZON_S = 5.0, 6.0
STEP = 5                     # 2 Hz samples
F = (W / 2) / math.tan(math.radians(HFOV / 2))


def episodes(data):
    meta = pq.read_table(glob.glob(os.path.join(data, 'meta/episodes/*/*.parquet'))[0]).to_pandas()
    for _, row in meta.iterrows():
        ep = int(row['episode_index'])
        df = pq.read_table(os.path.join(data, f"data/chunk-000/file-{int(row['data/file_index']):03d}.parquet")).to_pandas()
        df = df[df['episode_index'] == ep].reset_index(drop=True)
        video = os.path.join(data, f"videos/observation.images.front/chunk-{int(row['videos/observation.images.front/chunk_index']):03d}/"
                                   f"file-{int(row['videos/observation.images.front/file_index']):03d}.mp4")
        yield ep, df, video, float(row['videos/observation.images.front/from_timestamp'])


def decode(video, start, n):
    frames = []
    with av.open(video) as box:
        for frame in box.decode(video=0):
            if frame.time is not None and frame.time + 1e-3 < start:
                continue
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) == n:
                break
    return frames


def signals(df):
    state = np.stack(df['observation.state'].values).astype(np.float64)
    rpm = state[:, 18:22]
    speed = M_PER_RPM * (rpm[:, 0] + rpm[:, 1]) / 2.0
    diff = rpm[:, 0] - rpm[:, 1]
    return speed, diff


def image_shift(a, b):
    """Horizontal shift (px, 640-wide stretched frame) of the upper scene between two frames: mostly rotation."""
    ga = cv2.cvtColor(a[40:230], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b[40:230], cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ga, gb, window)
    return dx, response


def calibrate(data, limit=40):
    """Yaw rate (deg/s, + = left) per unit of wheel difference, from the camera's horizontal image shift."""
    deg_per_px = HFOV / 640.0
    xs, ys = [], []
    for k, (ep, df, video, start) in enumerate(episodes(data)):
        if k >= limit:
            break
        frames = decode(video, start, len(df))
        speed, diff = signals(df)
        for i in range(0, min(len(frames), len(df)) - 1):
            dx, response = image_shift(frames[i], frames[i + 1])
            if response < 0.15 or speed[i] < 0.2:
                continue
            yaw_rate = dx * deg_per_px * FPS  # content moving right = the rover turned left
            xs.append(diff[i]); ys.append(yaw_rate)
    xs, ys = np.array(xs), np.array(ys)
    keep = np.abs(ys) < 120
    scale = float((xs[keep] * ys[keep]).sum() / (xs[keep] ** 2).sum())
    corr = float(np.corrcoef(xs[keep], ys[keep])[0, 1])
    return scale, corr, int(keep.sum())


def odometry(speed, yaw_rate_deg, i, n):
    """Poses (x forward, y left, heading rad) in the frame-i robot frame, until PATH_M or HORIZON_S."""
    x = y = th = 0.0
    poses, length = [(0.0, 0.0, 0.0)], 0.0
    for k in range(i, min(n - 1, i + int(HORIZON_S * FPS))):
        v = speed[k] / FPS
        th += math.radians(yaw_rate_deg[k]) / FPS
        x += v * math.cos(th); y += v * math.sin(th); length += abs(v)
        poses.append((x, y, th))
        if length >= PATH_M:
            break
    return np.array(poses), length


def project(points):
    """Ground points (x forward, y left) in the rover frame to pixels; None when behind/under the camera."""
    xc = points[:, 0] - CAM_FWD
    ok = xc > 0.22
    u = W / 2 + F * (-points[:, 1]) / np.where(ok, xc, 1)
    v = H / 2 + F * CAM_H / np.where(ok, xc, 1)
    return np.stack([u, v], 1), ok


def footprint(poses):
    mask = np.zeros((H, W), np.uint8)
    half = ROVER_WIDTH / 2
    left = poses[:, :2] + half * np.stack([-np.sin(poses[:, 2]), np.cos(poses[:, 2])], 1)
    right = poses[:, :2] - half * np.stack([-np.sin(poses[:, 2]), np.cos(poses[:, 2])], 1)
    pl, okl = project(left)
    pr, okr = project(right)
    for k in range(len(poses) - 1):
        if not (okl[k] and okl[k + 1] and okr[k] and okr[k + 1]):
            continue
        quad = np.array([pl[k], pl[k + 1], pr[k + 1], pr[k]], np.float64)
        quad = np.clip(quad, -4 * W, 4 * W).round().astype(np.int32)
        cv2.fillConvexPoly(mask, quad, 255)
    return mask


def heading_at(poses, metres):
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poses[:, :2], axis=0).T))])
    k = int(np.searchsorted(d, metres))
    if k >= len(poses):
        return None
    x, y = poses[k, :2]
    return math.degrees(math.atan2(-y, x))  # + = right, like GeNIE's waypoint_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--yaw-scale', type=float, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'footprint'), exist_ok=True)
    if args.yaw_scale is None:
        scale, corr, n = calibrate(args.data)
        print(f'yaw scale {scale:.4f} deg/s per rpm difference (corr {corr:.3f}, n {n})', flush=True)
    else:
        scale = args.yaw_scale
    meta = open(os.path.join(args.out, 'meta.jsonl'), 'w')
    total = 0
    for ep, df, video, start in episodes(args.data):
        frames = decode(video, start, len(df))
        speed, diff = signals(df)
        yaw = scale * diff
        pos = np.stack(df['extra.position_m'].values).astype(np.float64)
        goal = np.stack(df['extra.goal_m'].values).astype(np.float64)
        bot = int(df['extra.bot_index'].iloc[0])
        valid = np.stack(df['extra.camera_valid'].values)[:, 0]
        n = min(len(frames), len(df))
        split = 'val' if ep % 7 == 3 else 'train'
        for i in range(0, n - 20, STEP):
            if valid[i] < 0.5 or speed[i:i + 20].min() < -0.05:
                continue
            poses, length = odometry(speed, yaw, i, n)
            if length < 1.5:
                continue
            mask = footprint(poses)
            if mask.sum() < 255 * 400:
                continue
            name = f'{ep:03d}_{i:04d}'
            image = cv2.resize(frames[i], (W, H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(args.out, 'images', name + '.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(args.out, 'footprint', name + '.png'), mask)
            # Goal direction from GPS (east/north metres); heading from the GPS course of the next 2 s when moving.
            meta.write(json.dumps({'name': name, 'episode': ep, 'frame': i, 'bot': bot, 'split': split,
                                   'path_m': round(length, 2), 'human_deg_1m': heading_at(poses, 1.0),
                                   'human_deg_2m': heading_at(poses, 2.0), 'pos_m': pos[i].round(2).tolist(),
                                   'goal_m': goal[i].round(2).tolist(),
                                   'course_deg': (math.degrees(math.atan2(*(pos[min(n - 1, i + 20)] - pos[i])))
                                                  if np.hypot(*(pos[min(n - 1, i + 20)] - pos[i])) > 1.0 else None)}) + '\n')
            total += 1
        print(f'episode {ep}: {total} samples so far', flush=True)
    meta.close()
    print('done', total, flush=True)


if __name__ == '__main__':
    main()
