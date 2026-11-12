"""Soft training targets with the clearance the human drivers kept from obstacles (BIT-3406).

Every label pixel below the horizon is put on the ground plane with GeNIE's camera model. The distance from each
ground pixel to the nearest non-traversable ground point (curb, wall, step, bollard...) is measured on a bird's-eye
grid. Pass 1 measures how close the driven path came to obstacles in each frame; the margin is a low percentile
of that clearance. Pass 2 writes the targets:

  0       not traversable
  254     traversable: the driven strip (always), and ground farther than the margin from any obstacle
  soft    ground inside the margin that no human drove on: FLOOR at the obstacle edge rising to 1.0 at the margin
  255     unknown (the loss keeps the original model's answer)

    python refine.py prepared prepared/labels prepared/targets [percentile] [floor]
"""
import json, math, os, sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

H, W = 288, 512
HFOV, CAM_H, CAM_FWD = 92.6, 0.14, 0.11
F = (W / 2) / math.tan(math.radians(HFOV / 2))
RES, X_MAX, Y_MAX = 0.04, 8.0, 5.0
NEAR_OBSTACLE_M, NEAR_PATH_M = 1.8, 1.5
FLOOR = 0.2  # target at the obstacle edge inside the margin
NX, NY = int(X_MAX / RES), int(2 * Y_MAX / RES)

v = np.arange(H, dtype=np.float64)[:, None] + 0.5
u = np.arange(W, dtype=np.float64)[None, :] + 0.5
below = v > H / 2 + 2
x_cam = np.where(below, F * CAM_H / np.maximum(v - H / 2, 1e-6), 1e6)
x_cam = np.broadcast_to(x_cam, (H, W)).copy()
X = x_cam + CAM_FWD
Y = -(u - W / 2) * x_cam / F
ground = below & (X < X_MAX) & (np.abs(Y) < Y_MAX)
IX = np.clip((X / RES).astype(int), 0, NX - 1)
IY = np.clip(((Y + Y_MAX) / RES).astype(int), 0, NY - 1)


def obstacles(label):
    """Non-traversable ground regions that are real obstacles: within 1.8 m, not the vignetted image borders, not
    thin paving joints (opened away), at least 150 px."""
    o = ((label == 0) & ground).astype(np.uint8)
    o[:, :24] = 0
    o[:, W - 24:] = 0
    o[X > NEAR_OBSTACLE_M] = 0  # near the horizon a 0.14 m-high camera cannot place things on the ground reliably
    o = cv2.morphologyEx(o, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(o, connectivity=8)
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 150)
    big = big[big > 0]
    return np.isin(comp, big)


def distance_map(label):
    """Metres from each ground pixel to the nearest obstacle on the ground plane (inf where none within range)."""
    occ = np.zeros((NX, NY), bool)
    obstacle = obstacles(label)
    occ[IX[obstacle], IY[obstacle]] = True
    if not occ.any():
        return np.full((H, W), np.inf)
    edt = distance_transform_edt(~occ) * RES
    dist = np.full((H, W), np.inf)
    dist[ground] = edt[IX[ground], IY[ground]]
    return dist


def main():
    root, labels, out = sys.argv[1], sys.argv[2], sys.argv[3]
    percentile = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
    global FLOOR
    FLOOR = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(root, 'meta.jsonl'))]
    rows = [r for r in rows if os.path.exists(os.path.join(labels, r['name'] + '.png'))]
    clearances = []
    for r in rows:
        label = cv2.imread(os.path.join(labels, r['name'] + '.png'), cv2.IMREAD_GRAYSCALE)
        foot = cv2.imread(os.path.join(root, 'footprint', r['name'] + '.png'), cv2.IMREAD_GRAYSCALE) > 127
        dist = distance_map(label)
        near = foot & ground & (X > 0.3) & (X < NEAR_PATH_M)
        c = float(dist[near].min()) if near.any() else math.inf
        if c < 2.5:
            clearances.append(c)
    clearances = np.array(clearances)
    stats = {p: round(float(np.percentile(clearances, p)), 3) for p in (5, 10, 25, 50, 75, 90)}
    margin = float(np.percentile(clearances, percentile))
    print(f'human clearance (m) over {len(clearances)} frames with an obstacle within 2.5 m: {stats}; margin {margin:.2f} m', flush=True)
    counts = {'frames': 0, 'margin_px': 0, 'traversable_px': 0}
    for r in rows:
        label = cv2.imread(os.path.join(labels, r['name'] + '.png'), cv2.IMREAD_GRAYSCALE)
        foot = cv2.imread(os.path.join(root, 'footprint', r['name'] + '.png'), cv2.IMREAD_GRAYSCALE) > 127
        dist = distance_map(label)
        target = np.full((H, W), 255, np.uint8)
        target[label == 0] = 0
        trav = label == 255
        target[trav] = 254
        soft = trav & ~foot & (dist < margin)
        target[soft] = np.round(254 * (FLOOR + (1 - FLOOR) * dist[soft] / margin)).astype(np.uint8).clip(int(254 * FLOOR), 253)
        cv2.imwrite(os.path.join(out, r['name'] + '.png'), target)
        counts['frames'] += 1; counts['margin_px'] += int(soft.sum()); counts['traversable_px'] += int(trav.sum())
    json.dump({'clearance_percentiles_m': stats, 'margin_m': margin, 'percentile': percentile, **counts},
              open(os.path.join(out, 'margin.json'), 'w'), indent=1)
    print('done', counts, flush=True)


if __name__ == '__main__':
    main()
