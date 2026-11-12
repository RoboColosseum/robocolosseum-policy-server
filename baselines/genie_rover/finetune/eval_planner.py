"""GeNIE's planner on held-out Wuhan frames with two SAM-TP checkpoints (BIT-3406).

For each held-out frame with a GPS course: the goal is checkpoint 2 relative to the course; GeNIE plans exactly as
on the rover. Compared per checkpoint:
  - clearance: the planned path's closest approach (0.3-1.5 m ahead) to labelled obstacles within 1.8 m;
  - heading error: the planned waypoint (1 m) against the human's heading 1 m along the path actually driven;
  - recoveries: plans with no path or a blocked front strip (the controller would back up).

    python eval_planner.py prepared prepared/labels weights/checkpoint_2.pt runs/ft/best.pt out.json [max_frames]
"""
import json, math, os, sys

import cv2
import numpy as np

from genie_rover.geo import goal_in_robot_frame, wrap_deg  # noqa: E402
from genie_rover.planning import GeniePlanner, SamTP  # noqa: E402
from refine import IX, IY, NEAR_PATH_M, NX, NY, RES, Y_MAX, obstacles  # noqa: E402

root, labels, ckpt_a, ckpt_b, out = sys.argv[1:6]
limit = int(sys.argv[6]) if len(sys.argv) > 6 else 10**9
rows = [json.loads(l) for l in open(os.path.join(root, 'meta.jsonl'))]
rows = [r for r in rows if r['split'] == 'val' and r['course_deg'] is not None and r['human_deg_1m'] is not None
        and os.path.exists(os.path.join(labels, r['name'] + '.png'))]
rows = rows[:: max(1, len(rows) // limit)][:limit]
planners = {'original': GeniePlanner(SamTP(ckpt_a, 'cuda')), 'finetuned': GeniePlanner(SamTP(ckpt_b, 'cuda'))}


def clearance(path_xy, label):
    """Closest approach (m) of the path's 0.3-1.5 m section to near obstacles (refine.obstacles), on the labels' grid."""
    from scipy.ndimage import distance_transform_edt
    occ = np.zeros((NX, NY), bool)
    obstacle = obstacles(label)
    occ[IX[obstacle], IY[obstacle]] = True
    if not occ.any() or len(path_xy) == 0:
        return math.inf
    edt = distance_transform_edt(~occ) * RES
    fwd, left = path_xy[:, 1], -path_xy[:, 0]
    keep = (fwd > 0.3) & (fwd < NEAR_PATH_M) & (np.abs(left) < Y_MAX)
    if not keep.any():
        return math.inf
    ix = np.clip((fwd[keep] / RES).astype(int), 0, NX - 1)
    iy = np.clip(((left[keep] + Y_MAX) / RES).astype(int), 0, NY - 1)
    return float(edt[ix, iy].min())


results = {k: [] for k in planners}
for n, r in enumerate(rows):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(root, 'images', r['name'] + '.jpg')), cv2.COLOR_BGR2RGB)
    label = cv2.imread(os.path.join(labels, r['name'] + '.png'), cv2.IMREAD_GRAYSCALE)
    pos, goal = np.array(r['pos_m']), np.array(r['goal_m'])
    bearing = math.degrees(math.atan2(*(goal - pos)))
    distance = float(np.hypot(*(goal - pos)))
    rel = wrap_deg(bearing - r['course_deg'])
    gx, gy = goal_in_robot_frame(max(distance, 3.0), rel)
    for key, planner in planners.items():
        plan = planner.plan(rgb, gx, gy)
        wp = plan.waypoint_deg
        results[key].append({'name': r['name'], 'status': plan.status, 'front_clear': round(plan.front_clear, 3),
                             'waypoint_deg': None if wp is None else round(wp, 1),
                             'heading_err': None if wp is None else round(abs(wrap_deg(wp - r['human_deg_1m'])), 1),
                             'clearance_m': clearance(plan.path_xy, label)})
    if n % 50 == 0:
        print(n, len(rows), flush=True)

summary = {}
for key, items in results.items():
    errs = np.array([i['heading_err'] for i in items if i['heading_err'] is not None])
    clear = np.array([i['clearance_m'] for i in items])
    finite = clear[np.isfinite(clear)]
    recover = np.mean([(i['waypoint_deg'] is None) or i['front_clear'] < 0.45 for i in items])
    summary[key] = {'frames': len(items), 'recovery_rate': round(float(recover), 3),
                    'heading_err_median_deg': round(float(np.median(errs)), 1), 'heading_within_15deg': round(float((errs <= 15).mean()), 3),
                    'clearance_median_m': round(float(np.median(finite)), 3) if len(finite) else None,
                    'paths_within_0.3m_of_obstacle': round(float((finite < 0.3).sum() / max(1, len(items))), 3),
                    'paths_within_0.5m_of_obstacle': round(float((finite < 0.5).sum() / max(1, len(items))), 3)}
json.dump({'summary': summary, 'frames': results}, open(out, 'w'), indent=1, default=str)
print(json.dumps(summary, indent=1), flush=True)
