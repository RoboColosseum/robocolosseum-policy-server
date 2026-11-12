"""GeNIE's perception and path planning: SAM-TP -> ground-plane BEV -> path bank -> clusters -> path.

``sam2`` and ``genie_path_planner`` come from https://github.com/jiaming-ai/GENIE-SAMTP (installed from git at a
pinned commit) and are used as released: SAM-TP inference, the ground projection, the seeded polynomial path
bank, high-cost filtering, adaptive k-means path clusters with the goal-closest group, footprint costs and the
top-k path average.

One piece of the paper is not in the release: its path cost is ``C(l) = sum f(p_i) + beta * sum g(p_i)``, with
``g`` a goal-directed term, while the released planner scores paths on traversability only. ``plan`` adds that
term (``beta = 1``, the value the paper reports; ``g`` = a point's angular distance from the goal direction,
divided by 180 degrees). Without it, the path in an open plaza where everything is traversable is arbitrary.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import MiniPlusCamera

# Some SAM 2 ops have no MPS kernel; must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from genie_path_planner.costs import compute_paths_costs  # noqa: E402
from genie_path_planner.geometry import bev_pixel_to_xy, goal_xy_to_bev_pixel  # noqa: E402
from genie_path_planner.path_sampling import sample_paths_polynomial  # noqa: E402
from genie_path_planner.path_selection import (  # noqa: E402
    adaptive_kmeans,
    filter_paths_with_high_costs,
    merge_centroids_by_angle,
    select_best_group_by_closest_path_angle,
)
from genie_path_planner.planner import (  # noqa: E402
    PlannerConfig,
    _draw_plan_overlay,
    _planner_path_to_bev_path,
    _resize_float_map,
    _resize_mask,
    _resize_pixel,
    _smooth_cost,
    cost_to_vis,
    traversability_to_cost,
)
from genie_path_planner.projection import logits_to_traversability, project_score_to_bev  # noqa: E402


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SamTP:
    """The SAM-TP traversability model (SAM 2.1 Hiera-tiny with GeNIE's learned traversability prompt)."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        config = Path(sam2.__file__).parent / "configs" / "sam2.1_inference_tiny" / "sam2.1_custom2.yaml"
        self.device = pick_device(device)
        self._predictor = SAM2ImagePredictor(build_sam2(str(config), str(checkpoint), device=self.device), mask_threshold=0.0)
        self._lock = threading.Lock()

    def logits(self, rgb: np.ndarray) -> np.ndarray:
        """Per-pixel traversability logits for an (H, W, 3) uint8 RGB frame.

        Same call as ``sam2.sam_tp.SAM_TP.run_sam2_inference``: the bottom-left, bottom-right and bottom-centre
        points (the learned prompt replaces them in the custom prompt encoder), single mask, raw logits.
        """
        height, width = rgb.shape[:2]
        points = np.array([[0, height - 1], [width - 1, height - 1], [(width - 1) // 2, height - 1]], np.float32)
        with self._lock:
            self._predictor.reset_predictor()
            self._predictor.set_image(rgb)
            masks, _, _ = self._predictor.predict(point_coords=points, point_labels=np.ones(3, np.int32),
                                                  multimask_output=False, return_logits=True, normalize_coords=False)
        return np.asarray(masks[0], np.float32)


@dataclass(frozen=True)
class BevSettings:
    """Local map for the Mini+: 5 m ahead, 3 m to each side, 5 cm cells."""

    resolution_m: float = 0.05
    forward_m: float = 5.0
    side_m: float = 3.0
    max_ray_m: float = 6.0


def planner_config(bev: BevSettings, footprint_m: float = 0.30) -> PlannerConfig:
    """GeNIE's live-planner settings (configs/stretch_path_planner.yaml) with a Mini+-sized footprint."""
    grid = 240
    metres_per_px = max(bev.forward_m, 2.0 * bev.side_m) / grid
    return PlannerConfig(grid_size=grid, unknown_cost=0.2, smooth_kernel=3, num_goals=30, num_mid_points_per_goal=20,
                         path_num_samples=100, footprint_px=max(4, round(footprint_m / metres_per_px)),
                         threshold_cost=0.5, threshold_points_ratio=0.05, number_of_points_to_filter=60, alpha=1.0,
                         best_k=12, use_clustering=True, max_clusters=4, cluster_angle_threshold_deg=40.0, random_seed=42,
                         include_goal_in_path_bank=False, include_random_goals=True)


@dataclass
class Plan:
    status: str
    traversability: np.ndarray
    bev: np.ndarray
    observed: np.ndarray
    path_xy: np.ndarray
    waypoint_xy: tuple[float, float] | None
    waypoint_deg: float | None
    front_clear: float
    goal_xy: tuple[float, float]
    visualization: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def clip_goal_to_map(goal_x: float, goal_y: float, bev: BevSettings) -> tuple[float, float]:
    """Scale a far goal back along its direction so it lands inside the local map.

    ``goal_xy_to_bev_pixel`` clips rows and columns independently, which bends the direction of a far goal.
    """
    margin = 0.98
    scale = 1.0
    if goal_y > bev.forward_m * margin:
        scale = min(scale, bev.forward_m * margin / goal_y)
    if abs(goal_x) > bev.side_m * margin:
        scale = min(scale, bev.side_m * margin / abs(goal_x))
    return goal_x * scale, max(0.0, goal_y * scale)


def front_clearance(bev: np.ndarray, observed: np.ndarray, settings: BevSettings, depth_m: float = 0.8,
                    half_width_m: float = 0.2) -> float:
    """Mean traversability of the strip straight ahead (unknown cells count as 0.5)."""
    rows = max(1, int(depth_m / settings.resolution_m))
    half = max(1, int(half_width_m / settings.resolution_m))
    centre = bev.shape[1] // 2
    strip = bev[bev.shape[0] - rows:, centre - half:centre + half]
    seen = observed[bev.shape[0] - rows:, centre - half:centre + half].astype(bool)
    values = np.where(seen & (strip >= 0), strip, 0.5)
    return float(values.mean())


def waypoint_along(path_xy: np.ndarray, lookahead_m: float) -> tuple[float, float] | None:
    """The point ``lookahead_m`` along the path (the controller's next waypoint)."""
    if len(path_xy) < 2:
        return None
    steps = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(steps)])
    index = int(np.searchsorted(travelled, lookahead_m))
    point = path_xy[min(index, len(path_xy) - 1)]
    return float(point[0]), float(point[1])


class GeniePlanner:
    def __init__(self, model: SamTP, camera: MiniPlusCamera | None = None, bev: BevSettings | None = None,
                 lookahead_m: float = 1.0, goal_weight: float = 1.0) -> None:
        self.model = model
        self.camera = camera or MiniPlusCamera()
        self.bev = bev or BevSettings()
        self.config = planner_config(self.bev)
        self.lookahead_m = lookahead_m
        self.goal_weight = goal_weight
        grid = int(self.config.grid_size)
        self.bev_shape = (int(np.ceil(self.bev.forward_m / self.bev.resolution_m)),
                          int(np.ceil(2.0 * self.bev.side_m / self.bev.resolution_m)))
        self.start = _resize_pixel((self.bev_shape[0] - 1, self.bev_shape[1] // 2), self.bev_shape, grid)
        # Metres per planner-grid pixel along rows (forward) and columns (right).
        self.row_m = self.bev.forward_m / (grid - 1)
        self.col_m = 2.0 * self.bev.side_m / (grid - 1)
        # GeNIE's path bank is fixed (seeded, independent of the goal), so sample it once instead of per frame.
        self.path_bank = sample_paths_polynomial(robot=self.start, num_goals=int(self.config.num_goals),
                                                 num_mid_points_per_goal=int(self.config.num_mid_points_per_goal),
                                                 num_samples=int(self.config.path_num_samples), grid_size=grid, goal=None,
                                                 include_random_goals=bool(self.config.include_random_goals),
                                                 random_seed=self.config.random_seed)

    def goal_cost(self, path_rc: np.ndarray, goal_deg: float, min_range_m: float = 0.3) -> float:
        """The paper's goal term: summed angular distance of the path's points from the goal direction (0..1 each)."""
        forward = (self.start[0] - path_rc[:, 0]) * self.row_m
        right = (path_rc[:, 1] - self.start[1]) * self.col_m
        far = np.hypot(forward, right) >= min_range_m
        angles = np.degrees(np.arctan2(right[far], forward[far]))
        return float(np.sum(np.abs((angles - goal_deg + 180.0) % 360.0 - 180.0) / 180.0))

    def plan(self, rgb: np.ndarray, goal_x: float, goal_y: float, *, visualize: bool = False) -> Plan:
        cfg = self.config
        grid = int(cfg.grid_size)
        traversability = logits_to_traversability(self.model.logits(rgb), "sigmoid")
        height, width = traversability.shape
        bev, observed, _ = project_score_to_bev(traversability, self.camera.intrinsics(width, height), self.camera.pose(),
                                                ground_z=0.0, bev_resolution_m_per_px=self.bev.resolution_m,
                                                bev_forward_range_m=self.bev.forward_m, bev_side_range_m=self.bev.side_m,
                                                max_ray_distance_m=self.bev.max_ray_m)
        goal = clip_goal_to_map(goal_x, goal_y, self.bev)
        goal_deg = math.degrees(math.atan2(goal_x, max(goal_y, 1e-6))) if goal_y > 0 or goal_x else 0.0
        known = np.isfinite(bev) & (bev >= 0.0) & observed.astype(bool)
        cost = _resize_float_map(_smooth_cost(traversability_to_cost(bev, unknown_cost=float(cfg.unknown_cost)),
                                              int(cfg.smooth_kernel)), grid)
        planner_known = _resize_mask(known, grid)
        planner_goal = _resize_pixel(goal_xy_to_bev_pixel(goal[0], goal[1], bev.shape, self.bev.resolution_m), bev.shape, grid)
        front = front_clearance(bev, observed, self.bev)
        filtered = filter_paths_with_high_costs(self.path_bank, cost, num_points=int(min(cfg.number_of_points_to_filter, cfg.path_num_samples + 1)),
                                                footprint_px=int(cfg.footprint_px), threshold_points_ratio=float(cfg.threshold_points_ratio),
                                                threshold_cost=float(cfg.threshold_cost))
        selected = list(filtered)
        if len(filtered) >= 2 and cfg.use_clustering:
            labels, centroids, _ = adaptive_kmeans(filtered, min_clusters=1, max_clusters=int(cfg.max_clusters))
            if len(centroids) >= 2:
                _, groups, _ = merge_centroids_by_angle(centroids, angle_threshold_deg=float(cfg.cluster_angle_threshold_deg))
                robot_goal_angle = math.degrees(math.atan2(planner_goal[0] - self.start[0], planner_goal[1] - self.start[1]))
                best, _ = select_best_group_by_closest_path_angle(filtered, labels, groups, robot_goal_angle, lookahead_index=30)
                selected = [path for path, label in zip(filtered, labels) if int(label) in groups[best]] or selected
        final_rc = np.empty((0, 2), np.float32)
        if selected:
            scored = [(traversal + self.goal_weight * self.goal_cost(path, goal_deg), path) for traversal, path in
                      compute_paths_costs(selected, cost, alpha=float(cfg.alpha), footprint_px=int(cfg.footprint_px))]
            final_rc = self._pick(scored, cost, goal_deg)
        status = "ok" if len(final_rc) else "no_valid_paths_after_filtering"
        path_xy = np.empty((0, 2))
        if len(final_rc):
            bev_rc = _planner_path_to_bev_path(final_rc, src_shape=bev.shape, grid_size=grid)
            x_right, y_forward = bev_pixel_to_xy(bev_rc[:, 0], bev_rc[:, 1], bev.shape, self.bev.resolution_m)
            path_xy = np.stack([x_right, y_forward], axis=1).astype(np.float64)
        waypoint = waypoint_along(path_xy, self.lookahead_m)
        waypoint_deg = None if waypoint is None else math.degrees(math.atan2(waypoint[0], max(waypoint[1], 1e-3)))
        visualization = None
        if visualize:
            visualization = _draw_plan_overlay(cost_to_vis(cost, known_mask=planner_known), all_paths=self.path_bank,
                                               selected_paths=selected, final_rows=final_rc[:, 0] if len(final_rc) else np.array([]),
                                               final_cols=final_rc[:, 1] if len(final_rc) else np.array([]),
                                               start_rc=self.start, goal_rc=planner_goal)
        return Plan(status=status, traversability=traversability, bev=bev, observed=observed, path_xy=path_xy,
                    waypoint_xy=waypoint, waypoint_deg=waypoint_deg, front_clear=front, goal_xy=goal,
                    visualization=visualization,
                    metadata={"filtered_paths": len(filtered), "selected_paths": len(selected), "goal_deg": round(goal_deg, 1)})

    def _pick(self, scored: list[tuple[float, np.ndarray]], cost: np.ndarray, goal_deg: float) -> np.ndarray:
        """``genie_path_planner.costs.pick_final_path`` with the goal term in every cost it compares."""
        top = sorted(scored, key=lambda item: item[0])[: max(1, int(self.config.best_k))]
        weights = np.array([item[0] for item in top], np.float64)
        weights = np.ones_like(weights) if weights.sum() <= 1e-8 else weights
        weights /= weights.sum()
        averaged = np.einsum("k,kij->ij", weights, np.stack([np.asarray(path, np.float64) for _, path in top]))
        traversal = compute_paths_costs([averaged], cost, alpha=float(self.config.alpha), footprint_px=int(self.config.footprint_px))[0][0]
        if traversal + self.goal_weight * self.goal_cost(averaged, goal_deg) > top[-1][0]:
            return np.asarray(top[0][1], np.float32)
        return averaged.astype(np.float32)
