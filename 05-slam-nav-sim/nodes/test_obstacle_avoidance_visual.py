#!/usr/bin/env python3
# ruff: noqa: UP006,UP007
from __future__ import annotations
"""Standalone visual test — obstacle avoidance demo in Rerun.

Demonstrates all 5 avoidance scenarios without requiring DORA daemon or MuJoCo.
All core math is inlined from the actual node files.

Run:
    python3 python/test_obstacle_avoidance_visual.py
    # or from python/ directory:
    python3 test_obstacle_avoidance_visual.py
"""

import math
import os
import struct
import sys
import time
from typing import Optional

import numpy as np
import rerun as rr
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Waypoints loading — tries multiple candidate paths
# ---------------------------------------------------------------------------

def load_waypoints(override_path: Optional[str] = None) -> np.ndarray:
    """Load Waypoints.txt → (N, 2) float64 array of [x, y] pairs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        override_path,
        os.path.join(script_dir, "..", "map", "pub_road", "Waypoints.txt"),
        os.path.join(script_dir, "map", "pub_road", "Waypoints.txt"),
        "map/pub_road/Waypoints.txt",
    ]
    for path in candidates:
        if path is None:
            continue
        path = os.path.normpath(path)
        if os.path.isfile(path):
            print(f"[waypoints] Loading from {path}")
            pts = []
            with open(path) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pts.append([float(parts[0]), float(parts[1])])
            arr = np.array(pts, dtype=np.float64)
            print(f"[waypoints] Loaded {len(arr)} waypoints")
            return arr

    # No file found — generate a synthetic oval track so the demo still runs
    print("[waypoints] WARNING: Waypoints.txt not found — using synthetic 80m oval")
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    xs = 40.0 * np.cos(t)
    ys = 20.0 * np.sin(t)
    return np.stack([xs, ys], axis=1)


# ---------------------------------------------------------------------------
# Frenet math  (inlined from planning_node.py / road_lane_publisher_node.py)
# ---------------------------------------------------------------------------

TRANS_PARA_BACK = 0.25
NUM_PATH_POINTS = 30


def compute_cumulative_s(maps_x: np.ndarray, maps_y: np.ndarray) -> np.ndarray:
    s = np.zeros(len(maps_x), dtype=np.float64)
    for i in range(1, len(maps_x)):
        dx = maps_x[i] - maps_x[i - 1]
        dy = maps_y[i] - maps_y[i - 1]
        s[i] = s[i - 1] + math.sqrt(dx * dx + dy * dy)
    return s


def _next_waypoint(x: float, y: float, maps_x: np.ndarray, maps_y: np.ndarray) -> int:
    dx = maps_x - x
    dy = maps_y - y
    dists = dx * dx + dy * dy
    closest = int(np.argmin(dists))
    n = len(maps_x)
    if closest < n - 1:
        seg_x = maps_x[closest + 1] - maps_x[closest]
        seg_y = maps_y[closest + 1] - maps_y[closest]
        pt_x = x - maps_x[closest]
        pt_y = y - maps_y[closest]
        if pt_x * seg_x + pt_y * seg_y > 0:
            return closest + 1
    return closest


def get_frenet(
    x: float, y: float,
    maps_x: np.ndarray, maps_y: np.ndarray, maps_s: np.ndarray,
) -> tuple[float, float]:
    n = len(maps_x)
    if n < 2:
        return 0.0, 0.0
    nwp = _next_waypoint(x, y, maps_x, maps_y)
    if nwp == 0:
        nwp = 1
    prev = nwp - 1
    line_x = maps_x[nwp] - maps_x[prev]
    line_y = maps_y[nwp] - maps_y[prev]
    qx = x - maps_x[prev]
    qy = y - maps_y[prev]
    line_len_sq = line_x * line_x + line_y * line_y
    proj_norm = (qx * line_x + qy * line_y) / line_len_sq if line_len_sq >= 1e-12 else 0.0
    proj_x = proj_norm * line_x
    proj_y = proj_norm * line_y
    cross = qx * line_y - qy * line_x
    sign = 1.0 if cross > 0 else -1.0
    frenet_d = sign * math.sqrt((qx - proj_x) ** 2 + (qy - proj_y) ** 2)
    frenet_s = maps_s[prev] + math.sqrt(proj_x * proj_x + proj_y * proj_y)
    return frenet_s, frenet_d


def get_xy_from_frenet(
    s_val: float, d_val: float,
    maps_s: np.ndarray, maps_x: np.ndarray, maps_y: np.ndarray,
) -> tuple[float, float]:
    n = len(maps_s)
    if n < 2:
        return 0.0, 0.0
    if s_val <= maps_s[0]:
        s_val = maps_s[0] + 0.001
    if s_val >= maps_s[-1]:
        return float(maps_x[-1]), float(maps_y[-1])
    prev_wp = 0
    for i in range(n - 1):
        if maps_s[i] <= s_val < maps_s[i + 1]:
            prev_wp = i
            break
    else:
        prev_wp = n - 2
    nxt = prev_wp + 1
    heading = math.atan2(maps_y[nxt] - maps_y[prev_wp], maps_x[nxt] - maps_x[prev_wp])
    seg_s = s_val - maps_s[prev_wp]
    seg_x = maps_x[prev_wp] + seg_s * math.cos(heading)
    seg_y = maps_y[prev_wp] + seg_s * math.sin(heading)
    perp = heading - math.pi / 2.0
    x = seg_x + d_val * math.cos(perp)
    y = seg_y + d_val * math.sin(perp)
    return x, y


def _heading_at_s(s_val: float, maps_s: np.ndarray, maps_x: np.ndarray, maps_y: np.ndarray) -> float:
    """Tangent heading (radians) along the reference line at arc-length s_val."""
    n = len(maps_s)
    prev_wp = 0
    for i in range(n - 1):
        if maps_s[i] <= s_val < maps_s[i + 1]:
            prev_wp = i
            break
    else:
        prev_wp = n - 2
    nxt = prev_wp + 1
    return math.atan2(maps_y[nxt] - maps_y[prev_wp], maps_x[nxt] - maps_x[prev_wp])


def plan_path(
    cur_x: float, cur_y: float, cur_theta: float,
    cur_s: float, cur_d: float,
    maps_s: np.ndarray, maps_x: np.ndarray, maps_y: np.ndarray,
    speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    vel_mps = speed * 0.00277
    plan_dist = 15.0 + vel_mps * 0.277
    rear_x = cur_x - TRANS_PARA_BACK * math.cos(cur_theta)
    rear_y = cur_y - TRANS_PARA_BACK * math.sin(cur_theta)
    map_end_s = maps_s[-1] if len(maps_s) > 0 else 0.0

    path_x_global = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    path_y_global = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    for i in range(NUM_PATH_POINTS):
        target_s = cur_s + i * (plan_dist / 50.0)
        if target_s >= map_end_s:
            path_x_global[i] = float(maps_x[-1])
            path_y_global[i] = float(maps_y[-1])
        else:
            gx, gy = get_xy_from_frenet(target_s, 0.0, maps_s, maps_x, maps_y)
            path_x_global[i] = gx
            path_y_global[i] = gy

    sin_yaw = math.sin(cur_theta)
    cos_yaw = math.cos(cur_theta)
    path_x_veh = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    path_y_veh = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    for i in range(NUM_PATH_POINTS):
        shift_x = path_x_global[i] - rear_x
        shift_y = path_y_global[i] - rear_y
        path_x_veh[i] = shift_x * sin_yaw - shift_y * cos_yaw
        path_y_veh[i] = shift_y * sin_yaw + shift_x * cos_yaw

    return path_x_veh, path_y_veh


def plan_path_with_d(
    cur_x: float, cur_y: float, cur_theta: float,
    cur_s: float, cur_d: float, target_d: float,
    maps_s: np.ndarray, maps_x: np.ndarray, maps_y: np.ndarray,
    speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    vel_mps = speed * 0.00277
    plan_dist = 15.0 + vel_mps * 0.277
    rear_x = cur_x - TRANS_PARA_BACK * math.cos(cur_theta)
    rear_y = cur_y - TRANS_PARA_BACK * math.sin(cur_theta)
    map_end_s = maps_s[-1] if len(maps_s) > 0 else 0.0

    transition_s = 3.0
    pts_s = [cur_s, cur_s + transition_s * 0.5, cur_s + transition_s, cur_s + plan_dist]
    pts_d = [cur_d, (cur_d + target_d) / 2.0, target_d, target_d]
    d_spline = CubicSpline(pts_s, pts_d, bc_type='clamped')

    path_x_global = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    path_y_global = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    for i in range(NUM_PATH_POINTS):
        target_s = cur_s + i * (plan_dist / 50.0)
        d_val = float(d_spline(target_s)) if target_s <= cur_s + plan_dist else target_d
        if target_s >= map_end_s:
            path_x_global[i] = float(maps_x[-1])
            path_y_global[i] = float(maps_y[-1])
        else:
            gx, gy = get_xy_from_frenet(target_s, d_val, maps_s, maps_x, maps_y)
            path_x_global[i] = gx
            path_y_global[i] = gy

    sin_yaw = math.sin(cur_theta)
    cos_yaw = math.cos(cur_theta)
    path_x_veh = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    path_y_veh = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    for i in range(NUM_PATH_POINTS):
        shift_x = path_x_global[i] - rear_x
        shift_y = path_y_global[i] - rear_y
        path_x_veh[i] = shift_x * sin_yaw - shift_y * cos_yaw
        path_y_veh[i] = shift_y * sin_yaw + shift_x * cos_yaw

    return path_x_veh, path_y_veh


# ---------------------------------------------------------------------------
# Costmap2D  (inlined from costmap_node.py)
# ---------------------------------------------------------------------------

COSTMAP_SIZE_M = 20.0
COSTMAP_RESOLUTION = 0.1
INFLATION_RADIUS = 0.5
HALF_SIZE_M = COSTMAP_SIZE_M / 2.0
GRID_CELLS = int(round(COSTMAP_SIZE_M / COSTMAP_RESOLUTION))


def _build_inflation_kernel(radius_m: float, resolution: float) -> np.ndarray:
    r_cells = int(math.ceil(radius_m / resolution))
    size = 2 * r_cells + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    cy = cx = r_cells
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            d = math.sqrt(dx * dx + dy * dy) * resolution
            if d <= radius_m and d > 0:
                cost = int(round(253.0 * (1.0 - d / radius_m))) + 1
                kernel[cy + dy, cx + dx] = max(1, min(253, cost))
    return kernel


_INFLATION_KERNEL = _build_inflation_kernel(INFLATION_RADIUS, COSTMAP_RESOLUTION)
_KERNEL_HALF = _INFLATION_KERNEL.shape[0] // 2


class Costmap2D:
    """Robot-centred rolling-window occupancy grid."""

    def __init__(self) -> None:
        self._n = GRID_CELLS
        self._res = COSTMAP_RESOLUTION
        self._grid = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.uint8)
        self.origin_x = 0.0
        self.origin_y = 0.0

    def update(self, points_global: np.ndarray, robot_x: float, robot_y: float) -> None:
        self._grid[:] = 0
        self.origin_x = robot_x - HALF_SIZE_M
        self.origin_y = robot_y - HALF_SIZE_M
        if points_global.shape[0] == 0:
            return
        col = ((points_global[:, 0] - self.origin_x) / self._res).astype(np.int32)
        row = ((points_global[:, 1] - self.origin_y) / self._res).astype(np.int32)
        valid = (col >= 0) & (col < self._n) & (row >= 0) & (row < self._n)
        col = col[valid]
        row = row[valid]
        if col.shape[0] == 0:
            return
        unique_cells = np.unique(np.stack([row, col], axis=1), axis=0)
        kh = _KERNEL_HALF
        kernel = _INFLATION_KERNEL
        for r_obs, c_obs in unique_cells:
            r0 = int(r_obs) - kh
            c0 = int(c_obs) - kh
            r1 = r0 + kernel.shape[0]
            c1 = c0 + kernel.shape[1]
            gr0 = max(0, r0)
            gc0 = max(0, c0)
            gr1 = min(self._n, r1)
            gc1 = min(self._n, c1)
            kr0 = gr0 - r0
            kc0 = gc0 - c0
            kr1 = kr0 + (gr1 - gr0)
            kc1 = kc0 + (gc1 - gc0)
            if gr1 <= gr0 or gc1 <= gc0:
                continue
            region = self._grid[gr0:gr1, gc0:gc1]
            np.maximum(region, kernel[kr0:kr1, kc0:kc1], out=region)
        self._grid[row, col] = 254


# ---------------------------------------------------------------------------
# Fake pointcloud generator
# ---------------------------------------------------------------------------

PC_HEADER_SIZE = 16
PC_POINT_SIZE = 16   # float32 x, y, z, intensity


def create_fake_pointcloud(
    obstacle_positions: list[tuple[float, float, float, float, float]],
    robot_x: float, robot_y: float, robot_theta: float,
    pts_per_obstacle: int = 30,
) -> bytes:
    """Synthesize a 16B-header + Nx16B pointcloud in robot-local (sensor) frame.

    obstacle_positions: list of (ox, oy, length, width, height) in global frame.
    Robot theta is in radians.
    """
    all_local_pts: list[tuple[float, float, float, float]] = []

    cos_t = math.cos(-robot_theta)   # inverse rotation: global → local
    sin_t = math.sin(-robot_theta)

    rng = np.random.default_rng(42)

    for ox, oy, length, width, height in obstacle_positions:
        for _ in range(pts_per_obstacle):
            # Random point inside obstacle bounding box in global frame
            gx = ox + rng.uniform(-length / 2.0, length / 2.0)
            gy = oy + rng.uniform(-width / 2.0, width / 2.0)
            gz = rng.uniform(0.2, max(0.3, height))   # above ground

            # Translate to robot-centred, then rotate to local frame
            tx = gx - robot_x
            ty = gy - robot_y
            lx = cos_t * tx - sin_t * ty
            ly = sin_t * tx + cos_t * ty

            all_local_pts.append((lx, ly, gz, 50.0))

    header = b'\x00' * PC_HEADER_SIZE
    point_bytes = b''.join(
        struct.pack('<4f', lx, ly, lz, inten)
        for lx, ly, lz, inten in all_local_pts
    )
    return header + point_bytes


# ---------------------------------------------------------------------------
# Inline obstacle detection  (simplified from obstacle_detector_node.py)
# ---------------------------------------------------------------------------

GROUND_THRESHOLD = 0.15
MAX_DETECT_RANGE = 15.0
CLUSTER_TOLERANCE = 0.5
MIN_CLUSTER_SIZE = 3

_obs_id_counter = 0


def _parse_pointcloud(raw: bytes) -> np.ndarray:
    payload = raw[PC_HEADER_SIZE:]
    n = len(payload) // PC_POINT_SIZE
    if n == 0:
        return np.empty((0, 4), dtype=np.float32)
    return np.frombuffer(payload[: n * PC_POINT_SIZE], dtype=np.float32).reshape(n, 4)


def _euclidean_cluster(xy: np.ndarray, tolerance: float, min_size: int) -> list[np.ndarray]:
    if len(xy) == 0:
        return []
    tree = KDTree(xy)
    visited = np.zeros(len(xy), dtype=bool)
    clusters: list[np.ndarray] = []
    for i in range(len(xy)):
        if visited[i]:
            continue
        queue = list(tree.query_ball_point(xy[i], tolerance))
        visited[i] = True
        cluster_indices: list[int] = []
        head = 0
        while head < len(queue):
            idx = queue[head]
            head += 1
            if visited[idx] and idx != i:
                continue
            visited[idx] = True
            cluster_indices.append(idx)
            for nb in tree.query_ball_point(xy[idx], tolerance):
                if not visited[nb]:
                    queue.append(nb)
                    visited[nb] = True
        if len(cluster_indices) >= min_size:
            clusters.append(np.array(cluster_indices, dtype=np.int32))
    return clusters


def detect_obstacles(
    raw_pc: bytes,
    robot_x: float, robot_y: float, robot_theta_rad: float,
    maps_x: np.ndarray, maps_y: np.ndarray, maps_s: np.ndarray,
) -> list[dict]:
    """Detect obstacles from a raw pointcloud; returns list of obstacle dicts."""
    global _obs_id_counter

    pts = _parse_pointcloud(raw_pc)
    if len(pts) == 0:
        return []

    # Transform from robot-local to global frame
    theta_deg = math.degrees(robot_theta_rad)
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    lx, ly = pts[:, 0], pts[:, 1]
    gx = cos_t * lx - sin_t * ly + robot_x
    gy = sin_t * lx + cos_t * ly + robot_y

    g_pts = np.stack([gx, gy, pts[:, 2], pts[:, 3]], axis=1)

    # Ground removal
    g_pts = g_pts[g_pts[:, 2] > GROUND_THRESHOLD]
    if len(g_pts) == 0:
        return []

    # Distance filter
    dx = g_pts[:, 0] - robot_x
    dy = g_pts[:, 1] - robot_y
    robot_dists = np.sqrt(dx * dx + dy * dy)
    mask = robot_dists <= MAX_DETECT_RANGE
    g_pts = g_pts[mask]
    robot_dists = robot_dists[mask]
    if len(g_pts) == 0:
        return []

    clusters = _euclidean_cluster(g_pts[:, :2], CLUSTER_TOLERANCE, MIN_CLUSTER_SIZE)
    has_lane = len(maps_x) >= 2
    obstacles: list[dict] = []

    for indices in clusters:
        cpts = g_pts[indices]
        cx = float(np.mean(cpts[:, 0]))
        cy = float(np.mean(cpts[:, 1]))
        length = max(0.3, float(np.max(cpts[:, 0]) - np.min(cpts[:, 0])))
        width = max(0.3, float(np.max(cpts[:, 1]) - np.min(cpts[:, 1])))
        height = max(0.3, float(np.max(cpts[:, 2]) - np.min(cpts[:, 2])))
        frenet_s, frenet_d = (
            get_frenet(cx, cy, maps_x, maps_y, maps_s) if has_lane else (0.0, 0.0)
        )
        dist = float(np.min(robot_dists[indices]))
        _obs_id_counter += 1
        obstacles.append({
            "id": _obs_id_counter,
            "x": cx, "y": cy,
            "s": frenet_s, "d": frenet_d,
            "vx": 0.0, "vy": 0.0,
            "length": length, "width": width, "height": height,
            "distance": dist,
        })
    return obstacles


# ---------------------------------------------------------------------------
# Planning logic  (inlined from planning_node.py)
# ---------------------------------------------------------------------------

AEB_DISTANCE = 5.0
ROAD_HALF_WIDTH = 2.0
AEB_COLLISION_DIST = 3.0
LOOK_AHEAD_DIST = 15.0
RUN_SPEED = 320.0


def plan_with_avoidance(
    cur_s: float, cur_d: float,
    obstacles: list[dict],
    recovery_state: int,
) -> tuple[int, float, float]:
    """Return (request_type, best_d, closest_obs_dist).

    request_type: 0=FORWARD, 1=STOP, 2=BACK, 3=AEB
    """
    # Recovery overrides
    if recovery_state == 1:    # WAITING
        return 1, 0.0, float('inf')
    if recovery_state == 2:    # BACKING_UP
        return 2, 0.0, float('inf')
    if recovery_state == 3:    # STOPPED
        return 1, 0.0, float('inf')

    # Phase 2: d-offset candidate scoring
    #
    # Key design: only offset while obstacle is CLOSE AHEAD (within avoidance
    # window).  Once the robot passes the obstacle, best_d returns to 0
    # immediately so the path merges back to the reference lane.
    AVOIDANCE_WINDOW = 5.0   # only offset for obstacles within 5m ahead
    OBSTACLE_WIDTH = 1.0     # treat each obstacle as blocking ±1m laterally

    candidates = [
        d for d in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        if abs(d) <= ROAD_HALF_WIDTH
    ]
    best_d = 0.0
    best_cost = float('inf')
    all_blocked = True  # assume blocked until a clear candidate is found

    if obstacles:
        for d_cand in candidates:
            obstacle_cost = 0.0
            blocked = False
            for obs in obstacles:
                obs_s, obs_d = obs["s"], obs["d"]
                s_ahead = obs_s - cur_s
                if 0 < s_ahead < AVOIDANCE_WINDOW:
                    lateral_dist = abs(obs_d - d_cand)
                    if lateral_dist < OBSTACLE_WIDTH:
                        # This candidate is blocked by this obstacle
                        blocked = True
                        obstacle_cost += 10.0  # heavy penalty
                    else:
                        obstacle_cost += 0.5 / lateral_dist  # mild repulsion
            # Strong preference for d=0 (stay on lane)
            deviation_cost = abs(d_cand) * 2.0
            smoothness_cost = abs(d_cand - cur_d) * 0.5
            total = obstacle_cost + deviation_cost + smoothness_cost
            if not blocked:
                all_blocked = False
            if total < best_cost:
                best_cost = total
                best_d = d_cand

    # Phase 1: AEB check — triggers when ALL candidates are blocked
    closest_obs_dist = float('inf')
    for obs in obstacles:
        obs_s, obs_d, obs_dist = obs["s"], obs["d"], obs["distance"]
        s_ahead = obs_s - cur_s
        if 0 < s_ahead < AEB_DISTANCE:
            if obs_dist < closest_obs_dist:
                closest_obs_dist = obs_dist

    if all_blocked and obstacles and closest_obs_dist < AEB_COLLISION_DIST:
        return 3, 0.0, closest_obs_dist

    return 0, best_d, closest_obs_dist


# ---------------------------------------------------------------------------
# Robot kinematic simulation — simple Pure Pursuit step
# ---------------------------------------------------------------------------

ROBOT_SPEED_MPS = 0.6   # metres per simulated step (0.05 s → 12 m/s real would be too fast)
BACK_SPEED_MPS = 0.4


def simulate_robot_step(
    x: float, y: float, theta: float,
    path_x_veh: np.ndarray, path_y_veh: np.ndarray,
    backward: bool = False,
) -> tuple[float, float, float]:
    """Pure Pursuit kinematic step.  path_*_veh are in vehicle (rear-axle) frame."""
    look_ahead = 2.5  # m — pure pursuit look-ahead in vehicle frame

    # Find first path point beyond look_ahead distance (in vehicle frame +Y = forward)
    best_idx = min(4, len(path_x_veh) - 1)
    for i in range(len(path_x_veh)):
        dist = math.sqrt(path_x_veh[i] ** 2 + path_y_veh[i] ** 2)
        if dist >= look_ahead:
            best_idx = i
            break

    target_xv = float(path_x_veh[best_idx])
    target_yv = float(path_y_veh[best_idx])

    # Steering angle in vehicle frame: path_y_veh is lateral offset
    # path_x_veh = forward in a rotated vehicle-frame convention from planning_node:
    #   path_x_veh[i] = shift_x * sin_yaw - shift_y * cos_yaw   (tangential)
    #   path_y_veh[i] = shift_y * sin_yaw + shift_x * cos_yaw   (lateral)
    # Heading correction = atan2(target_yv, target_xv) in vehicle frame.
    steer_angle = math.atan2(target_yv, max(0.01, target_xv))
    steer_angle = max(-0.6, min(0.6, steer_angle))

    speed = -BACK_SPEED_MPS if backward else ROBOT_SPEED_MPS
    new_x = x + speed * math.cos(theta + steer_angle * 0.5)
    new_y = y + speed * math.sin(theta + steer_angle * 0.5)
    new_theta = theta + speed * math.sin(steer_angle) / 1.0   # wheelbase ≈ 1 m

    return new_x, new_y, new_theta


# ---------------------------------------------------------------------------
# Rerun visualization helpers
# ---------------------------------------------------------------------------

REQUEST_NAMES = {0: "FORWARD", 1: "STOP", 2: "BACKING UP", 3: "AEB"}
RECOVERY_NAMES = {0: "NORMAL", 1: "WAITING", 2: "BACKING UP", 3: "STOPPED"}


def viz_waypoints(waypoints: np.ndarray) -> None:
    pts3d = np.column_stack([waypoints[:, :2], np.zeros(len(waypoints))])
    rr.log("global_path/waypoints", rr.LineStrips3D(
        [pts3d.astype(np.float32)],
        colors=[[255, 255, 255, 255]],
        radii=[0.08],
    ))


def viz_robot(x: float, y: float, theta: float, label: str = "Robot") -> None:
    angle_rad = theta
    rr.log("robot/body", rr.Boxes3D(
        centers=[[x, y, 0.2]],
        half_sizes=[[0.6, 0.4, 0.25]],
        rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], angle=angle_rad)],
        colors=[[255, 255, 0, 255]],
        labels=[label],
    ))
    dx = 2.5 * math.cos(angle_rad)
    dy = 2.5 * math.sin(angle_rad)
    rr.log("robot/arrow", rr.Arrows3D(
        origins=[[x, y, 0.3]],
        vectors=[[dx, dy, 0.0]],
        colors=[[0, 255, 255, 255]],
    ))


def viz_path(
    path_x_veh: np.ndarray, path_y_veh: np.ndarray,
    robot_x: float, robot_y: float, robot_theta: float,
) -> None:
    """Convert vehicle-frame path back to global frame for visualization."""
    # Inverse of planning_node transform:
    #   path_x_veh = shift_x * sin_yaw - shift_y * cos_yaw
    #   path_y_veh = shift_y * sin_yaw + shift_x * cos_yaw
    # Invert: shift_x = path_x_veh * sin_yaw + path_y_veh * cos_yaw
    #         shift_y = -path_x_veh * cos_yaw + path_y_veh * sin_yaw
    rear_x = robot_x - TRANS_PARA_BACK * math.cos(robot_theta)
    rear_y = robot_y - TRANS_PARA_BACK * math.sin(robot_theta)
    sin_yaw = math.sin(robot_theta)
    cos_yaw = math.cos(robot_theta)
    shift_x = path_x_veh * sin_yaw + path_y_veh * cos_yaw
    shift_y = -path_x_veh * cos_yaw + path_y_veh * sin_yaw
    gx = (shift_x + rear_x).astype(np.float32)
    gy = (shift_y + rear_y).astype(np.float32)
    gz = np.full_like(gx, 0.05)
    pts = np.stack([gx, gy, gz], axis=1)
    rr.log("path/planned", rr.LineStrips3D(
        [pts],
        colors=[[0, 128, 255, 255]],
        radii=[0.08],
    ))


def viz_obstacles(obstacles: list[dict]) -> None:
    if not obstacles:
        rr.log("obstacles", rr.Clear(recursive=True))
        return
    centers, half_sizes, colors, labels = [], [], [], []
    for o in obstacles:
        centers.append([o["x"], o["y"], o["height"] / 2.0])
        half_sizes.append([
            max(0.15, o["length"] / 2.0),
            max(0.15, o["width"] / 2.0),
            max(0.15, o["height"] / 2.0),
        ])
        colors.append([255, 0, 0, 200])
        labels.append(f"#{o['id']} d={o['d']:.2f} dist={o['distance']:.1f}m")
    rr.log("obstacles", rr.Boxes3D(
        centers=centers,
        half_sizes=half_sizes,
        colors=colors,
        labels=labels,
    ))


def viz_live_points(raw_pc: bytes, robot_x: float, robot_y: float, robot_theta: float) -> None:
    """Show synthetic pointcloud in green (transformed to global frame)."""
    pts = _parse_pointcloud(raw_pc)
    if len(pts) == 0:
        return
    cos_t = math.cos(robot_theta)
    sin_t = math.sin(robot_theta)
    lx, ly = pts[:, 0], pts[:, 1]
    gx = cos_t * lx - sin_t * ly + robot_x
    gy = sin_t * lx + cos_t * ly + robot_y
    gz = pts[:, 2]
    global_pts = np.stack([gx, gy, gz], axis=1).astype(np.float32)
    rr.log("live_points", rr.Points3D(
        global_pts,
        colors=np.full((len(global_pts), 4), [0, 220, 80, 180], dtype=np.uint8),
        radii=[0.05],
    ))


def viz_costmap(costmap: Costmap2D, robot_x: float, robot_y: float) -> None:
    grid = costmap._grid
    occupied = np.argwhere(grid > 0)
    if len(occupied) == 0:
        rr.log("costmap", rr.Clear(recursive=True))
        return
    cell_row = occupied[:, 0]
    cell_col = occupied[:, 1]
    gx = costmap.origin_x + (cell_col + 0.5) * COSTMAP_RESOLUTION
    gy = costmap.origin_y + (cell_row + 0.5) * COSTMAP_RESOLUTION
    gz = np.full_like(gx, -0.05)
    costs = grid[cell_row, cell_col]
    colors = np.zeros((len(costs), 4), dtype=np.uint8)
    occ_mask = costs >= 254
    colors[occ_mask] = [255, 0, 0, 180]
    colors[~occ_mask] = [255, 220, 0, 90]
    pts = np.stack([gx, gy, gz], axis=1).astype(np.float32)
    rr.log("costmap", rr.Points3D(pts, colors=colors, radii=[COSTMAP_RESOLUTION / 2.0]))


def viz_status(
    scenario_name: str,
    request_type: int,
    best_d: float,
    recovery_state: int,
    step: int,
    extra: str = "",
) -> None:
    req_name = REQUEST_NAMES.get(request_type, str(request_type))
    rec_name = RECOVERY_NAMES.get(recovery_state, str(recovery_state))
    msg = (
        f"[{scenario_name}]  step={step:04d}  "
        f"request={req_name}  best_d={best_d:+.2f}m  "
        f"recovery={rec_name}"
        + (f"  {extra}" if extra else "")
    )
    level = rr.TextLogLevel.WARN if request_type in (1, 3) else rr.TextLogLevel.INFO
    if recovery_state > 0:
        level = rr.TextLogLevel.WARN
    rr.log("status/planner", rr.TextLog(msg, level=level))


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    name: str,
    maps_x: np.ndarray, maps_y: np.ndarray, maps_s: np.ndarray,
    start_s: float,
    obstacles_global: list[tuple[float, float, float, float, float]],
    num_steps: int = 100,
    step_offset: int = 0,
) -> int:
    """Run one scenario; returns final step index."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    rr.log("status/scenario", rr.TextLog(f"=== {name} ===", level=rr.TextLogLevel.INFO))

    # Starting pose
    robot_x, robot_y = get_xy_from_frenet(start_s, 0.0, maps_s, maps_x, maps_y)
    robot_theta = _heading_at_s(start_s, maps_s, maps_x, maps_y)

    costmap = Costmap2D()
    recovery_state = 0
    cur_d = 0.0
    trail: list[list[float]] = []
    aeb_hold_steps = 0
    stuck_steps = 0
    backing_steps = 0
    STUCK_THRESHOLD = 8
    BACKING_DURATION = 15

    for step in range(num_steps):
        global_step = step_offset + step
        rr.set_time_sequence("step", global_step)

        # Frenet coords of robot
        cur_s, cur_d_actual = get_frenet(robot_x, robot_y, maps_x, maps_y, maps_s)

        # Generate synthetic pointcloud from obstacle bounding boxes
        raw_pc = create_fake_pointcloud(obstacles_global, robot_x, robot_y, robot_theta)

        # Detect obstacles
        detected = detect_obstacles(
            raw_pc, robot_x, robot_y, robot_theta, maps_x, maps_y, maps_s
        )

        # Build costmap from detected obstacle points (re-use their centroids)
        if detected:
            global_pts = np.array([[o["x"], o["y"]] for o in detected], dtype=np.float32)
        else:
            global_pts = np.empty((0, 2), dtype=np.float32)
        costmap.update(global_pts, robot_x, robot_y)

        # Planning decision
        request_type, best_d, aeb_dist = plan_with_avoidance(
            cur_s, cur_d_actual, detected, recovery_state
        )

        # Recovery state machine (Scenario 5)
        if request_type == 3:      # AEB triggered
            aeb_hold_steps += 1
            stuck_steps += 1
            if stuck_steps >= STUCK_THRESHOLD and recovery_state == 0:
                recovery_state = 1   # WAITING
                rr.log("status/recovery", rr.TextLog(
                    "Transition → WAITING", level=rr.TextLogLevel.WARN
                ))
        elif request_type == 0:    # FORWARD
            stuck_steps = 0
            aeb_hold_steps = 0
            if recovery_state == 3:
                recovery_state = 0  # clear STOPPED once unblocked

        if recovery_state == 1:
            # Stay WAITING for a couple steps, then start BACKING_UP
            if stuck_steps >= STUCK_THRESHOLD + 3:
                recovery_state = 2
                backing_steps = 0
                rr.log("status/recovery", rr.TextLog(
                    "Transition → BACKING_UP", level=rr.TextLogLevel.WARN
                ))

        if recovery_state == 2:
            backing_steps += 1
            if backing_steps >= BACKING_DURATION:
                recovery_state = 3
                rr.log("status/recovery", rr.TextLog(
                    "Transition → STOPPED", level=rr.TextLogLevel.ERROR
                ))

        # Re-plan after recovery override updates request_type
        request_type, best_d, aeb_dist = plan_with_avoidance(
            cur_s, cur_d_actual, detected, recovery_state
        )

        # Generate path
        if abs(best_d) > 0.01:
            path_x_veh, path_y_veh = plan_path_with_d(
                robot_x, robot_y, robot_theta,
                cur_s, cur_d_actual, best_d,
                maps_s, maps_x, maps_y, RUN_SPEED,
            )
        else:
            path_x_veh, path_y_veh = plan_path(
                robot_x, robot_y, robot_theta,
                cur_s, cur_d_actual,
                maps_s, maps_x, maps_y, RUN_SPEED,
            )

        # --- Visualize ---
        viz_robot(robot_x, robot_y, robot_theta)
        viz_path(path_x_veh, path_y_veh, robot_x, robot_y, robot_theta)
        viz_obstacles(detected)
        viz_live_points(raw_pc, robot_x, robot_y, robot_theta)
        viz_costmap(costmap, robot_x, robot_y)

        extra = ""
        if request_type == 3:
            extra = f"AEB! closest={aeb_dist:.2f}m"
        elif request_type == 2:
            extra = "REVERSING"
        viz_status(name, request_type, best_d, recovery_state, global_step, extra)

        # --- Move robot ---
        if request_type == 0:     # FORWARD
            robot_x, robot_y, robot_theta = simulate_robot_step(
                robot_x, robot_y, robot_theta, path_x_veh, path_y_veh
            )
        elif request_type == 2:   # BACKING_UP
            robot_x, robot_y, robot_theta = simulate_robot_step(
                robot_x, robot_y, robot_theta, path_x_veh, path_y_veh, backward=True
            )
        elif request_type == 3:
            # AEB — robot is stationary but log the event prominently
            if aeb_hold_steps == 1:
                rr.log("status/aeb", rr.TextLog(
                    f"AEB TRIGGERED — closest obstacle at {aeb_dist:.2f} m",
                    level=rr.TextLogLevel.ERROR,
                ))
        # request_type 1 (STOP/WAITING) — robot stays put

        trail.append([robot_x, robot_y, 0.05])
        if len(trail) >= 2:
            rr.log("robot/trail", rr.LineStrips3D(
                [trail], colors=[[255, 102, 0, 255]], radii=[0.1]
            ))

        progress = (step + 1) / num_steps
        bar = "#" * int(progress * 30) + "." * (30 - int(progress * 30))
        sys.stdout.write(
            f"\r  [{bar}] {step+1}/{num_steps}  "
            f"pos=({robot_x:.1f},{robot_y:.1f})  "
            f"req={REQUEST_NAMES.get(request_type,'?')}  "
            f"d={best_d:+.2f}"
        )
        sys.stdout.flush()

        time.sleep(0.05)

    print()   # newline after progress bar
    return step_offset + num_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rr.init("obstacle_avoidance_test", spawn=True)

    print("Loading waypoints...")
    waypoints = load_waypoints()
    maps_x = waypoints[:, 0]
    maps_y = waypoints[:, 1]
    maps_s = compute_cumulative_s(maps_x, maps_y)
    total_length = maps_s[-1]
    print(f"Road length: {total_length:.1f} m  ({len(maps_x)} waypoints)")

    # Safe starting s — leave room for obs at s+10
    start_s = min(5.0, total_length * 0.05)
    obs_s = start_s + 10.0   # place obstacle 10 m ahead

    # Draw waypoints (static, logged once)
    viz_waypoints(waypoints)

    global_step = 0

    # -----------------------------------------------------------------------
    # Scenario 1: No obstacles — baseline straight driving
    # -----------------------------------------------------------------------
    global_step = run_scenario(
        "Scenario 1: No Obstacles — Baseline",
        maps_x, maps_y, maps_s,
        start_s=start_s,
        obstacles_global=[],
        num_steps=80,
        step_offset=global_step,
    )
    time.sleep(0.5)
    rr.log("robot/trail", rr.Clear(recursive=True))

    # -----------------------------------------------------------------------
    # Scenario 2: Staggered wall across entire road → no lateral escape → AEB
    # Obstacles spread in both s and d so they cluster individually,
    # each blocking one d-candidate.
    # -----------------------------------------------------------------------
    aeb_wall = []
    for i, d_val in enumerate([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]):
        s_offset = obs_s + (i % 3) * 1.5  # stagger s: 15, 16.5, 18, 15, 16.5, ...
        wx, wy = get_xy_from_frenet(s_offset, d_val, maps_s, maps_x, maps_y)
        aeb_wall.append((wx, wy, 0.6, 0.6, 0.8))
    print(f"\nScenario 2: staggered wall of {len(aeb_wall)} obstacles")
    global_step = run_scenario(
        "Scenario 2: Staggered Wall — AEB Emergency Stop",
        maps_x, maps_y, maps_s,
        start_s=start_s,
        obstacles_global=aeb_wall,
        num_steps=100,
        step_offset=global_step,
    )
    time.sleep(0.5)
    rr.log("robot/trail", rr.Clear(recursive=True))
    rr.log("obstacles", rr.Clear(recursive=True))
    rr.log("costmap", rr.Clear(recursive=True))

    # -----------------------------------------------------------------------
    # Scenario 3: Single obstacle on path → lateral swerve around it
    # -----------------------------------------------------------------------
    obs_x3, obs_y3 = get_xy_from_frenet(obs_s, 0.0, maps_s, maps_x, maps_y)
    print(f"\nScenario 3 obstacle at global ({obs_x3:.2f}, {obs_y3:.2f})  "
          f"[s={obs_s:.1f} d=0.0]")
    global_step = run_scenario(
        "Scenario 3: Single Obstacle — Lateral Swerve",
        maps_x, maps_y, maps_s,
        start_s=start_s,
        obstacles_global=[(obs_x3, obs_y3, 1.0, 1.0, 1.0)],
        num_steps=120,
        step_offset=global_step,
    )
    time.sleep(0.5)
    rr.log("robot/trail", rr.Clear(recursive=True))
    rr.log("obstacles", rr.Clear(recursive=True))
    rr.log("costmap", rr.Clear(recursive=True))

    # -----------------------------------------------------------------------
    # Scenario 4: Two obstacles at ±1.5 m → planner finds gap at d=0
    # -----------------------------------------------------------------------
    obs4a_x, obs4a_y = get_xy_from_frenet(obs_s, -1.5, maps_s, maps_x, maps_y)
    obs4b_x, obs4b_y = get_xy_from_frenet(obs_s, 1.5, maps_s, maps_x, maps_y)
    print(f"\nScenario 4 obstacles at d=-1.5 and d=+1.5")
    global_step = run_scenario(
        "Scenario 4: Two Obstacles — Lane Gap Selection",
        maps_x, maps_y, maps_s,
        start_s=start_s,
        obstacles_global=[
            (obs4a_x, obs4a_y, 0.8, 0.8, 0.8),
            (obs4b_x, obs4b_y, 0.8, 0.8, 0.8),
        ],
        num_steps=120,
        step_offset=global_step,
    )
    time.sleep(0.5)
    rr.log("robot/trail", rr.Clear(recursive=True))
    rr.log("obstacles", rr.Clear(recursive=True))
    rr.log("costmap", rr.Clear(recursive=True))

    # -----------------------------------------------------------------------
    # Scenario 5: Full road blockage → recovery state machine
    # -----------------------------------------------------------------------
    obstacles_full = []
    for i, d_val in enumerate([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]):
        s_offset = obs_s + (i % 3) * 1.5
        ox, oy = get_xy_from_frenet(s_offset, d_val, maps_s, maps_x, maps_y)
        obstacles_full.append((ox, oy, 0.6, 0.6, 0.8))
    print(f"\nScenario 5: full road blockage ({len(obstacles_full)} obstacles across width)")
    global_step = run_scenario(
        "Scenario 5: Full Blockage — Recovery State Machine",
        maps_x, maps_y, maps_s,
        start_s=start_s,
        obstacles_global=obstacles_full,
        num_steps=150,
        step_offset=global_step,
    )

    print("\n")
    print("All 5 scenarios complete.")
    print("Check the Rerun viewer — use the timeline scrubber to replay each scenario.")
    print(f"Total steps logged: {global_step}")


if __name__ == "__main__":
    main()
