#!/usr/bin/env python3
"""DORA node — Phase 1 obstacle detection from lidar pointcloud.

Inputs:  pointcloud, ground_truth_pose, road_lane
Output:  obstacle_list
"""

import math
import os
import struct

import numpy as np
import pyarrow as pa
from dora import Node
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Thresholds (overridable via environment variables)
# ---------------------------------------------------------------------------
GROUND_THRESHOLD: float = float(os.environ.get("GROUND_THRESHOLD", "0.15"))
MAX_DETECT_RANGE: float = float(os.environ.get("MAX_DETECT_RANGE", "15.0"))
CLUSTER_TOLERANCE: float = float(os.environ.get("CLUSTER_TOLERANCE", "0.5"))
MIN_CLUSTER_SIZE: int = int(os.environ.get("MIN_CLUSTER_SIZE", "3"))

PC_HEADER_SIZE = 16   # bytes to skip at start of pointcloud message
PC_POINT_SIZE = 16    # float32 x, y, z, intensity per point
POSE_FMT = "<3f"      # float32 x, y, theta_deg  (12 bytes)
OBSTACLE_STRUCT = "<I 2f 2f 2f 3f f"  # id, x,y, s,d, vx,vy, l,w,h, dist

# ---------------------------------------------------------------------------
# Cached state
# ---------------------------------------------------------------------------
_maps_x: np.ndarray = np.empty(0, dtype=np.float32)
_maps_y: np.ndarray = np.empty(0, dtype=np.float32)
_maps_s: np.ndarray = np.empty(0, dtype=np.float32)
_robot_x: float = 0.0
_robot_y: float = 0.0
_robot_theta_deg: float = 0.0
_obstacle_id_counter: int = 0

# ---------------------------------------------------------------------------
# Frenet helpers (exact logic from road_lane_publisher_node.py)
# ---------------------------------------------------------------------------

def compute_cumulative_s(maps_x: np.ndarray, maps_y: np.ndarray) -> np.ndarray:
    s = np.zeros(len(maps_x), dtype=np.float64)
    for i in range(1, len(maps_x)):
        dx = maps_x[i] - maps_x[i - 1]
        dy = maps_y[i] - maps_y[i - 1]
        s[i] = s[i - 1] + math.sqrt(dx * dx + dy * dy)
    return s


def next_waypoint(x: float, y: float, maps_x: np.ndarray, maps_y: np.ndarray) -> int:
    n = len(maps_x)
    dx = maps_x - x
    dy = maps_y - y
    dists = dx * dx + dy * dy
    closest = int(np.argmin(dists))
    cx = maps_x[closest]
    cy = maps_y[closest]
    if closest < n - 1:
        seg_x = maps_x[closest + 1] - cx
        seg_y = maps_y[closest + 1] - cy
        pt_x = x - cx
        pt_y = y - cy
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
    nwp = next_waypoint(x, y, maps_x, maps_y)
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

# ---------------------------------------------------------------------------
# Pointcloud helpers
# ---------------------------------------------------------------------------

def parse_pointcloud(raw: bytes) -> np.ndarray:
    """Return (N, 4) float32 array [x, y, z, intensity] or empty array."""
    payload = raw[PC_HEADER_SIZE:]
    n = len(payload) // PC_POINT_SIZE
    if n == 0:
        return np.empty((0, 4), dtype=np.float32)
    return np.frombuffer(payload[: n * PC_POINT_SIZE], dtype=np.float32).reshape(n, 4)


def transform_to_global(pts: np.ndarray, rx: float, ry: float, theta_deg: float) -> np.ndarray:
    """Rotate lidar points (robot frame) into global frame in-place copy."""
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = pts.copy()
    lx, ly = pts[:, 0], pts[:, 1]
    out[:, 0] = cos_t * lx - sin_t * ly + rx
    out[:, 1] = sin_t * lx + cos_t * ly + ry
    return out

# ---------------------------------------------------------------------------
# Euclidean clustering via KDTree
# ---------------------------------------------------------------------------

def euclidean_cluster(xy: np.ndarray, tolerance: float, min_size: int) -> list[np.ndarray]:
    """Return list of index arrays, one per cluster meeting min_size."""
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

# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

def detect_obstacles(raw_pc: bytes) -> list[dict]:
    global _obstacle_id_counter
    pts = parse_pointcloud(raw_pc)
    if len(pts) == 0:
        return []

    g_pts = transform_to_global(pts, _robot_x, _robot_y, _robot_theta_deg)

    # Ground removal
    g_pts = g_pts[g_pts[:, 2] > GROUND_THRESHOLD]
    if len(g_pts) == 0:
        return []

    # Distance filter
    dx = g_pts[:, 0] - _robot_x
    dy = g_pts[:, 1] - _robot_y
    robot_dists = np.sqrt(dx * dx + dy * dy)
    mask = robot_dists <= MAX_DETECT_RANGE
    g_pts = g_pts[mask]
    robot_dists = robot_dists[mask]
    if len(g_pts) == 0:
        return []

    clusters = euclidean_cluster(g_pts[:, :2], CLUSTER_TOLERANCE, MIN_CLUSTER_SIZE)
    has_lane = len(_maps_x) >= 2
    obstacles: list[dict] = []

    for indices in clusters:
        cpts = g_pts[indices]
        cx = float(np.mean(cpts[:, 0]))
        cy = float(np.mean(cpts[:, 1]))
        length = float(np.max(cpts[:, 0]) - np.min(cpts[:, 0]))
        width = float(np.max(cpts[:, 1]) - np.min(cpts[:, 1]))
        height = float(np.max(cpts[:, 2]) - np.min(cpts[:, 2]))
        frenet_s, frenet_d = get_frenet(cx, cy, _maps_x, _maps_y, _maps_s) if has_lane else (0.0, 0.0)
        dist = float(np.min(robot_dists[indices]))
        _obstacle_id_counter += 1
        obstacles.append({
            "id": _obstacle_id_counter,
            "x": cx, "y": cy,
            "s": frenet_s, "d": frenet_d,
            "vx": 0.0, "vy": 0.0,
            "length": length, "width": width, "height": height,
            "distance": dist,
        })
    return obstacles


def pack_obstacles(obstacles: list[dict]) -> bytes:
    parts = [struct.pack("<I", len(obstacles))]
    for o in obstacles:
        parts.append(struct.pack(
            OBSTACLE_STRUCT,
            o["id"], o["x"], o["y"], o["s"], o["d"],
            o["vx"], o["vy"], o["length"], o["width"], o["height"], o["distance"],
        ))
    return b"".join(parts)

# ---------------------------------------------------------------------------
# DORA event loop
# ---------------------------------------------------------------------------

def main() -> None:
    global _maps_x, _maps_y, _maps_s
    global _robot_x, _robot_y, _robot_theta_deg

    print(
        f"[obstacle-detector] starting | ground_thresh={GROUND_THRESHOLD}m "
        f"max_range={MAX_DETECT_RANGE}m tol={CLUSTER_TOLERANCE}m min_pts={MIN_CLUSTER_SIZE}"
    )

    node = Node()
    for event in node:
        if event["type"] == "STOP":
            print("[obstacle-detector] STOP received, shutting down")
            break
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        raw = bytes(event["value"].to_pylist())

        if eid == "road_lane":
            arr = np.frombuffer(raw, dtype=np.float32)
            if len(arr) >= 2 and len(arr) % 2 == 0:
                half = len(arr) // 2
                _maps_x = arr[:half].copy()
                _maps_y = arr[half:].copy()
                _maps_s = compute_cumulative_s(_maps_x, _maps_y).astype(np.float32)
                print(f"[obstacle-detector] road_lane updated: {half} waypoints")

        elif eid == "ground_truth_pose":
            if len(raw) >= struct.calcsize(POSE_FMT):
                _robot_x, _robot_y, _robot_theta_deg = struct.unpack_from(POSE_FMT, raw)

        elif eid == "pointcloud":
            obstacles = detect_obstacles(raw)
            out_bytes = pack_obstacles(obstacles)
            node.send_output("obstacle_list", pa.array(list(out_bytes), type=pa.uint8()))
            print(
                f"[obstacle-detector] {len(obstacles)} obstacle(s) | "
                f"robot=({_robot_x:.2f},{_robot_y:.2f}) theta={_robot_theta_deg:.1f}deg"
            )


if __name__ == "__main__":
    main()
