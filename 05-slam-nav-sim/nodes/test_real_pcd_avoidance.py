#!/usr/bin/env python3
"""Visual test — obstacle avoidance using a real PCD file (table_scene_lms400.pcd).

Loads the real laser scan, places it as an obstacle on the robot's path,
and demonstrates detection + swerve-and-return behavior in Rerun.

Run:  python3 python/test_real_pcd_avoidance.py
"""
from __future__ import annotations

import math
import struct
import time

import numpy as np
import open3d as o3d
import rerun as rr
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PC_HEADER_SIZE = 16
PC_POINT_SIZE = 16
TRANS_PARA_BACK = 0.25
NUM_PATH_POINTS = 30

GROUND_THRESHOLD = 0.15
MAX_DETECT_RANGE = 15.0
CLUSTER_TOLERANCE = 0.5
MIN_CLUSTER_SIZE = 5

AVOIDANCE_WINDOW = 5.0
OBSTACLE_HALF_W = 1.0
ROAD_HALF_WIDTH = 2.0
AEB_COLLISION_DIST = 3.0
AEB_DISTANCE = 5.0
RUN_SPEED = 320.0
ROBOT_SPEED_MPS = 0.5


# ---------------------------------------------------------------------------
# Frenet math (from road_lane_publisher_node.py)
# ---------------------------------------------------------------------------
def compute_cumulative_s(mx, my):
    s = np.zeros(len(mx))
    for i in range(1, len(mx)):
        s[i] = s[i - 1] + math.hypot(mx[i] - mx[i - 1], my[i] - my[i - 1])
    return s


def next_waypoint(x, y, mx, my):
    dx, dy = mx - x, my - y
    closest = int(np.argmin(dx * dx + dy * dy))
    if closest < len(mx) - 1:
        seg_x = mx[closest + 1] - mx[closest]
        seg_y = my[closest + 1] - my[closest]
        if (x - mx[closest]) * seg_x + (y - my[closest]) * seg_y > 0:
            return closest + 1
    return closest


def get_frenet(x, y, mx, my, ms):
    if len(mx) < 2:
        return 0.0, 0.0
    nwp = next_waypoint(x, y, mx, my)
    if nwp == 0:
        nwp = 1
    prev = nwp - 1
    lx, ly = mx[nwp] - mx[prev], my[nwp] - my[prev]
    qx, qy = x - mx[prev], y - my[prev]
    ll = lx * lx + ly * ly
    pn = (qx * lx + qy * ly) / ll if ll > 1e-12 else 0.0
    px, py = pn * lx, pn * ly
    cross = qx * ly - qy * lx
    sign = 1.0 if cross > 0 else -1.0
    return ms[prev] + math.hypot(px, py), sign * math.hypot(qx - px, qy - py)


def get_xy_from_frenet(s_val, d_val, ms, mx, my):
    if len(ms) < 2:
        return 0.0, 0.0
    s_val = max(ms[0] + 0.001, min(s_val, ms[-1] - 0.001))
    prev = 0
    for i in range(len(ms) - 1):
        if ms[i] <= s_val < ms[i + 1]:
            prev = i
            break
    nxt = prev + 1
    h = math.atan2(my[nxt] - my[prev], mx[nxt] - mx[prev])
    seg_s = s_val - ms[prev]
    sx = mx[prev] + seg_s * math.cos(h)
    sy = my[prev] + seg_s * math.sin(h)
    perp = h - math.pi / 2.0
    return sx + d_val * math.cos(perp), sy + d_val * math.sin(perp)


def heading_at_s(s_val, ms, mx, my):
    prev = 0
    for i in range(len(ms) - 1):
        if ms[i] <= s_val < ms[i + 1]:
            prev = i
            break
    return math.atan2(my[prev + 1] - my[prev], mx[prev + 1] - mx[prev])


# ---------------------------------------------------------------------------
# Path planning (from planning_node.py)
# ---------------------------------------------------------------------------
def plan_path(cx, cy, ct, cs, cd, ms, mx, my, speed):
    vel = speed * 0.00277
    pd = 15.0 + vel * 0.277
    rx = cx - TRANS_PARA_BACK * math.cos(ct)
    ry = cy - TRANS_PARA_BACK * math.sin(ct)
    end_s = ms[-1] if len(ms) > 0 else 0.0
    gx = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    gy = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    for i in range(NUM_PATH_POINTS):
        ts = cs + i * (pd / 50.0)
        if ts >= end_s:
            gx[i], gy[i] = float(mx[-1]), float(my[-1])
        else:
            gx[i], gy[i] = get_xy_from_frenet(ts, 0.0, ms, mx, my)
    sin_y, cos_y = math.sin(ct), math.cos(ct)
    vx = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    vy = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    for i in range(NUM_PATH_POINTS):
        sx, sy = gx[i] - rx, gy[i] - ry
        vx[i] = sx * sin_y - sy * cos_y
        vy[i] = sy * sin_y + sx * cos_y
    return vx, vy


def plan_path_with_d(cx, cy, ct, cs, cd, td, ms, mx, my, speed):
    from scipy.interpolate import CubicSpline
    vel = speed * 0.00277
    pd = 15.0 + vel * 0.277
    rx = cx - TRANS_PARA_BACK * math.cos(ct)
    ry = cy - TRANS_PARA_BACK * math.sin(ct)
    end_s = ms[-1] if len(ms) > 0 else 0.0
    tr = 3.0
    pts_s = [cs, cs + tr * 0.5, cs + tr, cs + pd]
    pts_d = [cd, (cd + td) / 2.0, td, td]
    spl = CubicSpline(pts_s, pts_d, bc_type="clamped")
    gx = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    gy = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    for i in range(NUM_PATH_POINTS):
        ts = cs + i * (pd / 50.0)
        dv = float(spl(ts)) if ts <= cs + pd else td
        if ts >= end_s:
            gx[i], gy[i] = float(mx[-1]), float(my[-1])
        else:
            gx[i], gy[i] = get_xy_from_frenet(ts, dv, ms, mx, my)
    sin_y, cos_y = math.sin(ct), math.cos(ct)
    vx = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    vy = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    for i in range(NUM_PATH_POINTS):
        sx, sy = gx[i] - rx, gy[i] - ry
        vx[i] = sx * sin_y - sy * cos_y
        vy[i] = sy * sin_y + sx * cos_y
    return vx, vy


# ---------------------------------------------------------------------------
# Detection (from obstacle_detector_node.py)
# ---------------------------------------------------------------------------
def euclidean_cluster(xy, tol, min_sz):
    if len(xy) == 0:
        return []
    tree = KDTree(xy)
    visited = np.zeros(len(xy), dtype=bool)
    clusters = []
    for i in range(len(xy)):
        if visited[i]:
            continue
        queue = list(tree.query_ball_point(xy[i], tol))
        visited[i] = True
        idxs = []
        head = 0
        while head < len(queue):
            idx = queue[head]
            head += 1
            if visited[idx] and idx != i:
                continue
            visited[idx] = True
            idxs.append(idx)
            for nb in tree.query_ball_point(xy[idx], tol):
                if not visited[nb]:
                    queue.append(nb)
                    visited[nb] = True
        if len(idxs) >= min_sz:
            clusters.append(np.array(idxs, dtype=np.int32))
    return clusters


_obs_id = 0


def detect_obstacles(global_pts, robot_x, robot_y, mx, my, ms):
    """Detect obstacles from already-global-frame points (Nx3 float)."""
    global _obs_id
    if len(global_pts) == 0:
        return []
    # Ground filter
    mask = global_pts[:, 2] > GROUND_THRESHOLD
    pts = global_pts[mask]
    if len(pts) == 0:
        return []
    # Distance filter
    dx, dy = pts[:, 0] - robot_x, pts[:, 1] - robot_y
    dists = np.sqrt(dx * dx + dy * dy)
    mask2 = dists <= MAX_DETECT_RANGE
    pts = pts[mask2]
    dists = dists[mask2]
    if len(pts) == 0:
        return []
    clusters = euclidean_cluster(pts[:, :2], CLUSTER_TOLERANCE, MIN_CLUSTER_SIZE)
    has_lane = len(mx) >= 2
    results = []
    for idxs in clusters:
        cp = pts[idxs]
        cx_o = float(np.mean(cp[:, 0]))
        cy_o = float(np.mean(cp[:, 1]))
        length = max(0.3, float(np.ptp(cp[:, 0])))
        width = max(0.3, float(np.ptp(cp[:, 1])))
        height = max(0.3, float(np.ptp(cp[:, 2])))
        fs, fd = get_frenet(cx_o, cy_o, mx, my, ms) if has_lane else (0.0, 0.0)
        dist = float(np.min(dists[idxs]))
        _obs_id += 1
        results.append({
            "id": _obs_id, "x": cx_o, "y": cy_o,
            "s": fs, "d": fd, "length": length, "width": width,
            "height": height, "distance": dist,
        })
    return results


# ---------------------------------------------------------------------------
# Planning logic (from planning_node.py)
# ---------------------------------------------------------------------------
def plan_with_avoidance(cur_s, cur_d, obstacles, recovery_state=0):
    if recovery_state == 1:
        return 1, 0.0, float("inf")
    if recovery_state == 2:
        return 2, 0.0, float("inf")
    if recovery_state == 3:
        return 1, 0.0, float("inf")

    candidates = [d for d in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
                  if abs(d) <= ROAD_HALF_WIDTH]
    best_d, best_cost, all_blocked = 0.0, float("inf"), True

    if obstacles:
        for dc in candidates:
            oc, blocked = 0.0, False
            for obs in obstacles:
                sa = obs["s"] - cur_s
                if 0 < sa < AVOIDANCE_WINDOW:
                    ld = abs(obs["d"] - dc)
                    if ld < OBSTACLE_HALF_W:
                        blocked = True
                        oc += 10.0
                    else:
                        oc += 0.5 / ld
            dev = abs(dc) * 2.0
            smooth = abs(dc - cur_d) * 0.5
            total = oc + dev + smooth
            if not blocked:
                all_blocked = False
            if total < best_cost:
                best_cost = total
                best_d = dc

    closest = float("inf")
    for obs in obstacles:
        sa = obs["s"] - cur_s
        if 0 < sa < AEB_DISTANCE and obs["distance"] < closest:
            closest = obs["distance"]

    if all_blocked and obstacles and closest < AEB_COLLISION_DIST:
        return 3, 0.0, closest
    return 0, best_d, closest


# ---------------------------------------------------------------------------
# Robot step
# ---------------------------------------------------------------------------
def robot_step(x, y, theta, px_v, py_v):
    look = 2.5
    best_i = 0
    for i in range(len(px_v)):
        d = math.hypot(px_v[i], py_v[i])
        if d >= look:
            best_i = i
            break
    else:
        best_i = len(px_v) - 1
    tx, ty = float(px_v[best_i]), float(py_v[best_i])
    # Vehicle frame: vx=lateral(right), vy=forward
    # Transform to global using (theta - 90°) rotation (same as rerun_viz_node.py)
    angle = theta - math.pi / 2.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rear_x = x - TRANS_PARA_BACK * math.cos(theta)
    rear_y = y - TRANS_PARA_BACK * math.sin(theta)
    gx = tx * cos_a - ty * sin_a + rear_x
    gy = tx * sin_a + ty * cos_a + rear_y
    heading = math.atan2(gy - y, gx - x)
    return x + ROBOT_SPEED_MPS * math.cos(heading), y + ROBOT_SPEED_MPS * math.sin(heading), heading


# ---------------------------------------------------------------------------
# Load and prepare real PCD
# ---------------------------------------------------------------------------
def load_and_prepare_pcd(filepath, place_s, maps_s, maps_x, maps_y, scale=3.0):
    """Load PCD, scale it up, and place it on the road at Frenet s-position.

    Returns global-frame Nx3 points.
    """
    pcd = o3d.io.read_point_cloud(filepath)
    pts = np.asarray(pcd.points).astype(np.float64)
    print(f"[pcd] Loaded {len(pts)} points from {filepath}")
    print(f"[pcd] Raw bounds: x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
          f"z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")

    # Center on median (more robust than mean for asymmetric scans) and scale
    center = np.median(pts, axis=0)
    pts -= center
    pts *= scale

    # The table surface becomes the ground plane — shift z so ground = 0
    z_min = pts[:, 2].min()
    pts[:, 2] -= z_min

    print(f"[pcd] After scale {scale}x: size = {np.ptp(pts[:,0]):.1f} x {np.ptp(pts[:,1]):.1f} x {np.ptp(pts[:,2]):.1f} m")

    # Place centroid on the road at (place_s, d=0)
    road_x, road_y = get_xy_from_frenet(place_s, 0.0, maps_s, maps_x, maps_y)
    road_heading = heading_at_s(place_s, maps_s, maps_x, maps_y)

    # Rotate point cloud to align with road direction
    cos_h, sin_h = math.cos(road_heading), math.sin(road_heading)
    rx = pts[:, 0] * cos_h - pts[:, 1] * sin_h + road_x
    ry = pts[:, 0] * sin_h + pts[:, 1] * cos_h + road_y
    rz = pts[:, 2]

    global_pts = np.stack([rx, ry, rz], axis=1).astype(np.float32)
    print(f"[pcd] Placed at road s={place_s:.1f} → global ({road_x:.1f}, {road_y:.1f})")
    return global_pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
REQUEST_NAMES = {0: "FORWARD", 1: "STOP", 2: "BACK", 3: "AEB"}


def main():
    rr.init("real_pcd_obstacle_test", spawn=True)

    # Build a straight 60m road
    n_wp = 120
    maps_x = np.linspace(0, 60, n_wp, dtype=np.float64)
    maps_y = np.zeros(n_wp, dtype=np.float64)
    maps_s = compute_cumulative_s(maps_x, maps_y)

    # Log waypoints
    wp_pts = np.zeros((n_wp, 3), dtype=np.float32)
    wp_pts[:, 0] = maps_x
    rr.log("global_path_points", rr.LineStrips3D(
        [wp_pts], colors=[[255, 255, 255, 255]], radii=[0.08]))

    # Load real PCD and place at s=25 on the road
    pcd_file = "table_scene_lms400.pcd"
    obstacle_pts = load_and_prepare_pcd(pcd_file, 25.0, maps_s, maps_x, maps_y, scale=3.0)

    # Use open3d to segment: remove table plane with RANSAC, cluster objects
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(obstacle_pts.astype(np.float64))

    # Downsample first for speed
    pcd_down = pcd_o3d.voxel_down_sample(voxel_size=0.1)
    print(f"[pcd] Voxel downsample: {len(obstacle_pts)} → {len(pcd_down.points)} points")

    # RANSAC plane segmentation (find the table surface)
    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=0.15, ransac_n=3, num_iterations=1000)
    objects_pcd = pcd_down.select_by_index(inliers, invert=True)
    print(f"[pcd] After plane removal: {len(objects_pcd.points)} object points")

    # DBSCAN clustering on remaining points
    labels = np.array(objects_pcd.cluster_dbscan(eps=0.3, min_points=10))
    obj_pts = np.asarray(objects_pcd.points).astype(np.float32)

    # Extract detected objects as pre-computed list
    precomputed_obstacles = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise
    for label in sorted(unique_labels):
        mask = labels == label
        cluster_pts = obj_pts[mask]
        cx_o = float(np.mean(cluster_pts[:, 0]))
        cy_o = float(np.mean(cluster_pts[:, 1]))
        length = max(0.3, float(np.ptp(cluster_pts[:, 0])))
        width = max(0.3, float(np.ptp(cluster_pts[:, 1])))
        height = max(0.3, float(np.ptp(cluster_pts[:, 2])))
        precomputed_obstacles.append({
            "id": label + 1, "x": cx_o, "y": cy_o,
            "s": 0.0, "d": 0.0,  # filled per-step
            "length": length, "width": width, "height": height,
            "distance": 0.0,
            "points": cluster_pts,
        })

    # Shift obstacle clusters onto the road (y=0) so they actually block it
    if precomputed_obstacles:
        y_shift = -precomputed_obstacles[0]["y"]  # center first cluster on road
        for obs in precomputed_obstacles:
            obs["y"] += y_shift
        # Also shift the full viz cloud
        obstacle_pts[:, 1] += y_shift

    print(f"[pcd] Found {len(precomputed_obstacles)} object clusters:")
    for obs in precomputed_obstacles:
        print(f"       #{obs['id']}: center=({obs['x']:.1f},{obs['y']:.1f}) "
              f"size={obs['length']:.1f}x{obs['width']:.1f}x{obs['height']:.1f}m")

    # For detection: update Frenet coords per robot position
    obstacle_pts_sparse = None  # not used; we use precomputed_obstacles directly

    # Log full PCD as red points
    # Subsample for viz too (460K points is heavy for Rerun)
    viz_stride = max(1, len(obstacle_pts) // 20000)
    viz_pts = obstacle_pts[::viz_stride]
    rr.log("real_pcd_obstacle", rr.Points3D(
        viz_pts,
        colors=np.full((len(viz_pts), 4), [255, 50, 50, 200], dtype=np.uint8),
        radii=[0.03]))
    print(f"[viz] Logged {len(viz_pts)} PCD points as red obstacle cloud")

    # ---- Run robot through the scene ----
    start_s = 5.0
    robot_x, robot_y = get_xy_from_frenet(start_s, 0.0, maps_s, maps_x, maps_y)
    robot_theta = heading_at_s(start_s, maps_s, maps_x, maps_y)
    trail = []
    cur_d = 0.0

    print(f"\n{'='*60}")
    print(f"  Robot driving through real PCD obstacle")
    print(f"{'='*60}")

    num_steps = 120
    for step in range(num_steps):
        rr.set_time_sequence("step", step)

        cur_s, cur_d_actual = get_frenet(robot_x, robot_y, maps_x, maps_y, maps_s)

        # Update precomputed obstacle Frenet coords and distance for current robot pos
        detected = []
        for obs in precomputed_obstacles:
            dx = obs["x"] - robot_x
            dy = obs["y"] - robot_y
            dist = math.hypot(dx, dy)
            if dist > MAX_DETECT_RANGE:
                continue
            fs, fd = get_frenet(obs["x"], obs["y"], maps_x, maps_y, maps_s)
            detected.append({
                **obs, "s": fs, "d": fd, "distance": dist,
            })

        # Plan
        req_type, best_d, closest_dist = plan_with_avoidance(cur_s, cur_d, detected)

        # Generate path
        if abs(best_d) > 0.01:
            px, py = plan_path_with_d(
                robot_x, robot_y, robot_theta, cur_s, cur_d_actual, best_d,
                maps_s, maps_x, maps_y, RUN_SPEED)
        else:
            px, py = plan_path(
                robot_x, robot_y, robot_theta, cur_s, cur_d_actual,
                maps_s, maps_x, maps_y, RUN_SPEED)
        cur_d = best_d

        # ---- Visualize ----
        req_name = REQUEST_NAMES.get(req_type, str(req_type))

        # Robot body (yellow)
        rr.log("robot/body", rr.Boxes3D(
            centers=[[robot_x, robot_y, 0.1]],
            half_sizes=[[0.6, 0.4, 0.25]],
            rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], angle=robot_theta)],
            colors=[[255, 255, 0, 255]], labels=["Robot"]))

        # Heading arrow (cyan)
        dx = 3.0 * math.cos(robot_theta)
        dy = 3.0 * math.sin(robot_theta)
        rr.log("robot/arrow", rr.Arrows3D(
            origins=[[robot_x, robot_y, 0.2]],
            vectors=[[dx, dy, 0.0]],
            colors=[[0, 255, 255, 255]]))

        # Planned path (blue) — transform vehicle→global using (theta-90°)
        path_pts = np.zeros((NUM_PATH_POINTS, 3), dtype=np.float32)
        angle = robot_theta - math.pi / 2.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx = robot_x - TRANS_PARA_BACK * math.cos(robot_theta)
        ry = robot_y - TRANS_PARA_BACK * math.sin(robot_theta)
        for i in range(NUM_PATH_POINTS):
            path_pts[i, 0] = float(px[i]) * cos_a - float(py[i]) * sin_a + rx
            path_pts[i, 1] = float(px[i]) * sin_a + float(py[i]) * cos_a + ry
        rr.log("path_points", rr.LineStrips3D(
            [path_pts], colors=[[0, 100, 255, 255]], radii=[0.08]))

        # Detected obstacle boxes (green outlines)
        if detected:
            centers = [[o["x"], o["y"], o["height"] / 2] for o in detected]
            halfs = [[o["length"] / 2, o["width"] / 2, o["height"] / 2] for o in detected]
            colors = [[0, 255, 0, 180]] * len(detected)
            labels = [f"obs#{o['id']} d={o['d']:.1f}m" for o in detected]
            rr.log("detected_obstacles", rr.Boxes3D(
                centers=centers, half_sizes=halfs, colors=colors, labels=labels))
        else:
            rr.log("detected_obstacles", rr.Clear(recursive=True))

        # Trail (orange)
        trail.append([robot_x, robot_y, 0.05])
        if len(trail) >= 2:
            rr.log("robot/trail", rr.LineStrips3D(
                [trail], colors=[[255, 102, 0, 255]], radii=[0.1]))

        # Status text
        level = rr.TextLogLevel.WARN if req_type in (1, 3) else rr.TextLogLevel.INFO
        rr.log("status", rr.TextLog(
            f"step={step:03d} req={req_name} d={best_d:+.1f} "
            f"obs={len(detected)} closest={closest_dist:.1f}m",
            level=level))

        # Progress bar
        bar_len = 30
        filled = int(bar_len * step / num_steps)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"\r  [{bar}] {step+1}/{num_steps}  req={req_name:<8} d={best_d:+.2f}  "
              f"obs={len(detected)}", end="", flush=True)

        # Move robot
        if req_type == 0:
            robot_x, robot_y, robot_theta = robot_step(robot_x, robot_y, robot_theta, px, py)
        elif req_type == 3:
            pass  # AEB — hold position

        time.sleep(0.05)

    print(f"\n\nDone! Check the Rerun viewer.")
    print("  Red cloud  = real PCD data (table_scene_lms400.pcd)")
    print("  Green boxes = detected obstacle clusters")
    print("  Blue line   = planned path")
    print("  Orange line = actual robot trail")
    print("  White line  = reference road lane")
    time.sleep(2)


if __name__ == "__main__":
    main()
