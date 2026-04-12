#!/usr/bin/env python3
"""Rerun visualization node — Python replacement for dora-nav's C++ rerun node.

Replicates the C++ points_to_rerun.cpp visualization:
  - Red points:    Static map (from data/map.pcd)
  - White line:    Global waypoints (from Waypoints.txt)
  - Green points:  Live LiDAR point cloud (in sensor frame, transformed via entity)
  - Blue line:     Planned path (from planning node)
  - Yellow box:    Robot body
  - Cyan arrow:    Robot heading direction
  - Orange trail:  Robot trajectory history
"""

import math
import os
import struct
from collections import deque

import numpy as np
import rerun as rr
from dora import Node


# ---- Configuration ----
WAYPOINTS_FILE = os.environ.get("WAYPOINTS_FILE", "Waypoints.txt")
MAP_PCD_FILE = os.environ.get("MAP_PCD_FILE", "")
MAX_TRAIL_POINTS = 500


def load_waypoints(filepath):
    """Load waypoints from text file (x y per line) → Nx3 float array."""
    points = []
    try:
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    points.append([float(parts[0]), float(parts[1]), 0.0])
        print(f"[rerun] Loaded {len(points)} waypoints from {filepath}")
    except FileNotFoundError:
        print(f"[rerun] WARNING: Waypoints file not found: {filepath}")
    return np.array(points, dtype=np.float32) if points else None


def load_pcd(filepath):
    """Load PCD file (ascii or binary) → Nx3 float array."""
    if not filepath:
        return None
    points = []
    try:
        with open(filepath, "rb") as f:
            num_points = 0
            data_type = "ascii"
            fields = []
            for raw_line in f:
                text = raw_line.decode("ascii", errors="ignore").strip()
                if text.startswith("FIELDS"):
                    fields = text.split()[1:]
                elif text.startswith("POINTS"):
                    num_points = int(text.split()[1])
                elif text.startswith("DATA"):
                    data_type = text.split()[1].lower()
                    break

            if data_type == "ascii":
                for raw_line in f:
                    parts = raw_line.decode("ascii", errors="ignore").strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif data_type == "binary":
                raw = f.read()
                # Determine stride from fields (typically x,y,z,intensity = 16 bytes)
                stride = max(16, len(fields) * 4)
                n = min(num_points, len(raw) // stride) if num_points > 0 else len(raw) // stride
                for i in range(n):
                    x, y, z = struct.unpack_from("<fff", raw, i * stride)
                    points.append([x, y, z])

        print(f"[rerun] Loaded {len(points)} map points from {filepath}")
    except FileNotFoundError:
        print(f"[rerun] WARNING: Map PCD not found: {filepath}")
    except Exception as e:
        print(f"[rerun] WARNING: Failed to load PCD: {e}")
    return np.array(points, dtype=np.float32) if points else None


def main():
    node = Node()

    # Initialize Rerun viewer
    rr.init("lidarpoints_viewer", spawn=True)

    # Pose state
    pose_x, pose_y, pose_theta = 0.0, 0.0, 0.0
    trail = deque(maxlen=MAX_TRAIL_POINTS)
    pc_count = 0  # debug counter

    # ---- Load static assets ----
    waypoints = load_waypoints(WAYPOINTS_FILE)
    if waypoints is not None and len(waypoints) > 0:
        rr.log("global_path_points", rr.LineStrips3D(
            [waypoints], colors=[[255, 255, 255, 255]], radii=[0.08]))

    map_pts = load_pcd(MAP_PCD_FILE)
    if map_pts is not None and len(map_pts) > 0:
        rr.log("map_points_static", rr.Points3D(
            map_pts, colors=np.full((len(map_pts), 4), [255, 0, 0, 255], dtype=np.uint8),
            radii=[0.02]))

    print("[rerun] Visualization ready — waiting for data...")

    # ---- Main event loop ----
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        raw = bytes(event["value"].to_pylist())

        # ---- Pose update ----
        if eid == "cur_pose" and len(raw) >= 12:
            pose_x, pose_y, pose_theta = struct.unpack("<fff", raw[:12])
            angle_rad = pose_theta * math.pi / 180.0

            # Direction arrow (cyan, 4m)
            dx = 4.0 * math.cos(angle_rad)
            dy = 4.0 * math.sin(angle_rad)
            rr.log("robot/arrow", rr.Arrows3D(
                origins=[[pose_x, pose_y, 0.2]],
                vectors=[[dx, dy, 0.0]],
                colors=[[0, 255, 255, 255]]))

            # Body (yellow box, rotated)
            rr.log("robot/body", rr.Boxes3D(
                centers=[[pose_x, pose_y, 0.1]],
                half_sizes=[[0.6, 0.4, 0.25]],
                rotation_axis_angles=[rr.RotationAxisAngle(
                    axis=[0, 0, 1], angle=angle_rad)],
                colors=[[255, 255, 0, 255]],
                labels=["Robot"]))

            # Trajectory trail (orange)
            trail.append([pose_x, pose_y, 0.05])
            if len(trail) >= 2:
                trail_pts = np.array(list(trail), dtype=np.float32)
                rr.log("robot/trail", rr.LineStrips3D(
                    [trail_pts], colors=[[255, 102, 0, 255]], radii=[0.1]))

        # ---- Live point cloud ----
        elif eid == "pointcloud":
            pc_count += 1
            if pc_count <= 3:
                print(f"[rerun] pointcloud #{pc_count}: {len(raw)} bytes")

            if len(raw) > 16:
                n_points = (len(raw) - 16) // 16
                if n_points > 0:
                    # Parse points in sensor (local) frame
                    pts = np.zeros((n_points, 3), dtype=np.float32)
                    for i in range(n_points):
                        off = 16 + i * 16
                        pts[i, 0], pts[i, 1], pts[i, 2] = struct.unpack_from("<fff", raw, off)

                    if pc_count <= 3:
                        print(f"[rerun]   → {n_points} points, first: ({pts[0,0]:.2f}, {pts[0,1]:.2f}, {pts[0,2]:.2f})")

                    # Transform to global frame (same as C++ — rotation by theta + translation)
                    angle_rad = pose_theta * math.pi / 180.0
                    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                    global_pts = np.empty_like(pts)
                    global_pts[:, 0] = pts[:, 0] * cos_a - pts[:, 1] * sin_a + pose_x
                    global_pts[:, 1] = pts[:, 0] * sin_a + pts[:, 1] * cos_a + pose_y
                    global_pts[:, 2] = pts[:, 2]

                    rr.log("live_points", rr.Points3D(
                        global_pts,
                        colors=np.full((n_points, 4), [0, 255, 0, 255], dtype=np.uint8),
                        radii=[0.02]))

        # ---- Planned path ----
        elif eid == "raw_path":
            n_floats = len(raw) // 4
            if n_floats >= 4:
                floats = np.frombuffer(raw[:n_floats * 4], dtype=np.float32)
                n_pts = n_floats // 2
                path_pts = np.zeros((n_pts, 3), dtype=np.float32)
                path_pts[:, 0] = floats[:n_pts]
                path_pts[:, 1] = floats[n_pts:2 * n_pts]

                # Transform from vehicle frame to global
                # Vehicle forward = +Y in local, rotate by (theta - 90°)
                angle_rad = (pose_theta - 90.0) * math.pi / 180.0
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                global_path = np.empty_like(path_pts)
                global_path[:, 0] = path_pts[:, 0] * cos_a - path_pts[:, 1] * sin_a + pose_x
                global_path[:, 1] = path_pts[:, 0] * sin_a + path_pts[:, 1] * cos_a + pose_y
                global_path[:, 2] = 0.0

                rr.log("path_points", rr.LineStrips3D(
                    [global_path], colors=[[0, 0, 255, 255]], radii=[0.08]))

        # ---- Obstacle boxes ----
        elif eid == "obstacle_list":
            if len(raw) < 4:
                continue
            num_obs = struct.unpack_from('<I', raw, 0)[0]
            if num_obs == 0:
                rr.log("obstacles", rr.Clear(recursive=True))
                continue

            centers = []
            half_sizes = []
            colors = []
            labels = []
            for i in range(num_obs):
                off = 4 + i * 44
                if off + 44 > len(raw):
                    break
                vals = struct.unpack_from('<I2f2f2f3ff', raw, off)
                obs_id, ox, oy, os_val, od, ovx, ovy, ol, ow, oh, odist = vals
                centers.append([ox, oy, oh / 2.0])
                half_sizes.append([ol / 2.0, ow / 2.0, oh / 2.0])
                colors.append([255, 0, 0, 200])
                labels.append(f"#{obs_id} d={odist:.1f}m")

            if centers:
                rr.log("obstacles", rr.Boxes3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    colors=colors,
                    labels=labels,
                ))

        # ---- Costmap heatmap ----
        elif eid == "costmap_grid":
            if len(raw) < 28:
                continue
            origin_x, origin_y, resolution = struct.unpack_from('<fff', raw, 0)
            width, height = struct.unpack_from('<II', raw, 12)
            robot_x, robot_y = struct.unpack_from('<ff', raw, 20)

            grid_start = 28
            grid_size = width * height
            if len(raw) < grid_start + grid_size:
                continue

            grid = np.frombuffer(raw[grid_start:grid_start + grid_size], dtype=np.uint8)
            grid = grid.reshape((height, width))

            occupied = np.argwhere(grid > 0)
            if len(occupied) == 0:
                rr.log("costmap", rr.Clear(recursive=True))
                continue

            cell_y = occupied[:, 0]
            cell_x = occupied[:, 1]
            gx = origin_x + (cell_x + 0.5) * resolution
            gy = origin_y + (cell_y + 0.5) * resolution
            gz = np.full_like(gx, -0.05)

            costs = grid[cell_y, cell_x]

            # Yellow for inflated (cost < 254), red for fully occupied (cost >= 254)
            costmap_colors = np.zeros((len(costs), 4), dtype=np.uint8)
            occupied_mask = costs >= 254
            costmap_colors[occupied_mask] = [255, 0, 0, 180]
            costmap_colors[~occupied_mask] = [255, 255, 0, 100]

            pts = np.stack([gx, gy, gz], axis=1).astype(np.float32)
            rr.log("costmap", rr.Points3D(
                pts, colors=costmap_colors, radii=[resolution / 2.0]))

    print("[rerun] Node stopped")


if __name__ == "__main__":
    main()
