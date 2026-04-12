#!/usr/bin/env python3
"""planning — Python port of dora-nav/planning/routing_planning

Frenet-based path planner: receives road lane + current pose + road attributes,
outputs a local path (30 waypoints in vehicle frame) and a speed request.

Input formats:
  road_lane:      [x0..xN, y0..yN] as float32
  cur_pose_all:   CurPose_h = 40 bytes (5 x float64: x, y, theta, s, d)
  road_attri_msg: RoadAttri_h = 40 bytes (10 x float32)

Output formats:
  raw_path: [x0..x29, y0..y29] as float32 (240 bytes)
  Request:  Request_h = 14 bytes (uint8 type, float speed, float stop_dist, float aeb_dist)
"""

import math
import struct

import numpy as np
import pyarrow as pa
from dora import Node


# Vehicle parameters
TRANS_PARA_BACK = 0.25  # rear axle offset from center (m)
NUM_PATH_POINTS = 30

# Cached state
_maps_x = None
_maps_y = None
_maps_s = None
_cur_pose = None  # (x, y, theta, s, d)
_run_speed = 320.0  # from road_attri_msg velocity field


def compute_cumulative_s(maps_x, maps_y):
    s = np.zeros(len(maps_x))
    for i in range(1, len(maps_x)):
        dx = maps_x[i] - maps_x[i - 1]
        dy = maps_y[i] - maps_y[i - 1]
        s[i] = s[i - 1] + math.sqrt(dx * dx + dy * dy)
    return s


def get_xy_from_frenet(s_val, d_val, maps_s, maps_x, maps_y):
    """Convert Frenet (s, d) to Cartesian (x, y)."""
    n = len(maps_s)
    if n < 2:
        return 0.0, 0.0

    # Clamp s to valid range
    if s_val <= maps_s[0]:
        s_val = maps_s[0] + 0.001
    if s_val >= maps_s[-1]:
        return float(maps_x[-1]), float(maps_y[-1])

    # Find segment containing s
    prev_wp = 0
    for i in range(n - 1):
        if maps_s[i] <= s_val < maps_s[i + 1]:
            prev_wp = i
            break
    else:
        prev_wp = n - 2

    nxt = prev_wp + 1

    # Heading of segment
    heading = math.atan2(
        maps_y[nxt] - maps_y[prev_wp],
        maps_x[nxt] - maps_x[prev_wp],
    )

    # Point on reference line at s
    seg_s = s_val - maps_s[prev_wp]
    seg_x = maps_x[prev_wp] + seg_s * math.cos(heading)
    seg_y = maps_y[prev_wp] + seg_s * math.sin(heading)

    # Offset perpendicular by d
    perp = heading - math.pi / 2.0
    x = seg_x + d_val * math.cos(perp)
    y = seg_y + d_val * math.sin(perp)

    return x, y


def plan_path(cur_x, cur_y, cur_theta, cur_s, cur_d, maps_s, maps_x, maps_y, speed):
    """Generate 30 path points in vehicle frame using Frenet planning."""
    # Planning distance based on speed
    vel_mps = speed * 0.00277  # rough conversion factor
    plan_dist = 15.0 + vel_mps * 0.277

    # Adjust pose to rear axle
    rear_x = cur_x - TRANS_PARA_BACK * math.cos(cur_theta)
    rear_y = cur_y - TRANS_PARA_BACK * math.sin(cur_theta)

    map_end_s = maps_s[-1] if len(maps_s) > 0 else 0.0

    # Sample 30 points along the path from current s
    path_x_global = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    path_y_global = np.zeros(NUM_PATH_POINTS, dtype=np.float32)

    for i in range(NUM_PATH_POINTS):
        target_s = cur_s + i * (plan_dist / 50.0)
        target_d = 0.0  # stay centered (no lane change)

        if target_s >= map_end_s:
            path_x_global[i] = float(maps_x[-1])
            path_y_global[i] = float(maps_y[-1])
        else:
            gx, gy = get_xy_from_frenet(target_s, target_d, maps_s, maps_x, maps_y)
            path_x_global[i] = gx
            path_y_global[i] = gy

    # Transform to vehicle frame (rear axle)
    sin_yaw = math.sin(cur_theta)
    cos_yaw = math.cos(cur_theta)

    path_x_veh = np.zeros(NUM_PATH_POINTS, dtype=np.float32)
    path_y_veh = np.zeros(NUM_PATH_POINTS, dtype=np.float32)

    for i in range(NUM_PATH_POINTS):
        shift_x = path_x_global[i] - rear_x
        shift_y = path_y_global[i] - rear_y
        # Rotation from map frame to vehicle frame
        path_x_veh[i] = shift_x * sin_yaw - shift_y * cos_yaw
        path_y_veh[i] = shift_y * sin_yaw + shift_x * cos_yaw

    return path_x_veh, path_y_veh


def main():
    global _maps_x, _maps_y, _maps_s, _cur_pose, _run_speed

    node = Node()
    print("[planning] Frenet path planner ready")

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        raw = bytes(event["value"].to_pylist())

        if eid == "road_lane":
            num_floats = len(raw) // 4
            if num_floats < 4:
                continue
            floats = np.frombuffer(raw[:num_floats * 4], dtype=np.float32)
            n = num_floats // 2
            _maps_x = floats[:n].astype(np.float64)
            _maps_y = floats[n:2 * n].astype(np.float64)
            _maps_s = compute_cumulative_s(_maps_x, _maps_y)

        elif eid == "cur_pose_all":
            if len(raw) < 40:
                continue
            _cur_pose = struct.unpack('<5d', raw[:40])

        elif eid == "road_attri_msg":
            if len(raw) >= 4:
                _run_speed = struct.unpack_from('<f', raw, 0)[0]

            # Plan on each road_attri tick (100ms) if we have all data
            if _maps_x is None or _cur_pose is None:
                continue

            cx, cy, ctheta, cs, cd = _cur_pose

            path_x, path_y = plan_path(
                cx, cy, ctheta, cs, cd,
                _maps_s, _maps_x, _maps_y, _run_speed,
            )

            # Output raw_path: [x0..x29, y0..y29] as float32
            raw_path = np.concatenate([path_x, path_y]).tobytes()
            node.send_output(
                "raw_path",
                pa.array(list(raw_path), type=pa.uint8()),
            )

            # Output Request: forward at run_speed
            # Request_h: uint8 type=0 (FORWARD), float run_speed, float stop_dist=0, float aeb_dist=0
            request = struct.pack('<Bfff', 0, _run_speed, 0.0, 0.0)
            node.send_output(
                "Request",
                pa.array(list(request), type=pa.uint8()),
            )

    print("[planning] Stopped")


if __name__ == "__main__":
    main()
