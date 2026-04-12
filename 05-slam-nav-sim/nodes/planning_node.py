#!/usr/bin/env python3
"""planning — Python port of dora-nav/planning/routing_planning

Frenet-based path planner with three-phase obstacle avoidance: receives road
lane, current pose, road attributes, and optional obstacle/costmap/recovery
inputs; outputs a local path (30 waypoints in vehicle frame) and a speed
request.

Input formats:
  road_lane:      [x0..xN, y0..yN] as float32
  cur_pose_all:   CurPose_h = 40 bytes (5 x float64: x, y, theta, s, d)
  road_attri_msg: RoadAttri_h = 40 bytes (10 x float32)
  obstacle_list:  uint32 count + N * 44-byte obstacle structs
                  each: '<I2f2f2f3ff' = (id, x, y, s, d, vx, vy, l, w, h, dist)
  costmap_grid:   raw costmap bytes (cached, reserved for future use)
  recovery_cmd:   '<Bf' = (state: 0=NORMAL 1=WAITING 2=BACKING_UP 3=STOPPED,
                            backup_speed)

Output formats:
  raw_path: [x0..x29, y0..y29] as float32 (240 bytes)
  Request:  Request_h = 14 bytes (uint8 type, float speed, float stop_dist,
                                   float aeb_dist)
            type: 0=FORWARD, 1=STOP, 2=BACK, 3=AEB
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

# Obstacle avoidance state
_obstacles = []       # list of (id, x, y, s, d, vx, vy, length, width, height, distance)
_costmap_data = None  # raw costmap bytes (reserved for future use)
_recovery_state = 0   # 0=NORMAL, 1=WAITING, 2=BACKING_UP, 3=STOPPED
_backup_speed = 0.0


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


def plan_path_with_d(cur_x, cur_y, cur_theta, cur_s, cur_d, target_d,
                     maps_s, maps_x, maps_y, speed):
    """Generate path points with smooth lateral offset transition to target_d.

    Uses a CubicSpline to blend from cur_d to target_d over ~3 m, then holds
    target_d for the remainder of the planning horizon.
    """
    from scipy.interpolate import CubicSpline

    vel_mps = speed * 0.00277
    plan_dist = 15.0 + vel_mps * 0.277
    rear_x = cur_x - TRANS_PARA_BACK * math.cos(cur_theta)
    rear_y = cur_y - TRANS_PARA_BACK * math.sin(cur_theta)
    map_end_s = maps_s[-1] if len(maps_s) > 0 else 0.0

    # Smooth d-profile: transition from cur_d to target_d over ~3 m
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

    # Transform to vehicle frame (rear axle)
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


def main():
    global _maps_x, _maps_y, _maps_s, _cur_pose, _run_speed
    global _obstacles, _costmap_data, _recovery_state, _backup_speed

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

        elif eid == "obstacle_list":
            if len(raw) < 4:
                continue
            num_obs = struct.unpack_from('<I', raw, 0)[0]
            _obstacles = []
            for i in range(num_obs):
                off = 4 + i * 44  # struct '<I2f2f2f3ff' = 44 bytes
                if off + 44 > len(raw):
                    break
                vals = struct.unpack_from('<I2f2f2f3ff', raw, off)
                _obstacles.append(vals)

        elif eid == "costmap_grid":
            _costmap_data = raw

        elif eid == "recovery_cmd":
            if len(raw) >= 5:
                _recovery_state, _backup_speed = struct.unpack_from('<Bf', raw, 0)

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

            # Serialize default raw_path from plan_path()
            raw_path = np.concatenate([path_x, path_y]).tobytes()

            # ---- Phase 3: Recovery override ----
            if _recovery_state == 1:   # WAITING
                request = struct.pack('<Bfff', 1, 0.0, 0.0, 0.0)
                node.send_output("Request", pa.array(list(request), type=pa.uint8()))
                node.send_output("raw_path", pa.array(list(raw_path), type=pa.uint8()))
                continue
            elif _recovery_state == 2:  # BACKING_UP
                request = struct.pack('<Bfff', 2, _backup_speed, 0.0, 0.0)
                node.send_output("Request", pa.array(list(request), type=pa.uint8()))
                node.send_output("raw_path", pa.array(list(raw_path), type=pa.uint8()))
                continue
            elif _recovery_state == 3:  # STOPPED
                request = struct.pack('<Bfff', 1, 0.0, 0.0, 0.0)
                node.send_output("Request", pa.array(list(request), type=pa.uint8()))
                node.send_output("raw_path", pa.array(list(raw_path), type=pa.uint8()))
                continue

            # ---- Phase 2: d-offset lateral avoidance ----
            # Only offset while obstacle is CLOSE AHEAD (within avoidance
            # window).  Once past, best_d returns to 0 so robot merges back.
            aeb_distance = 5.0
            road_half_width = 2.0
            avoidance_window = 5.0   # swerve for obstacles within 5m ahead
            obstacle_half_w = 1.0    # each obstacle blocks ±1m laterally

            candidates = [d for d in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
                          if abs(d) <= road_half_width]
            best_d = 0.0
            best_cost = float('inf')
            all_blocked = True

            if _obstacles:
                for d_cand in candidates:
                    obstacle_cost = 0.0
                    blocked = False
                    for obs in _obstacles:
                        obs_s, obs_d = obs[3], obs[4]
                        s_ahead = obs_s - cs
                        if 0 < s_ahead < avoidance_window:
                            lateral_dist = abs(obs_d - d_cand)
                            if lateral_dist < obstacle_half_w:
                                blocked = True
                                obstacle_cost += 10.0
                            else:
                                obstacle_cost += 0.5 / lateral_dist
                    deviation_cost = abs(d_cand) * 2.0
                    smoothness_cost = abs(d_cand - cd) * 0.5
                    total = obstacle_cost + deviation_cost + smoothness_cost
                    if not blocked:
                        all_blocked = False
                    if total < best_cost:
                        best_cost = total
                        best_d = d_cand

            # ---- Phase 1: AEB — triggers when ALL candidates blocked ----
            closest_obs_dist = float('inf')
            for obs in _obstacles:
                obs_s, obs_d, obs_dist = obs[3], obs[4], obs[10]
                s_ahead = obs_s - cs
                if 0 < s_ahead < aeb_distance:
                    if obs_dist < closest_obs_dist:
                        closest_obs_dist = obs_dist

            if all_blocked and _obstacles and closest_obs_dist < 3.0:
                request = struct.pack('<Bfff', 3, _run_speed, 0.0, closest_obs_dist)
                node.send_output("raw_path", pa.array(list(raw_path), type=pa.uint8()))
                node.send_output("Request", pa.array(list(request), type=pa.uint8()))
                continue

            # Replan with smooth lateral offset when avoidance is needed
            if abs(best_d) > 0.01:
                path_x, path_y = plan_path_with_d(
                    cx, cy, ctheta, cs, cd, best_d,
                    _maps_s, _maps_x, _maps_y, _run_speed,
                )
                raw_path = np.concatenate([path_x, path_y]).tobytes()

            # Normal forward
            node.send_output("raw_path", pa.array(list(raw_path), type=pa.uint8()))
            request = struct.pack('<Bfff', 0, _run_speed, 0.0, 0.0)
            node.send_output("Request", pa.array(list(request), type=pa.uint8()))

    print("[planning] Stopped")


if __name__ == "__main__":
    main()
