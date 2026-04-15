#!/usr/bin/env python3
"""lat-control — Python port of dora-nav/control/vehicle_control/lat_controller

Pure Pursuit lateral controller: receives a local path (vehicle frame) and
outputs a steering angle command.

Input format:
  raw_path: [x0..x29, y0..y29] as float32 (240 bytes)

Output format:
  SteeringCmd: SteeringCmd_h = 4 bytes (float32 angle in degrees)
"""

import math
import struct

import numpy as np
import pyarrow as pa
from dora import Node


# Vehicle parameters (matching C++ defaults for Hunter SE)
WHEELBASE = 0.5        # axle distance (m)
FRONT_TREAD = 0.3      # front wheel track (m)
MAX_WHEEL_ANGLE = 45.0  # degrees
LOOKAHEAD_DIST = 1.0   # base lookahead distance (m)

# Low-pass filter coefficient
FILTER_ALPHA = 0.2     # new = 0.8 * last + 0.2 * raw


def pure_pursuit(path_x, path_y, la_dist):
    """Compute steering angle via Pure Pursuit algorithm.

    path_x, path_y are in vehicle frame (rear axle origin, +Y forward).
    Returns steering angle in degrees.
    """
    n = len(path_x)
    if n < 2:
        return 0.0

    # Find lookahead point: first point at distance >= la_dist from origin
    goal_idx = 0
    for i in range(n):
        dist = math.sqrt(path_x[i] ** 2 + path_y[i] ** 2)
        if dist >= la_dist:
            goal_idx = i
            break
    else:
        goal_idx = n - 1

    gx = path_x[goal_idx]
    gy = path_y[goal_idx]

    # Ackermann steering geometry
    # tgangle = angle from vehicle forward (+Y) axis to goal point
    # C++ original: atan2(goal_y, goal_x) - pi/2
    tgangle = math.atan2(gy, gx) - math.pi / 2.0

    # Pure pursuit: delta = atan2(2 * L * sin(alpha), La)
    sin_tg = math.sin(tgangle)
    if abs(la_dist) < 1e-6:
        return 0.0
    delta = math.atan2(2.0 * WHEELBASE * sin_tg, la_dist)

    # Inner wheel angle (Ackermann correction)
    tan_delta = math.tan(delta)
    denom = WHEELBASE - (FRONT_TREAD / 2.0) * abs(tan_delta)
    if abs(denom) < 1e-6:
        deltai = delta
    else:
        deltai = math.atan2(WHEELBASE * tan_delta, denom)

    return deltai * 180.0 / math.pi


def main():
    node = Node()
    last_angle = 0.0

    print("[lat-control] Pure Pursuit lateral controller ready")

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] != "raw_path":
            continue

        raw = bytes(event["value"].to_pylist())
        num_floats = len(raw) // 4
        if num_floats < 4:
            continue

        floats = np.frombuffer(raw[:num_floats * 4], dtype=np.float32)
        n = num_floats // 2
        path_x = floats[:n]
        path_y = floats[n:2 * n]

        # Compute raw steering angle
        raw_angle = pure_pursuit(path_x, path_y, LOOKAHEAD_DIST)

        # Clamp to max
        raw_angle = max(-MAX_WHEEL_ANGLE, min(MAX_WHEEL_ANGLE, raw_angle))

        # Low-pass filter
        filtered = (1.0 - FILTER_ALPHA) * last_angle + FILTER_ALPHA * raw_angle
        last_angle = filtered

        # Pack SteeringCmd_h: 4 bytes float32
        cmd = struct.pack('<f', filtered)
        node.send_output(
            "SteeringCmd",
            pa.array(list(cmd), type=pa.uint8()),
        )

    print("[lat-control] Stopped")


if __name__ == "__main__":
    main()
