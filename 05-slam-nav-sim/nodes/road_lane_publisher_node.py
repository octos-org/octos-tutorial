#!/usr/bin/env python3
"""road-lane-publisher — Python port of dora-nav/map/road_line_publisher

Receives road_lane (waypoint array) and cur_pose (Pose2D_h), converts the
robot's current position to Frenet coordinates (s, d), and publishes
CurPose_h for the planning node.

Input formats:
  road_lane: [x0..xN, y0..yN] as float32 bytes
  cur_pose:  Pose2D_h = 12 bytes (3 x float32: x, y, theta_deg)

Output format:
  cur_pose_all: CurPose_h = 40 bytes (5 x float64: x, y, theta_rad, s, d)
"""

import math
import struct

import numpy as np
import pyarrow as pa
from dora import Node


# Cached road lane data
_maps_x = None
_maps_y = None
_maps_s = None


def compute_cumulative_s(maps_x, maps_y):
    """Compute cumulative arc-length for each waypoint."""
    s = np.zeros(len(maps_x))
    for i in range(1, len(maps_x)):
        dx = maps_x[i] - maps_x[i - 1]
        dy = maps_y[i] - maps_y[i - 1]
        s[i] = s[i - 1] + math.sqrt(dx * dx + dy * dy)
    return s


def next_waypoint(x, y, maps_x, maps_y):
    """Find the index of the closest waypoint ahead of (x, y)."""
    n = len(maps_x)
    # Find closest point
    dx = maps_x - x
    dy = maps_y - y
    dists = dx * dx + dy * dy
    closest = int(np.argmin(dists))

    # Check if target is ahead using dot product with segment direction
    cx = maps_x[closest]
    cy = maps_y[closest]

    if closest < n - 1:
        # Vector from closest to next
        seg_x = maps_x[closest + 1] - cx
        seg_y = maps_y[closest + 1] - cy
        # Vector from closest to point
        pt_x = x - cx
        pt_y = y - cy
        dot = pt_x * seg_x + pt_y * seg_y
        if dot > 0:
            return closest + 1
    return closest


def get_frenet(x, y, maps_x, maps_y, maps_s):
    """Convert Cartesian (x, y) to Frenet (s, d) coordinates."""
    n = len(maps_x)
    if n < 2:
        return 0.0, 0.0

    nwp = next_waypoint(x, y, maps_x, maps_y)
    if nwp == 0:
        nwp = 1

    prev = nwp - 1

    # Segment vector
    line_x = maps_x[nwp] - maps_x[prev]
    line_y = maps_y[nwp] - maps_y[prev]

    # Vector from segment start to point
    qx = x - maps_x[prev]
    qy = y - maps_y[prev]

    # Project point onto segment
    line_len_sq = line_x * line_x + line_y * line_y
    if line_len_sq < 1e-12:
        proj_norm = 0.0
    else:
        proj_norm = (qx * line_x + qy * line_y) / line_len_sq

    proj_x = proj_norm * line_x
    proj_y = proj_norm * line_y

    # Lateral distance d (signed via cross product)
    cross = qx * line_y - qy * line_x
    sign = 1.0 if cross > 0 else -1.0
    d_dist = math.sqrt((qx - proj_x) ** 2 + (qy - proj_y) ** 2)
    frenet_d = sign * d_dist

    # Longitudinal distance s
    frenet_s = maps_s[prev] + math.sqrt(proj_x * proj_x + proj_y * proj_y)

    return frenet_s, frenet_d


def main():
    global _maps_x, _maps_y, _maps_s

    node = Node()
    print("[road-lane-pub] Waiting for road_lane and cur_pose...")

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
            if _maps_s is not None:
                print(f"[road-lane-pub] Road lane: {n} waypoints, "
                      f"total length {_maps_s[-1]:.1f}m")

        elif eid == "cur_pose":
            if _maps_x is None:
                continue
            if len(raw) < 12:
                continue

            x, y, theta_deg = struct.unpack('<fff', raw[:12])
            theta_rad = theta_deg * math.pi / 180.0

            s, d = get_frenet(x, y, _maps_x, _maps_y, _maps_s)

            # Pack CurPose_h: 5 x float64 = 40 bytes
            cur_pose_all = struct.pack('<5d', x, y, theta_rad, s, d)
            node.send_output(
                "cur_pose_all",
                pa.array(list(cur_pose_all), type=pa.uint8()),
            )

    print("[road-lane-pub] Stopped")


if __name__ == "__main__":
    main()
