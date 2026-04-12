#!/usr/bin/env python3
"""pub-road — Python port of dora-nav/map/pub_road/src/pubroad.cpp

Reads Waypoints.txt and publishes the full road lane as a binary float array
on each tick event (200ms).

Output format (road_lane):
  [x0, x1, ..., xN, y0, y1, ..., yN] as contiguous float32 bytes
  Total size: num_waypoints * 2 * 4 bytes
"""

import os
import numpy as np
import pyarrow as pa
from dora import Node


def load_waypoints(filepath):
    """Load waypoints from text file (x y per line)."""
    points_x = []
    points_y = []
    try:
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    points_x.append(float(parts[0]))
                    points_y.append(float(parts[1]))
        print(f"[pub-road] Loaded {len(points_x)} waypoints from {filepath}")
    except FileNotFoundError:
        print(f"[pub-road] ERROR: Waypoints file not found: {filepath}")
    return np.array(points_x, dtype=np.float32), np.array(points_y, dtype=np.float32)


def main():
    node = Node()

    waypoints_file = os.environ.get("WAYPOINTS_FILE", "Waypoints.txt")
    x_ref, y_ref = load_waypoints(waypoints_file)

    if len(x_ref) == 0:
        print("[pub-road] No waypoints loaded, exiting")
        return

    # Pre-serialize: [all x floats, all y floats] as contiguous bytes
    road_lane_data = np.concatenate([x_ref, y_ref]).tobytes()
    road_lane_arr = pa.array(list(road_lane_data), type=pa.uint8())

    print(f"[pub-road] Publishing road_lane ({len(x_ref)} waypoints, {len(road_lane_data)} bytes) on tick")

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "tick":
            node.send_output("road_lane", road_lane_arr)
        elif event["type"] == "STOP":
            break

    print("[pub-road] Stopped")


if __name__ == "__main__":
    main()
