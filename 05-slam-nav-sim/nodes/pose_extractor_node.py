#!/usr/bin/env python3
"""pose-extractor — Extracts ground_truth_pose (Pose2D_h) from joint_positions.

The local dora-mujoco version only outputs joint_positions (qpos array).
The epyc version also outputs ground_truth_pose. This node bridges the gap
by extracting x, y, theta from qpos and formatting as Pose2D_h.

Hunter SE qpos layout (20 DOF):
  [0:3]  = base position (x, y, z)
  [3:7]  = base quaternion (w, x, y, z)
  [7:13] = wheel joints (6)
  [13:20] = arm joints (7)

Output: Pose2D_h = 12 bytes (float32 x, y, theta_degrees)
"""

import math
import struct

import numpy as np
import pyarrow as pa
from dora import Node


def quat_to_yaw(qw, qx, qy, qz):
    """Convert quaternion to yaw angle (radians)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def main():
    node = Node()
    print("[pose-extractor] Waiting for joint_positions...")
    count = 0

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] != "joint_positions":
            continue

        qpos = event["value"].to_numpy()
        if len(qpos) < 7:
            continue

        x = float(qpos[0])
        y = float(qpos[1])
        # Quaternion: MuJoCo uses (w, x, y, z) order
        qw, qx, qy, qz = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
        yaw_rad = quat_to_yaw(qw, qx, qy, qz)
        theta_deg = yaw_rad * 180.0 / math.pi

        # Pack as Pose2D_h (3 x float32 = 12 bytes)
        pose = struct.pack('<fff', x, y, theta_deg)
        node.send_output(
            "ground_truth_pose",
            pa.array(list(pose), type=pa.uint8()),
        )

        count += 1
        if count <= 3:
            print(f"[pose-extractor] Pose: x={x:.3f} y={y:.3f} θ={theta_deg:.1f}°")

    print("[pose-extractor] Stopped")


if __name__ == "__main__":
    main()
