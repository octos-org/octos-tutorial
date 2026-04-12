#!/usr/bin/env python3
"""costmap_node — Phase 2 DORA node: rolling-window 2D occupancy costmap.

Inputs:  pointcloud (16B header + Nx16B float32 points), ground_truth_pose (12B Pose2D_h)
Output:  costmap_grid — 28-byte header + WxH uint8 cells (0=free, 1–253=inflated, 254=occupied)

Header struct '<ff f II ff' (28 bytes):
  origin_x, origin_y, resolution, width, height, robot_x, robot_y
"""

import math
import os
import struct

import numpy as np
import pyarrow as pa
from dora import Node

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
COSTMAP_SIZE_M: float = float(os.environ.get("COSTMAP_SIZE_M", "20.0"))
COSTMAP_RESOLUTION: float = float(os.environ.get("COSTMAP_RESOLUTION", "0.1"))
INFLATION_RADIUS: float = float(os.environ.get("INFLATION_RADIUS", "0.5"))
GROUND_Z_THRESHOLD: float = 0.15  # metres — points below this are ground

GRID_CELLS: int = int(round(COSTMAP_SIZE_M / COSTMAP_RESOLUTION))  # e.g. 200
HALF_SIZE_M: float = COSTMAP_SIZE_M / 2.0

HEADER_STRUCT = struct.Struct("<ff f II ff")  # 28 bytes
HEADER_SIZE = HEADER_STRUCT.size  # 28


# ---------------------------------------------------------------------------
# Precompute inflation kernel
# ---------------------------------------------------------------------------
def _build_inflation_kernel(radius_m: float, resolution: float) -> np.ndarray:
    """2-D uint8 kernel: linear cost decay 253→1 over inflation radius."""
    r_cells = int(math.ceil(radius_m / resolution))
    size = 2 * r_cells + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    cy = cx = r_cells
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            d = math.sqrt(dx * dx + dy * dy) * resolution
            if d <= radius_m and d > 0:
                cost = int(round(253.0 * (1.0 - d / radius_m))) + 1  # 1–253
                kernel[cy + dy, cx + dx] = max(1, min(253, cost))
    return kernel


_INFLATION_KERNEL = _build_inflation_kernel(INFLATION_RADIUS, COSTMAP_RESOLUTION)
_KERNEL_HALF = _INFLATION_KERNEL.shape[0] // 2  # radius in cells


# ---------------------------------------------------------------------------
# Costmap2D
# ---------------------------------------------------------------------------
class Costmap2D:
    """Robot-centred rolling-window occupancy grid."""

    def __init__(self, grid_cells: int, resolution: float) -> None:
        self._n = grid_cells
        self._res = resolution
        self._grid: np.ndarray = np.zeros((grid_cells, grid_cells), dtype=np.uint8)

    # ------------------------------------------------------------------
    def update(self, points_global: np.ndarray, robot_x: float, robot_y: float) -> None:
        """Rebuild grid from (N,2) global-frame obstacle XY points."""
        self._grid[:] = 0

        if points_global.shape[0] == 0:
            return

        origin_x = robot_x - HALF_SIZE_M
        origin_y = robot_y - HALF_SIZE_M

        col = ((points_global[:, 0] - origin_x) / self._res).astype(np.int32)
        row = ((points_global[:, 1] - origin_y) / self._res).astype(np.int32)

        valid = (col >= 0) & (col < self._n) & (row >= 0) & (row < self._n)
        col = col[valid]
        row = row[valid]

        if col.shape[0] == 0:
            return

        occupied_rows, occupied_cols = row, col
        unique_cells = np.unique(np.stack([occupied_rows, occupied_cols], axis=1), axis=0)

        kh = _KERNEL_HALF
        kernel = _INFLATION_KERNEL

        for r_obs, c_obs in unique_cells:
            r0 = int(r_obs) - kh
            c0 = int(c_obs) - kh
            r1 = r0 + kernel.shape[0]
            c1 = c0 + kernel.shape[1]

            # Clip to grid
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

            # np.maximum keeps the highest-cost value already in the grid
            region = self._grid[gr0:gr1, gc0:gc1]
            np.maximum(region, kernel[kr0:kr1, kc0:kc1], out=region)

        # Stamp obstacle cells at cost 254 (overwrite inflation)
        self._grid[occupied_rows, occupied_cols] = 254

    # ------------------------------------------------------------------
    def to_bytes(self, robot_x: float, robot_y: float) -> bytes:
        """28-byte header + row-major grid data."""
        origin_x = robot_x - HALF_SIZE_M
        origin_y = robot_y - HALF_SIZE_M
        header = HEADER_STRUCT.pack(
            origin_x, origin_y,
            self._res,
            self._n, self._n,
            robot_x, robot_y,
        )
        return header + self._grid.tobytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_pointcloud(raw: bytes) -> np.ndarray:
    """16B header + Nx16B float32 points → (N,4) array, or empty on failure."""
    if len(raw) <= 16:
        return np.empty((0, 4), dtype=np.float32)
    n_points = (len(raw) - 16) // 16
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float32)
    data = np.frombuffer(raw, dtype=np.float32, count=n_points * 4, offset=16)
    return data.reshape(n_points, 4)


def _transform_to_global(pts_local: np.ndarray, robot_x: float, robot_y: float, theta_deg: float) -> np.ndarray:
    """Rotate + translate (N,2) local XY points into global frame."""
    theta_rad = math.radians(theta_deg)
    cos_a = math.cos(theta_rad)
    sin_a = math.sin(theta_rad)

    lx = pts_local[:, 0]
    ly = pts_local[:, 1]

    gx = lx * cos_a - ly * sin_a + robot_x
    gy = lx * sin_a + ly * cos_a + robot_y

    return np.stack([gx, gy], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    node = Node()

    costmap = Costmap2D(GRID_CELLS, COSTMAP_RESOLUTION)

    # Cached state
    _pose: tuple[float, float, float] | None = None  # (x, y, theta_deg)

    count_pc = 0
    count_out = 0

    print(
        f"[costmap] Init: size={COSTMAP_SIZE_M}m  res={COSTMAP_RESOLUTION}m  "
        f"grid={GRID_CELLS}x{GRID_CELLS}  inflation={INFLATION_RADIUS}m"
    )

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue

        eid: str = event["id"]
        raw: bytes = bytes(event["value"].to_pylist())

        # ---- Cache latest pose ----------------------------------------
        if eid == "ground_truth_pose":
            if len(raw) < 12:
                continue
            x, y, theta_deg = struct.unpack_from("<fff", raw, 0)
            _pose = (x, y, theta_deg)
            continue

        # ---- Process pointcloud ---------------------------------------
        if eid != "pointcloud":
            continue

        count_pc += 1

        if _pose is None:
            if count_pc <= 3:
                print(f"[costmap] Waiting for pose (pc #{count_pc} dropped)")
            continue

        robot_x, robot_y, theta_deg = _pose

        pts = _parse_pointcloud(raw)
        if pts.shape[0] == 0:
            continue
        pts = pts[pts[:, 2] >= GROUND_Z_THRESHOLD]  # ground removal
        if pts.shape[0] == 0:
            continue
        global_xy = _transform_to_global(pts[:, :2], robot_x, robot_y, theta_deg)
        dx = global_xy[:, 0] - robot_x
        dy = global_xy[:, 1] - robot_y
        global_xy = global_xy[(np.abs(dx) <= HALF_SIZE_M) & (np.abs(dy) <= HALF_SIZE_M)]
        costmap.update(global_xy, robot_x, robot_y)
        msg_bytes = costmap.to_bytes(robot_x, robot_y)
        node.send_output(
            "costmap_grid",
            pa.array(list(msg_bytes), type=pa.uint8()),
        )

        count_out += 1
        if count_out <= 3 or count_out % 100 == 0:
            n_obs = int(np.sum(costmap._grid == 254))
            print(
                f"[costmap] #{count_out}: robot=({robot_x:.2f},{robot_y:.2f}) "
                f"θ={theta_deg:.1f}°  pts={pts.shape[0]}→{global_xy.shape[0]}  "
                f"obstacles={n_obs}  msg={len(msg_bytes)}B"
            )

    print("[costmap] Stopped")


if __name__ == "__main__":
    main()
