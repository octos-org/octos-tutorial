#!/usr/bin/env python3
"""recovery-monitor — Phase 3 recovery behavior state machine for dora-nav.

Inputs:  ground_truth_pose (Pose2D_h, 12 bytes), Request (Request_h, 14 bytes)
Output:  recovery_cmd (5 bytes): uint8 state + float32 backup_speed

States: NORMAL(0) → WAITING(1) → BACKING_UP(2) → STOPPED(3) → NORMAL
"""

import struct
import time

import pyarrow as pa
from dora import Node

# Request types
REQ_FORWARD = 0
REQ_STOP = 1
REQ_BACK = 2
REQ_AEB = 3

# Recovery states
STATE_NORMAL = 0
STATE_WAITING = 1
STATE_BACKING_UP = 2
STATE_STOPPED = 3

STATE_NAMES = {
    STATE_NORMAL: "NORMAL",
    STATE_WAITING: "WAITING",
    STATE_BACKING_UP: "BACKING_UP",
    STATE_STOPPED: "STOPPED",
}

# Tuning constants
STUCK_WINDOW_S = 3.0       # seconds to look back for displacement check
STUCK_THRESHOLD_M = 0.1    # meters — below this is "stuck"
WAITING_TIMEOUT_S = 5.0    # seconds in WAITING before transitioning to BACKING_UP
BACKUP_DISTANCE_M = 1.0    # meters of reverse travel to complete recovery
MAX_RETRIES = 3            # retries before STOPPED
RETRY_RESET_DISTANCE_M = 2.0  # meters of forward travel to reset retry count
COOLDOWN_S = 30.0          # seconds in STOPPED before auto-reset
BACKUP_SPEED = -200.0      # longitudinal speed command for reverse
RECOVERY_STRUCT = "<Bf"    # uint8 state + float32 backup_speed = 5 bytes


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _pack_cmd(state: int, speed: float) -> bytes:
    return struct.pack(RECOVERY_STRUCT, state, speed)


def _transition(old: int, new: int) -> None:
    print(f"[recovery] {STATE_NAMES[old]} → {STATE_NAMES[new]}", flush=True)


def main() -> None:
    node = Node()
    print("[recovery] Recovery monitor ready", flush=True)

    # --- mutable state (all plain scalars / lists, no mutation of dicts) ---
    state = STATE_NORMAL
    request_type = REQ_FORWARD

    # Pose history: list of (timestamp, x, y)
    pose_history: list[tuple[float, float, float]] = []

    # WAITING
    waiting_start: float = 0.0

    # BACKING_UP
    backup_start_x: float = 0.0
    backup_start_y: float = 0.0
    retry_count: int = 0

    # STOPPED
    stopped_at: float = 0.0

    # NORMAL — track distance for retry reset
    normal_travel_start_x: float = 0.0
    normal_travel_start_y: float = 0.0

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue

        eid = event["id"]

        if eid == "Request":
            raw = bytes(event["value"].to_pylist())
            if len(raw) < 1:
                continue
            request_type = raw[0]
            continue

        if eid != "ground_truth_pose":
            continue

        raw = bytes(event["value"].to_pylist())
        if len(raw) < 12:
            continue

        x, y, _theta = struct.unpack_from("<fff", raw, 0)
        now = time.time()

        # Append current pose and prune history older than STUCK_WINDOW_S
        pose_history.append((now, x, y))
        pose_history = [(t, px, py) for (t, px, py) in pose_history
                        if now - t <= STUCK_WINDOW_S + 1.0]

        # ---- State machine ------------------------------------------------

        if state == STATE_NORMAL:
            # Reset retry count after meaningful forward travel
            travel = _dist(normal_travel_start_x, normal_travel_start_y, x, y)
            if travel >= RETRY_RESET_DISTANCE_M and retry_count > 0:
                retry_count = 0
                normal_travel_start_x = x
                normal_travel_start_y = y
                print("[recovery] Retry count reset (>2m forward travel)", flush=True)

            # Check for stuck condition
            if request_type != REQ_STOP:
                old_entries = [(t, px, py) for (t, px, py) in pose_history
                               if now - t >= STUCK_WINDOW_S]
                if old_entries:
                    _, ox, oy = old_entries[0]
                    if _dist(ox, oy, x, y) < STUCK_THRESHOLD_M:
                        _transition(STATE_NORMAL, STATE_WAITING)
                        state = STATE_WAITING
                        waiting_start = now

        elif state == STATE_WAITING:
            elapsed = now - waiting_start
            if elapsed >= WAITING_TIMEOUT_S:
                if retry_count >= MAX_RETRIES:
                    _transition(STATE_WAITING, STATE_STOPPED)
                    state = STATE_STOPPED
                    stopped_at = now
                else:
                    _transition(STATE_WAITING, STATE_BACKING_UP)
                    state = STATE_BACKING_UP
                    backup_start_x = x
                    backup_start_y = y
                    retry_count += 1
                    print(f"[recovery] Backup attempt {retry_count}/{MAX_RETRIES}",
                          flush=True)

        elif state == STATE_BACKING_UP:
            backed = _dist(backup_start_x, backup_start_y, x, y)
            if backed >= BACKUP_DISTANCE_M:
                _transition(STATE_BACKING_UP, STATE_NORMAL)
                state = STATE_NORMAL
                normal_travel_start_x = x
                normal_travel_start_y = y

        elif state == STATE_STOPPED:
            # Manual forward command resets immediately
            if request_type == REQ_FORWARD:
                _transition(STATE_STOPPED, STATE_NORMAL)
                state = STATE_NORMAL
                retry_count = 0
                normal_travel_start_x = x
                normal_travel_start_y = y
            elif now - stopped_at >= COOLDOWN_S:
                print("[recovery] 30s cooldown expired, auto-reset", flush=True)
                _transition(STATE_STOPPED, STATE_NORMAL)
                state = STATE_NORMAL
                retry_count = 0
                normal_travel_start_x = x
                normal_travel_start_y = y

        # ---- Emit recovery_cmd on every pose update -----------------------
        speed = BACKUP_SPEED if state == STATE_BACKING_UP else 0.0
        cmd = _pack_cmd(state, speed)
        node.send_output(
            "recovery_cmd",
            pa.array(list(cmd), type=pa.uint8()),
        )

    print("[recovery] Stopped", flush=True)


if __name__ == "__main__":
    main()
