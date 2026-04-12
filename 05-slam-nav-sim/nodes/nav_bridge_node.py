#!/usr/bin/env python3
"""
Navigation Bridge Node — nav-only dora-nav → wheel control for slam-nav-sim example.

Translates dora-nav SteeringCmd/TrqBreCmd into Hunter SE differential-drive
wheel_commands via MuJoCo velocity actuators.

Also provides navigation tools (navigate_to, stop_base, get_map, etc.) that
can be called via skill_request from the optional octos LLM agent.

Dataflow connections:
  Inputs:  skill_request (optional), ground_truth_pose, SteeringCmd, TrqBreCmd
  Outputs: skill_result, wheel_commands
"""

import json
import math
import os
import struct
import sys
import time

import numpy as np
import pyarrow as pa
from dora import Node

# Add parent dir for vendored octos package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octos.safety import SafetyTier, RobotPermissionPolicy


# ---- Safety tier configuration ----
# SAFETY_TIER env var controls max allowed tool tier for this session.
# "observe" = read-only (get_map, get_robot_state, read_lidar)
# "safe_motion" = can navigate and stop (default)
# "full_actuation" = all tools including arm control
_TIER_NAME = os.environ.get("SAFETY_TIER", "safe_motion")
_SAFETY_POLICY = RobotPermissionPolicy(
    max_tier=SafetyTier(_TIER_NAME),
)

# Tool → required safety tier mapping (from nav_tool_map.json)
TOOL_TIERS = {
    "navigate_to": SafetyTier.SAFE_MOTION,
    "stop_base": SafetyTier.SAFE_MOTION,
    "get_map": SafetyTier.OBSERVE,
    "get_robot_state": SafetyTier.OBSERVE,
    "read_lidar": SafetyTier.OBSERVE,
    "look_around": SafetyTier.OBSERVE,
}

# ---- Station configuration (warehouse path, Y-axis layout) ----
STATIONS = {
    "A": {"position": [-0.45, 10.0], "path_y": 10.0, "description": "Station A — 3 colored cubes"},
    "B": {"position": [-0.70, 18.0], "path_y": 18.0, "description": "Station B — 3 colored cubes"},
    "home": {"position": [-0.21, 0.28], "path_y": 0.28, "description": "Home (path start)"},
}

STATION_Y_THRESHOLD = 1.0  # meters

# ---- Cube positions (static, matching hunter_se_warehouse.xml) ----
STATIC_CUBES = {
    "A": [
        {"name": "A_red_block",   "color": "red",    "position": [2.60, 9.95, 0.245]},
        {"name": "A_green_block", "color": "green",  "position": [2.42, 10.00, 0.245]},
        {"name": "A_blue_block",  "color": "blue",   "position": [2.60, 9.90, 0.245]},
    ],
    "B": [
        {"name": "B_yellow_block", "color": "yellow", "position": [2.60, 17.95, 0.245]},
        {"name": "B_purple_block", "color": "purple", "position": [2.42, 18.00, 0.245]},
        {"name": "B_blue_block",   "color": "blue",   "position": [2.60, 17.90, 0.245]},
    ],
}

# ---- Hunter SE vehicle parameters ----
WHEEL_RADIUS = 0.1
MAX_STEER_RAD = 0.69
WHEELBASE = 0.55
TRQ_TO_VEL = 0.0015

# ---- Navigation parameters ----
NAV_TIMEOUT = 120.0
SETTLE_STEPS = 20

# ---- Simulated obstacle (set via env var OBSTACLE_Y) ----
# When set, navigate_to will report "obstacle detected" if the robot
# reaches this Y coordinate before arriving at the target station.
# Used to test LLM replanning for unexpected situations.
OBSTACLE_Y = float(os.environ.get("OBSTACLE_Y", "0")) if os.environ.get("OBSTACLE_Y") else None
OBSTACLE_MARGIN = 0.5  # meters — trigger zone around OBSTACLE_Y


# ---- Encoding helpers ----

def _encode_json(data) -> pa.Array:
    return pa.array(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def _decode_json(event) -> dict:
    raw = bytes(event["value"].to_pylist())
    return json.loads(raw.decode("utf-8"))


def _result(msg: str, state: dict) -> pa.Array:
    return _encode_json([msg, state])


class NavBridge:
    """Navigation bridge: dora-nav → Hunter SE wheel control."""

    def __init__(self, node: Node):
        self.node = node
        self.gt_x = 0.0
        self.gt_y = 0.0
        self.gt_theta_deg = 0.0
        self.current_station = "home"

        # dora-nav control state
        self.steering_deg = 0.0
        self.trq_enable = 0
        self.trq_value = 0.0
        self.bre_enable = 0
        self.nav_forwarding = False

    def update_pose(self, event):
        """Update from ground_truth_pose (Pose2D_h: 12 bytes = 3 floats)."""
        raw = bytes(event["value"].to_pylist())
        if len(raw) >= 12:
            x, y, theta_deg = struct.unpack('<fff', raw[:12])
            self.gt_x = x
            self.gt_y = y
            self.gt_theta_deg = theta_deg

    def update_nav_control(self, event_id, event):
        """Parse SteeringCmd or TrqBreCmd from dora-nav C++ nodes."""
        raw = bytes(event["value"].to_pylist())
        if event_id == "SteeringCmd" and len(raw) >= 4:
            self.steering_deg = struct.unpack_from('<f', raw, 0)[0]
        elif event_id == "TrqBreCmd" and len(raw) >= 17:
            self.trq_enable = raw[0]
            self.trq_value = struct.unpack_from('<f', raw, 8)[0]
            self.bre_enable = raw[12]

    # ---- Nav control → Hunter SE wheel commands ----

    def _send_wheel(self, steer_l, steer_r, speed_l, speed_r):
        steer_l = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer_l))
        steer_r = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer_r))
        # Order: [speed_l, speed_r, steer_l, steer_r]
        # Local dora-mujoco reads [0],[1] as rear wheel speeds at ctrl[9],ctrl[10]
        cmd = np.array([speed_l, speed_r, steer_l, steer_r], dtype=np.float32)
        self.node.send_output("wheel_commands", pa.array(cmd))

    def forward_nav_to_wheels(self):
        """Translate dora-nav SteeringCmd/TrqBreCmd to wheel commands."""
        if not self.nav_forwarding:
            return

        if self.trq_enable and not self.bre_enable:
            linear_vel = self.trq_value * TRQ_TO_VEL
        else:
            linear_vel = 0.0

        steer_rad = self.steering_deg * math.pi / 180.0
        angular_vel = 0.0
        if abs(linear_vel) > 0.01:
            angular_vel = linear_vel * math.tan(steer_rad) / WHEELBASE

        left_vel = linear_vel - angular_vel * WHEELBASE / 2.0
        right_vel = linear_vel + angular_vel * WHEELBASE / 2.0
        left_speed = left_vel / WHEEL_RADIUS
        right_speed = right_vel / WHEEL_RADIUS

        steer_clamped = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer_rad))
        self._send_wheel(steer_clamped, steer_clamped, left_speed, right_speed)

    def stop_wheels(self):
        self._send_wheel(0.0, 0.0, 0.0, 0.0)

    # ---- Event polling ----

    def poll_events(self, timeout=0.1, forward_nav=False):
        event = self.node.next(timeout=timeout)
        if event is not None and event["type"] == "INPUT":
            eid = event["id"]
            if eid == "ground_truth_pose":
                self.update_pose(event)
            elif eid in ("SteeringCmd", "TrqBreCmd"):
                self.update_nav_control(eid, event)
                if forward_nav:
                    self.forward_nav_to_wheels()
            return event
        return None

    # ---- Tool implementations ----

    def tool_get_map(self):
        waypoints = {name: info["position"] for name, info in STATIONS.items()}
        current_pos = [round(self.gt_x, 3), round(self.gt_y, 3)]
        msg = (f"SLAM Map: waypoints={waypoints}, "
               f"current_position={current_pos}, "
               f"current_station={self.current_station or 'home'}")
        return _result(msg, {
            "position": self.current_station or "home",
            "coordinates": current_pos,
        })

    def tool_navigate_to(self, waypoint):
        """Navigate to station by following dora-nav path and stopping at target Y."""
        wp = waypoint.strip().upper() if len(waypoint.strip()) <= 4 else waypoint.strip()
        if wp.lower() == "home":
            wp = "home"
        if wp not in STATIONS:
            return _result(f"Navigation failed: unknown waypoint '{waypoint}'", {})

        target_y = STATIONS[wp]["path_y"]
        print(f"[nav-bridge] Navigating to {wp}: target_y={target_y:.1f}, current_y={self.gt_y:.3f}")

        self.nav_forwarding = True
        deadline = time.time() + NAV_TIMEOUT
        log_interval = 2.0
        last_log = 0.0
        stall_check_y = self.gt_y
        stall_check_time = time.time()
        STALL_TIMEOUT = 10.0
        STALL_THRESHOLD = 0.1

        while time.time() < deadline:
            self.poll_events(timeout=0.02, forward_nav=True)
            dy = abs(self.gt_y - target_y)
            now = time.time()

            if now - last_log > log_interval:
                print(f"[nav-bridge] NAV: x={self.gt_x:.3f} y={self.gt_y:.3f} "
                      f"θ={self.gt_theta_deg:.1f}° dy={dy:.3f}")
                last_log = now

            # Simulated obstacle check
            if OBSTACLE_Y is not None and abs(self.gt_y - OBSTACLE_Y) < OBSTACLE_MARGIN:
                self.nav_forwarding = False
                self.stop_wheels()
                pos = [round(self.gt_x, 3), round(self.gt_y, 3)]
                print(f"[nav-bridge] OBSTACLE at y={OBSTACLE_Y:.1f}! Robot stopped at {pos}")
                return _result(
                    f"Navigation to {wp} FAILED: obstacle detected at y={OBSTACLE_Y:.1f}. "
                    f"Robot stopped at {pos}. Path to {wp} is blocked. "
                    f"Consider skipping this station or trying an alternative route.",
                    {"position": "blocked", "coordinates": pos,
                     "obstacle_y": OBSTACLE_Y, "target": wp, "error": "obstacle_detected"})

            if dy < STATION_Y_THRESHOLD:
                self.nav_forwarding = False
                self.stop_wheels()
                for _ in range(SETTLE_STEPS):
                    self.poll_events(timeout=0.02, forward_nav=False)

                self.current_station = wp
                pos = [round(self.gt_x, 3), round(self.gt_y, 3)]
                print(f"[nav-bridge] Arrived at {wp} (x={pos[0]}, y={pos[1]})")
                return _result(
                    f"Successfully navigated to waypoint {wp}. Base position: {pos}",
                    {"position": wp, "coordinates": pos})

            if abs(self.gt_y - stall_check_y) > STALL_THRESHOLD:
                stall_check_y = self.gt_y
                stall_check_time = now
            elif now - stall_check_time > STALL_TIMEOUT:
                self.nav_forwarding = False
                self.stop_wheels()
                pos = [round(self.gt_x, 3), round(self.gt_y, 3)]
                print(f"[nav-bridge] NAV STALL at y={self.gt_y:.3f}, target was {target_y}")
                return _result(
                    f"Navigation to {wp} stopped — path ended at {pos}.",
                    {"position": "path_end", "coordinates": pos})

        self.nav_forwarding = False
        self.stop_wheels()
        return _result(f"Navigation to {wp} timed out", {})

    def tool_stop_base(self):
        self.nav_forwarding = False
        self.stop_wheels()
        return _result("Base stopped.", {"status": "stopped"})

    def tool_get_robot_state(self):
        base = [round(self.gt_x, 3), round(self.gt_y, 3), round(self.gt_theta_deg, 1)]
        msg = (f"Robot state: pos=({base[0]}, {base[1]}), heading={base[2]}°, "
               f"station={self.current_station}")
        return _result(msg, {
            "base_position": base[:2],
            "heading_deg": base[2],
            "current_station": self.current_station,
        })

    def tool_look_around(self):
        station = self.current_station
        if station and station != "home":
            cubes = list(STATIC_CUBES.get(station, []))
        else:
            cubes = []
            for sc in STATIC_CUBES.values():
                cubes.extend(sc)

        colors = [c["color"] for c in cubes]
        loc = station or "current location"
        msg = (f"Found {len(cubes)} cube(s) at {loc}: "
               f"{', '.join(colors) if colors else 'none'}.")
        return _result(msg, {
            "station": station,
            "cube_count": len(cubes),
            "cube_colors": colors,
        })

    def tool_read_lidar(self):
        msg = "LiDAR active: 360 rays, 100m range, 16 vertical beams, 10Hz pointcloud"
        return _result(msg, {
            "rays": 360,
            "range_m": 100.0,
            "vertical_beams": 16,
            "rate_hz": 10,
        })

    # ---- Dispatcher ----

    def handle_skill_request(self, event):
        try:
            request = _decode_json(event)
        except Exception:
            raw = bytes(event["value"].to_pylist()).decode("utf-8")
            request = json.loads(raw)

        tool = request.get("tool", "")
        args = request.get("args", {})
        print(f"[nav-bridge] Executing: {tool}({args})")

        # Safety tier check
        required_tier = TOOL_TIERS.get(tool, SafetyTier.OBSERVE)
        allowed, err_msg = _SAFETY_POLICY.authorize(tool, required_tier)
        if not allowed:
            print(f"[nav-bridge] DENIED: {err_msg}")
            return _result(
                f"PERMISSION DENIED: {err_msg}. "
                f"Current session tier is '{_TIER_NAME}'. "
                f"Tool '{tool}' requires '{required_tier.value}' tier.",
                {"error": "permission_denied", "tool": tool,
                 "required_tier": required_tier.value,
                 "current_tier": _TIER_NAME})

        dispatch = {
            "get_map": lambda: self.tool_get_map(),
            "navigate_to": lambda: self.tool_navigate_to(args.get("waypoint", "home")),
            "stop_base": lambda: self.tool_stop_base(),
            "get_robot_state": lambda: self.tool_get_robot_state(),
            "look_around": lambda: self.tool_look_around(),
            "read_lidar": lambda: self.tool_read_lidar(),
        }

        handler = dispatch.get(tool)
        if handler:
            return handler()
        return _result(f"Unknown tool: {tool}", {"error": "unknown_tool"})


def main():
    node = Node()
    bridge = NavBridge(node)

    print("[nav-bridge] Waiting for initial pose...")

    deadline = time.time() + 30.0
    got_state = False
    while time.time() < deadline:
        event = node.next(timeout=1.0)
        if event is not None and event["type"] == "INPUT":
            if event["id"] == "ground_truth_pose":
                bridge.update_pose(event)
                got_state = True
                print(f"[nav-bridge] Initial pose: ({bridge.gt_x:.2f}, {bridge.gt_y:.2f}) "
                      f"θ={bridge.gt_theta_deg:.1f}°")
                break
            elif event["id"] in ("SteeringCmd", "TrqBreCmd"):
                bridge.update_nav_control(event["id"], event)

    if not got_state:
        print("[nav-bridge] ERROR: No pose received in 30s")
        return

    # In standalone mode (no robot-edge-a), always forward nav commands
    bridge.nav_forwarding = True
    print("[nav-bridge] Nav skill bridge ready — 6 tools, nav forwarding enabled")

    while True:
        event = node.next(timeout=0.1)
        if event is None:
            bridge.forward_nav_to_wheels()
            continue
        if event["type"] == "STOP":
            print("[nav-bridge] Shutting down")
            break
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        if eid == "ground_truth_pose":
            bridge.update_pose(event)
        elif eid in ("SteeringCmd", "TrqBreCmd"):
            bridge.update_nav_control(eid, event)
            bridge.forward_nav_to_wheels()
        elif eid == "skill_request":
            result = bridge.handle_skill_request(event)
            node.send_output("skill_result", result, {"encoding": "json"})
            print("[nav-bridge] Skill result sent")

    print("[nav-bridge] Node finished")


if __name__ == "__main__":
    main()
