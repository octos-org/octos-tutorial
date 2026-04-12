#!/usr/bin/env python3
"""Safe robot node — mock robot with safety tier enforcement.

Same as mock_robot_node but checks tool permissions against a configurable
safety tier before execution.

Env vars:
  SAFETY_TIER: Maximum allowed tier (default: "observe")
    - "observe":           get_robot_state, get_map, read_lidar
    - "safe_motion":       + navigate_to, stop_base
    - "full_actuation":    + scan_station (arm control)
    - "emergency_override": all tools
"""

import json
import os
import sys
import time

import pyarrow as pa
from dora import Node

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from octos_py.safety import SafetyTier, RobotPermissionPolicy

# Tool → required tier
TOOL_TIERS = {
    "navigate_to": SafetyTier.SAFE_MOTION,
    "stop_base": SafetyTier.SAFE_MOTION,
    "get_robot_state": SafetyTier.OBSERVE,
    "get_map": SafetyTier.OBSERVE,
    "read_lidar": SafetyTier.OBSERVE,
    "scan_station": SafetyTier.FULL_ACTUATION,
}

# Simulated state
_position = {"x": 0.0, "y": 0.0, "station": "home"}
STATIONS = {
    "A": {"x": -0.45, "y": 10.0},
    "B": {"x": -0.70, "y": 18.0},
    "home": {"x": -0.21, "y": 0.28},
}


def _encode(data):
    return pa.array(list(json.dumps(data).encode("utf-8")), type=pa.uint8())


def _handle(tool, args):
    global _position

    if tool == "get_robot_state":
        return [f"Robot at ({_position['x']:.2f}, {_position['y']:.2f}), station={_position['station']}",
                {"position": _position.copy()}]
    elif tool == "get_map":
        return ["Map: A(y=10), B(y=18), home(y=0.28)", {"stations": STATIONS}]
    elif tool == "read_lidar":
        return ["LiDAR: 360 rays, clear", {"status": "clear"}]
    elif tool == "stop_base":
        return ["Stopped.", {"status": "stopped"}]
    elif tool == "navigate_to":
        wp = args.get("waypoint", "home")
        wp_key = wp.strip().upper() if len(wp.strip()) <= 4 else wp.strip()
        if wp_key.lower() == "home":
            wp_key = "home"
        if wp_key not in STATIONS:
            return [f"Unknown: {wp}", {"error": "unknown"}]
        target = STATIONS[wp_key]
        time.sleep(0.5)
        _position = {"x": target["x"], "y": target["y"], "station": wp_key}
        return [f"Arrived at {wp_key}", {"position": wp_key}]
    elif tool == "scan_station":
        return [f"Scanned {_position['station']}: 3 objects found", {"objects": 3}]

    return [f"Unknown tool: {tool}", {"error": "unknown_tool"}]


def main():
    tier_name = os.environ.get("SAFETY_TIER", "observe")
    policy = RobotPermissionPolicy(max_tier=SafetyTier(tier_name))

    node = Node()
    print(f"[safe-robot] Ready — tier: {tier_name}")
    print(f"[safe-robot] Allowed: {[t for t, tier in TOOL_TIERS.items() if tier <= policy.max_tier]}")
    print(f"[safe-robot] Blocked: {[t for t, tier in TOOL_TIERS.items() if tier > policy.max_tier]}")

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "skill_request":
            continue

        raw = bytes(event["value"].to_pylist())
        request = json.loads(raw.decode("utf-8"))
        tool = request.get("tool", "")
        args = request.get("args", {})

        # Safety check
        required = TOOL_TIERS.get(tool, SafetyTier.OBSERVE)
        allowed, err = policy.authorize(tool, required)
        if not allowed:
            print(f"[safe-robot] DENIED: {tool} (needs {required.value}, have {tier_name})")
            result = [f"PERMISSION DENIED: {err}",
                      {"error": "permission_denied", "tool": tool,
                       "required": required.value, "current": tier_name}]
        else:
            print(f"[safe-robot] ALLOWED: {tool}({args})")
            result = _handle(tool, args)

        node.send_output("skill_result", _encode(result), {"encoding": "json"})

    print("[safe-robot] Stopped")


if __name__ == "__main__":
    main()
