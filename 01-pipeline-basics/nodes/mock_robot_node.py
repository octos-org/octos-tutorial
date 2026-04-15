#!/usr/bin/env python3
"""Mock robot node — simulates robot tools without MuJoCo or hardware.

Responds to skill_request with simulated results. Supports:
  navigate_to(waypoint) — prints movement, returns after 2s delay
  get_robot_state()     — returns current simulated position
  get_map()             — returns station layout
  read_lidar()          — returns simulated LiDAR status
  stop_base()           — stops simulated movement
"""

import json
import os
import sys
import time

import pyarrow as pa
from dora import Node

# Simulated robot state
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
        msg = f"Robot at ({_position['x']:.2f}, {_position['y']:.2f}), station={_position['station']}"
        return [msg, {"position": _position.copy()}]

    elif tool == "get_map":
        return ["Map: stations A(y=10), B(y=18), home(y=0.28)", {"stations": STATIONS}]

    elif tool == "read_lidar":
        return ["LiDAR: 360 rays, 100m range, clear", {"rays": 360, "range_m": 100.0}]

    elif tool == "stop_base":
        return ["Robot stopped.", {"status": "stopped"}]

    elif tool == "navigate_to":
        waypoint = args.get("waypoint", "home")
        wp = waypoint.strip().upper() if len(waypoint.strip()) <= 4 else waypoint.strip()
        if wp.lower() == "home":
            wp = "home"

        if wp not in STATIONS:
            return [f"Unknown waypoint: {waypoint}", {"error": "unknown_waypoint"}]

        target = STATIONS[wp]
        print(f"[mock-robot] Navigating to {wp}...")
        # Simulate travel time proportional to distance
        dist = abs(target["y"] - _position["y"])
        travel_time = min(dist * 0.1, 2.0)  # max 2s
        time.sleep(travel_time)

        _position = {"x": target["x"], "y": target["y"], "station": wp}
        print(f"[mock-robot] Arrived at {wp} ({target['x']:.2f}, {target['y']:.2f})")
        return [
            f"Arrived at {wp} ({target['x']:.2f}, {target['y']:.2f})",
            {"position": wp, "coordinates": [target["x"], target["y"]]},
        ]

    return [f"Unknown tool: {tool}", {"error": "unknown_tool"}]


def main():
    node = Node()
    print("[mock-robot] Ready — simulated robot with 5 tools")

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] != "skill_request":
            continue

        raw = bytes(event["value"].to_pylist())
        request = json.loads(raw.decode("utf-8"))
        tool = request.get("tool", "")
        args = request.get("args", {})

        print(f"[mock-robot] {tool}({args})")
        result = _handle(tool, args)

        node.send_output("skill_result", _encode(result), {"encoding": "json"})
        print(f"[mock-robot] → {result[0][:80]}")

    print("[mock-robot] Stopped")


if __name__ == "__main__":
    main()
