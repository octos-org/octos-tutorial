#!/usr/bin/env python3
"""Mock robot — same as 01-pipeline-basics."""

import json
import time

import pyarrow as pa
from dora import Node

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
        print(f"[mock-robot] Arrived at {wp_key}")
        return [f"Arrived at {wp_key}", {"position": wp_key}]
    return [f"Unknown tool: {tool}", {"error": "unknown_tool"}]


def main():
    node = Node()
    print("[mock-robot] Ready")
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "skill_request":
            continue
        raw = bytes(event["value"].to_pylist())
        request = json.loads(raw.decode("utf-8"))
        tool, args = request.get("tool", ""), request.get("args", {})
        print(f"[mock-robot] {tool}({args})")
        result = _handle(tool, args)
        node.send_output("skill_result", _encode(result), {"encoding": "json"})

    print("[mock-robot] Stopped")


if __name__ == "__main__":
    main()
