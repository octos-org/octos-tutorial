#!/usr/bin/env python3
"""Mock robot with simulated obstacle for LLM replanning demo.

Same as 01's mock robot but OBSTACLE_Y env var blocks navigation
at a configurable Y position, forcing the LLM to adapt.

Env vars:
  OBSTACLE_Y: Y position where navigation is blocked (default: none)
"""

import json
import os
import time

import pyarrow as pa
from dora import Node

_position = {"x": 0.0, "y": 0.0, "station": "home"}

STATIONS = {
    "A": {"x": -0.45, "y": 10.0},
    "B": {"x": -0.70, "y": 18.0},
    "home": {"x": -0.21, "y": 0.28},
}

OBSTACLE_Y = float(os.environ.get("OBSTACLE_Y", "0")) if os.environ.get("OBSTACLE_Y") else None


def _encode(data):
    return pa.array(list(json.dumps(data).encode("utf-8")), type=pa.uint8())


def _handle(tool, args):
    global _position

    if tool == "get_robot_state":
        return [f"Robot at ({_position['x']:.2f}, {_position['y']:.2f}), station={_position['station']}",
                {"position": _position.copy()}]

    elif tool == "get_map":
        info = "Map: stations A(y=10), B(y=18), home(y=0.28)"
        if OBSTACLE_Y:
            info += f". WARNING: obstacle reported near y={OBSTACLE_Y}"
        return [info, {"stations": STATIONS, "obstacle_y": OBSTACLE_Y}]

    elif tool == "read_lidar":
        if OBSTACLE_Y and abs(_position["y"] - OBSTACLE_Y) < 3.0:
            return [f"LiDAR: OBSTACLE DETECTED ahead at y={OBSTACLE_Y}",
                    {"status": "obstacle", "obstacle_y": OBSTACLE_Y}]
        return ["LiDAR: 360 rays, path clear", {"status": "clear"}]

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

        # Check obstacle
        if OBSTACLE_Y is not None:
            curr_y = _position["y"]
            target_y = target["y"]
            # Obstacle blocks if it's between current position and target
            if (curr_y < OBSTACLE_Y < target_y) or (target_y < OBSTACLE_Y < curr_y):
                print(f"[mock-robot] OBSTACLE at y={OBSTACLE_Y} blocks path to {wp}")
                _position["y"] = OBSTACLE_Y - 0.5 if curr_y < OBSTACLE_Y else OBSTACLE_Y + 0.5
                return [
                    f"Navigation to {wp} FAILED: obstacle detected at y={OBSTACLE_Y}. "
                    f"Path blocked. Robot stopped at y={_position['y']:.1f}. "
                    f"Consider skipping this station or trying a different route.",
                    {"error": "obstacle_detected", "obstacle_y": OBSTACLE_Y,
                     "target": wp, "stopped_at": _position.copy()}]

        # Simulate travel
        time.sleep(0.5)
        _position = {"x": target["x"], "y": target["y"], "station": wp}
        print(f"[mock-robot] Arrived at {wp}")
        return [f"Arrived at {wp} ({target['x']:.2f}, {target['y']:.2f})",
                {"position": wp, "coordinates": [target["x"], target["y"]]}]

    return [f"Unknown tool: {tool}", {"error": "unknown_tool"}]


def main():
    node = Node()
    print(f"[mock-robot] Ready — obstacle={'y=' + str(OBSTACLE_Y) if OBSTACLE_Y else 'none'}")

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "skill_request":
            continue

        raw = bytes(event["value"].to_pylist())
        request = json.loads(raw.decode("utf-8"))
        tool = request.get("tool", "")
        args = request.get("args", {})

        print(f"[mock-robot] {tool}({args})")
        result = _handle(tool, args)
        node.send_output("skill_result", _encode(result), {"encoding": "json"})
        print(f"[mock-robot] → {result[0][:100]}")

    print("[mock-robot] Stopped")


if __name__ == "__main__":
    main()
