#!/usr/bin/env python3
"""Agent node for safety tier demo — tries all tools to discover permissions."""

import json
import os
import sys
import time

import pyarrow as pa
from dora import Node

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from octos_py import (
    Agent, AgentConfig, MockProvider,
    Tool, ToolRegistry, ToolResult, Pipeline,
)


class SkillTool(Tool):
    def __init__(self, tool_name, desc, bridge, schema=None):
        self._name = tool_name
        self._desc = desc
        self._bridge = bridge
        self._schema = schema or {"type": "object", "properties": {}}

    def name(self): return self._name
    def description(self): return self._desc
    def input_schema(self): return self._schema
    def tags(self): return ["robotics"]

    def execute(self, args):
        self._bridge.send("skill_request", {"tool": self._name, "args": args})
        result = self._bridge.recv("skill_result", timeout=10.0)
        return ToolResult(output=json.dumps(result or {"error": "timeout"}, default=str))


class Bridge:
    def __init__(self, node):
        self.node = node

    def send(self, output_id, data):
        raw = json.dumps(data).encode("utf-8")
        self.node.send_output(output_id, pa.array(list(raw), type=pa.uint8()))

    def recv(self, input_id, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            event = self.node.next(timeout=1.0)
            if event and event["type"] == "INPUT" and event["id"] == input_id:
                return json.loads(bytes(event["value"].to_pylist()).decode("utf-8"))
        return None


def main():
    pipeline_path = os.environ.get("OCTOS_PIPELINE", "test_all_tools.dot")
    user_command = os.environ.get("USER_COMMAND", "Test all tools to discover permissions")

    print("=" * 50)
    print("  Octos Agent — Safety Tiers Demo")
    print(f"  Pipeline: {pipeline_path}")
    print("=" * 50)

    node = Node()
    bridge = Bridge(node)

    registry = ToolRegistry()
    registry.register(SkillTool("get_robot_state", "Get position", bridge))
    registry.register(SkillTool("get_map", "Get map", bridge))
    registry.register(SkillTool("read_lidar", "Read LiDAR", bridge))
    registry.register(SkillTool("navigate_to", "Navigate to station", bridge,
        {"type": "object", "properties": {"waypoint": {"type": "string"}}, "required": ["waypoint"]}))
    registry.register(SkillTool("stop_base", "Stop robot", bridge))
    registry.register(SkillTool("scan_station", "Scan for objects", bridge))

    pipeline = Pipeline.from_dot_file(pipeline_path)
    config = AgentConfig(max_iterations=30, max_timeout_secs=120.0, temperature=0.1)
    agent = Agent(provider=MockProvider(), registry=registry, config=config, pipeline=pipeline)

    def tool_executor(tool_name, args):
        tool = registry.get(tool_name)
        if not tool:
            return json.dumps({"error": f"Unknown: {tool_name}"})
        return tool.execute(args).output

    response = agent.process_message(user_command, tool_executor)
    print(f"\n{'=' * 50}")
    print("  Complete!")
    print(f"{'=' * 50}")

    time.sleep(1)
    import subprocess
    subprocess.Popen(["dora", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
