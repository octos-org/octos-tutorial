#!/usr/bin/env python3
"""Octos agent node — executes a DOT pipeline with direct tool calls.

Loads a pipeline from OCTOS_PIPELINE env var (or patrol.dot by default),
steps through each node, and calls tools on the mock-robot via
skill_request/skill_result.

No LLM required — tools are called directly from DOT node attributes.

Env vars:
  OCTOS_PIPELINE: Path to .dot pipeline file (default: patrol.dot)
  USER_COMMAND:   Command description for logging
"""

import json
import os
import sys
import time

import pyarrow as pa
from dora import Node

# Add octos_py to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from octos_py import (
    Agent, AgentConfig,
    MockProvider,
    Tool, ToolRegistry, ToolResult,
    Pipeline,
)


class SkillBridgeTool(Tool):
    """Generic tool that forwards calls to mock-robot via dora."""

    def __init__(self, tool_name, description, bridge, schema=None):
        self._name = tool_name
        self._description = description
        self._bridge = bridge
        self._schema = schema or {"type": "object", "properties": {}}

    def name(self):
        return self._name

    def description(self):
        return self._description

    def input_schema(self):
        return self._schema

    def tags(self):
        return ["robotics"]

    def execute(self, args):
        request = {"tool": self._name, "args": args}
        self._bridge.send_json("skill_request", request)
        result = self._bridge.wait_for("skill_result", timeout=30.0)
        return ToolResult(output=json.dumps(result or {"error": "timeout"}, default=str))


class DoraBridge:
    """Minimal bridge for sending/receiving dora messages."""

    def __init__(self, node):
        self.node = node

    def send_json(self, output_id, data):
        raw = json.dumps(data).encode("utf-8")
        self.node.send_output(output_id, pa.array(list(raw), type=pa.uint8()))

    def wait_for(self, input_id, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            event = self.node.next(timeout=1.0)
            if event is None:
                continue
            if event["type"] == "INPUT" and event["id"] == input_id:
                raw = bytes(event["value"].to_pylist())
                return json.loads(raw.decode("utf-8"))
        return None


def main():
    pipeline_path = os.environ.get("OCTOS_PIPELINE", "patrol.dot")
    user_command = os.environ.get("USER_COMMAND", "Execute patrol pipeline")

    print("=" * 50)
    print("  Octos Agent — Pipeline Basics")
    print(f"  Pipeline: {pipeline_path}")
    print(f"  Command: {user_command}")
    print("=" * 50)

    node = Node()
    bridge = DoraBridge(node)

    # Build tool registry
    registry = ToolRegistry()
    registry.register(SkillBridgeTool("navigate_to", "Navigate to station",
        bridge, {"type": "object", "properties": {"waypoint": {"type": "string"}}, "required": ["waypoint"]}))
    registry.register(SkillBridgeTool("get_robot_state", "Get robot position", bridge))
    registry.register(SkillBridgeTool("get_map", "Get station map", bridge))
    registry.register(SkillBridgeTool("read_lidar", "Read LiDAR", bridge))
    registry.register(SkillBridgeTool("stop_base", "Stop robot", bridge))

    # Load pipeline
    pipeline = Pipeline.from_dot_file(pipeline_path)
    print(f"\nPipeline: {pipeline.name} ({len(pipeline.nodes)} steps)")

    # Build agent with mock provider (no LLM needed)
    config = AgentConfig(max_iterations=50, max_timeout_secs=300.0, temperature=0.1)
    agent = Agent(
        provider=MockProvider(),
        registry=registry,
        config=config,
        pipeline=pipeline,
    )

    def tool_executor(tool_name, args):
        tool = registry.get(tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        return tool.execute(args).output

    # Run agent
    print(f"\nExecuting: {user_command}\n")
    response = agent.process_message(user_command, tool_executor)
    print(f"\n{'=' * 50}")
    print(f"  Complete!")
    print(f"{'=' * 50}")

    # Clean exit
    time.sleep(1)
    import subprocess
    subprocess.Popen(["dora", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
