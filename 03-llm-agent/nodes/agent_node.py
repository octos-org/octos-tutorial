#!/usr/bin/env python3
"""Octos LLM agent — free-form reasoning with OpenAI-compatible endpoint.

Unlike 01-pipeline-basics (deterministic DOT pipeline), this agent uses
an LLM to decide which tools to call and how to handle failures.

Env vars:
  OPENAI_API_BASE: LLM endpoint (default: http://localhost:11434/v1)
  OPENAI_API_KEY:  API key (default: "none" for Ollama)
  OPENAI_MODEL:    Model name (default: qwen3:30b)
  USER_COMMAND:    Natural language command for the agent
"""

import json
import os
import sys
import time

import pyarrow as pa
from dora import Node

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from octos_py import (
    Agent, AgentConfig,
    OpenAIProvider, RetryProvider,
    Tool, ToolRegistry, ToolResult,
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
        result = self._bridge.recv("skill_result", timeout=30.0)
        return ToolResult(output=json.dumps(result or {"error": "timeout"}, default=str))


class Bridge:
    def __init__(self, node):
        self.node = node

    def send(self, output_id, data):
        raw = json.dumps(data).encode("utf-8")
        self.node.send_output(output_id, pa.array(list(raw), type=pa.uint8()))

    def recv(self, input_id, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            event = self.node.next(timeout=1.0)
            if event and event["type"] == "INPUT" and event["id"] == input_id:
                return json.loads(bytes(event["value"].to_pylist()).decode("utf-8"))
        return None


def main():
    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "none")
    model = os.environ.get("OPENAI_MODEL", "qwen3:30b")
    user_command = os.environ.get("USER_COMMAND", "Check robot state and navigate to station A")

    print("=" * 50)
    print("  Octos Agent — LLM Reasoning")
    print(f"  Model: {model}")
    print(f"  Endpoint: {api_base}")
    print(f"  Command: {user_command}")
    print("=" * 50)

    node = Node()
    bridge = Bridge(node)

    # LLM provider with retry
    base_provider = OpenAIProvider(model=model, api_key=api_key, api_base=api_base)
    provider = RetryProvider(base_provider, max_retries=3, base_delay=1.0)

    # Tools
    registry = ToolRegistry()
    registry.register(SkillTool("navigate_to", "Navigate to a station (A, B, or home)", bridge,
        {"type": "object", "properties": {"waypoint": {"type": "string", "description": "Station: A, B, or home"}},
         "required": ["waypoint"]}))
    registry.register(SkillTool("get_robot_state", "Get current robot position and station", bridge))
    registry.register(SkillTool("get_map", "Get warehouse map with station locations", bridge))
    registry.register(SkillTool("read_lidar", "Read LiDAR sensor — detects obstacles", bridge))
    registry.register(SkillTool("stop_base", "Emergency stop the robot", bridge))

    config = AgentConfig(max_iterations=15, max_timeout_secs=120.0, temperature=0.1)
    agent = Agent(provider=provider, registry=registry, config=config)

    def tool_executor(tool_name, args):
        tool = registry.get(tool_name)
        if not tool:
            return json.dumps({"error": f"Unknown: {tool_name}"})
        return tool.execute(args).output

    print(f"\nRunning agent...\n")
    response = agent.process_message(user_command, tool_executor)

    print(f"\n{'=' * 50}")
    print(f"  Agent response:")
    print(f"  {response[:500]}")
    print(f"{'=' * 50}")

    time.sleep(1)
    import subprocess
    subprocess.Popen(["dora", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
