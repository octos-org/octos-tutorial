#!/usr/bin/env python3
"""Octos agent with human gate support.

Executes a DOT pipeline but pauses at gate nodes and checks GATE_POLICY
to decide whether to approve or reject. Rejected gates skip the next
tool execution step.

Env vars:
  OCTOS_PIPELINE: Path to .dot pipeline file
  GATE_POLICY:    Comma-separated list of approved gate labels (substring match)
                  Default: "all" (approve everything)
                  Example: "Station A,home" (approve A and home, reject B)
  USER_COMMAND:   Command description
"""

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
from octos_py.pipeline import PipelineExecutor


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

    def recv(self, input_id, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            event = self.node.next(timeout=1.0)
            if event and event["type"] == "INPUT" and event["id"] == input_id:
                return json.loads(bytes(event["value"].to_pylist()).decode("utf-8"))
        return None


def check_gate(gate_label, policy_str):
    """Check if a gate should be approved based on policy.

    Returns (approved: bool, reason: str).
    """
    if policy_str.lower() == "all":
        return True, "policy: approve all"

    approved_keywords = [k.strip().lower() for k in policy_str.split(",")]
    label_lower = gate_label.lower()

    for keyword in approved_keywords:
        if keyword in label_lower:
            return True, f"policy: '{keyword}' matches gate"

    return False, f"policy: gate '{gate_label}' not in approved list [{policy_str}]"


def main():
    pipeline_path = os.environ.get("OCTOS_PIPELINE", "inspection_gated.dot")
    gate_policy = os.environ.get("GATE_POLICY", "all")
    user_command = os.environ.get("USER_COMMAND", "Execute gated inspection patrol")

    print("=" * 55)
    print("  Octos Agent — Human Gate Demo")
    print(f"  Pipeline: {pipeline_path}")
    print(f"  Gate policy: {gate_policy}")
    print("=" * 55)

    node = Node()
    bridge = Bridge(node)

    registry = ToolRegistry()
    registry.register(SkillTool("navigate_to", "Navigate to station", bridge,
        {"type": "object", "properties": {"waypoint": {"type": "string"}}, "required": ["waypoint"]}))
    registry.register(SkillTool("get_robot_state", "Get robot position", bridge))
    registry.register(SkillTool("get_map", "Get station map", bridge))
    registry.register(SkillTool("read_lidar", "Read LiDAR", bridge))
    registry.register(SkillTool("stop_base", "Stop robot", bridge))

    pipeline = Pipeline.from_dot_file(pipeline_path)
    executor = PipelineExecutor(pipeline)

    print(f"\nPipeline: {pipeline.name} ({len(pipeline.nodes)} nodes)")
    gate_count = sum(1 for n in pipeline.nodes.values() if n.is_gate())
    print(f"Gates: {gate_count}")

    def tool_executor(tool_name, args):
        tool = registry.get(tool_name)
        if not tool:
            return json.dumps({"error": f"Unknown: {tool_name}"})
        return tool.execute(args).output

    # Manual pipeline execution with gate handling
    approved_gates = []
    rejected_gates = []
    executed_tools = []
    skip_next_tool = False

    print(f"\nExecuting: {user_command}\n")

    while not executor.is_complete:
        step_node = executor.advance()
        if step_node is None:
            break

        step_num = executor.current_step
        total = executor.total_steps

        if step_node.is_gate():
            # Gate node — check policy
            approved, reason = check_gate(step_node.label, gate_policy)

            if approved:
                print(f"  [gate {step_num}/{total}] APPROVED: {step_node.label}")
                print(f"    Reason: {reason}")
                executor.confirm_gate(True)
                approved_gates.append(step_node.label)
                skip_next_tool = False
            else:
                print(f"  [gate {step_num}/{total}] REJECTED: {step_node.label}")
                print(f"    Reason: {reason}")
                executor.confirm_gate(True)  # advance past gate
                rejected_gates.append(step_node.label)
                skip_next_tool = True  # skip the next tool step

        elif step_node.has_direct_tool():
            if skip_next_tool:
                print(f"  [step {step_num}/{total}] SKIPPED: {step_node.label} (gate rejected)")
                skip_next_tool = False
                continue

            tool_name = step_node.tool
            try:
                tool_args = json.loads(step_node.args) if step_node.args else {}
            except json.JSONDecodeError:
                tool_args = {}

            print(f"  [step {step_num}/{total}] {tool_name}({tool_args})")
            result = tool_executor(tool_name, tool_args)
            executed_tools.append(tool_name)

            if step_node.checkpoint:
                print(f"    Checkpoint: {step_node.checkpoint}")
        else:
            print(f"  [step {step_num}/{total}] {step_node.label} (no tool)")
            skip_next_tool = False

    # Summary
    print(f"\n{'=' * 55}")
    print("  Mission Summary")
    print(f"{'=' * 55}")
    print(f"  Approved gates: {approved_gates}")
    print(f"  Rejected gates: {rejected_gates}")
    print(f"  Tools executed: {executed_tools}")
    print(f"  Steps skipped:  {len(rejected_gates)}")
    print(f"{'=' * 55}")

    time.sleep(1)
    import subprocess
    subprocess.Popen(["dora", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
