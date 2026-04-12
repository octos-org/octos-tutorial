#!/usr/bin/env python3
"""
Octos Robot Edge Node — nav-only version for slam-nav-sim example.

Stripped from octos_inspection/nodes/octos_robot_edge_node.py:
  - Removed: arm tools (scan_station, look_around arm poses)
  - Removed: arm-related DATAFLOW_INFO entries (planner, trajectory-executor)
  - Kept: navigation tools (navigate_to, get_map, get_robot_state, stop_base)
  - Kept: ONE-SHOT and PERSISTENT modes
  - Kept: full octos agent pattern (provider, registry, skills, pipeline)

Two operating modes:
  ONE-SHOT:   Set USER_COMMAND env var — agent runs once, then stops dataflow.
  PERSISTENT: Omit USER_COMMAND — agent runs forever, accepting tasks from cloud-brain.

Env vars:
  USER_COMMAND:     Natural language command (one-shot mode)
  OPENAI_API_BASE:  LLM endpoint (default: http://10.10.5.26:4567/v1)
  OPENAI_API_KEY:   API key (default: "none" for vLLM)
  OPENAI_MODEL:     Model name (default: RoboBrain2.5-8B-NV)
  OCTOS_PROVIDER:   "openai" (default), "mock"
  OCTOS_SKILLS_DIR: Path to skills/ (default: ./skills/robot/)
  OCTOS_PIPELINE:   Path to .dot pipeline (optional)
  ROBOT_NAME:       Robot identity (default: demo_robot)
"""

import json
import os
import subprocess
import sys
import time

import pyarrow as pa
from dora import Node

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from octos import (
    Agent, AgentConfig,
    OpenAIProvider, MockProvider, RetryProvider,
    Tool, ToolRegistry, ToolResult,
    load_skills, Pipeline,
)
from octos.mission import Mission, MissionExecutor
from octos.session import SessionManager


# ---- Station layout ----
STATIONS = {
    "A":    {"position": [-0.45, 10.0], "path_y": 10.0},
    "B":    {"position": [-0.70, 18.0], "path_y": 18.0},
    "home": {"position": [-0.21, 0.28], "path_y": 0.28},
}

DATAFLOW_INFO = {
    "nodes": [
        {"id": "mujoco-sim", "description": "MuJoCo simulation (Hunter SE + warehouse)"},
        {"id": "nav-bridge", "description": "Navigation bridge: dora-nav → wheel commands"},
        {"id": "dora-nav", "description": "Path following pipeline (pub-road → planning → control)"},
        {"id": "rerun", "description": "3D visualization (Rerun)"},
    ],
}


# ===========================================================================
# Navigation tools — octos Tool ABC
# ===========================================================================

class NavigateToTool(Tool):
    def __init__(self, bridge): self._bridge = bridge
    def name(self): return "navigate_to"
    def description(self):
        return ("Navigate the Hunter SE mobile base to a named station. "
                "Available targets: 'A' (y=10m), 'B' (y=18m), 'home' (y=0.28m).")
    def input_schema(self):
        return {
            "type": "object",
            "properties": {"waypoint": {"type": "string", "description": "Station: 'A', 'B', or 'home'"}},
            "required": ["waypoint"],
        }
    def tags(self): return ["navigation", "robotics"]
    def execute(self, args):
        request = {"tool": "navigate_to", "args": {"waypoint": args.get("waypoint", "home")}}
        self._bridge.send_json("skill_request", request)
        result = self._bridge.wait_for_input("skill_result", timeout=180.0)
        return ToolResult(output=json.dumps(result or {"error": "timeout"}, default=str))


class StopBaseTool(Tool):
    def __init__(self, bridge): self._bridge = bridge
    def name(self): return "stop_base"
    def description(self): return "Emergency stop the mobile base immediately."
    def input_schema(self): return {"type": "object", "properties": {}}
    def tags(self): return ["safety", "robotics"]
    def execute(self, args):
        self._bridge.send_json("skill_request", {"tool": "stop_base", "args": {}})
        result = self._bridge.wait_for_input("skill_result", timeout=10.0)
        return ToolResult(output=json.dumps(result or {"stopped": True}, default=str))


class GetRobotStateTool(Tool):
    def __init__(self, bridge): self._bridge = bridge
    def name(self): return "get_robot_state"
    def description(self): return "Get current robot state: base position, heading, current station."
    def input_schema(self): return {"type": "object", "properties": {}}
    def tags(self): return ["robotics"]
    def execute(self, args):
        self._bridge.send_json("skill_request", {"tool": "get_robot_state", "args": {}})
        result = self._bridge.wait_for_input("skill_result", timeout=10.0)
        return ToolResult(output=json.dumps(result or {"error": "timeout"}, default=str))


class GetMapTool(Tool):
    def __init__(self, bridge): self._bridge = bridge
    def name(self): return "get_map"
    def description(self): return "Get warehouse map with station positions and robot location."
    def input_schema(self): return {"type": "object", "properties": {}}
    def tags(self): return ["navigation"]
    def execute(self, args):
        self._bridge.send_json("skill_request", {"tool": "get_map", "args": {}})
        result = self._bridge.wait_for_input("skill_result", timeout=10.0)
        return ToolResult(output=json.dumps(result or {"error": "timeout"}, default=str))


class ReadLidarTool(Tool):
    def __init__(self, bridge): self._bridge = bridge
    def name(self): return "read_lidar"
    def description(self): return "Read current LiDAR sensor status and parameters."
    def input_schema(self): return {"type": "object", "properties": {}}
    def tags(self): return ["perception"]
    def execute(self, args):
        self._bridge.send_json("skill_request", {"tool": "read_lidar", "args": {}})
        result = self._bridge.wait_for_input("skill_result", timeout=10.0)
        return ToolResult(output=json.dumps(result or {"error": "timeout"}, default=str))


class DoraListTool(Tool):
    def __init__(self, info): self._info = info
    def name(self): return "dora_list"
    def description(self): return "List available dataflow nodes."
    def input_schema(self): return {"type": "object", "properties": {}}
    def tags(self): return ["robotics"]
    def execute(self, args):
        return ToolResult(output=json.dumps(self._info))


# ===========================================================================
# Bridge — dora node communication
# ===========================================================================

class DoraAgentBridge:
    def __init__(self, node):
        self.node = node
        self.cache = {}
        self.drain_events(2.0)

    def drain_events(self, duration):
        deadline = time.time() + duration
        while time.time() < deadline:
            event = self.node.next(timeout=0.1)
            if event and event["type"] == "INPUT":
                self._cache_event(event)

    def _cache_event(self, event):
        input_id = event["id"]
        try:
            raw = bytes(event["value"].to_pylist())
            try:
                data = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = raw.hex()
            self.cache[input_id] = (data, time.time())
        except Exception as e:
            print(f"  [cache] Error caching {input_id}: {e}")

    def send_json(self, output_id, data):
        raw = json.dumps(data).encode("utf-8")
        self.node.send_output(output_id, pa.array(list(raw), type=pa.uint8()))

    def wait_for_input(self, input_id, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            event = self.node.next(timeout=1.0)
            if event is None:
                continue
            if event["type"] == "INPUT":
                self._cache_event(event)
                if event["id"] == input_id:
                    return self.cache.get(input_id, (None, 0))[0]
        return None


# ===========================================================================
# Main
# ===========================================================================

def _build_provider(provider_name: str):
    if provider_name == "mock":
        return MockProvider()
    api_base = os.environ.get("OPENAI_API_BASE", "http://10.10.5.26:4567/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "none")
    model = os.environ.get("OPENAI_MODEL", "RoboBrain2.5-8B-NV")
    base = OpenAIProvider(model=model, api_key=api_key, api_base=api_base)
    return RetryProvider(base, max_retries=3, base_delay=1.0)


def _load_skills_from_env(skills_dir: str) -> list:
    if skills_dir and os.path.isdir(skills_dir):
        return load_skills(skills_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(script_dir, "..", "skills", "robot"),
        os.path.join(os.getcwd(), "skills", "robot"),
    ]:
        if os.path.isdir(candidate):
            return load_skills(candidate)
    return []


def _deserialize_event(event):
    try:
        raw = bytes(event["value"].to_pylist())
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def main():
    provider_name = os.environ.get("OCTOS_PROVIDER", "openai")
    skills_dir = os.environ.get("OCTOS_SKILLS_DIR", "")
    pipeline_path = os.environ.get("OCTOS_PIPELINE", "")
    user_command = os.environ.get("USER_COMMAND", "")
    robot_name = os.environ.get("ROBOT_NAME", "demo_robot")
    persistent_mode = not bool(user_command)

    mode_label = "PERSISTENT" if persistent_mode else "ONE-SHOT"
    print("=" * 60)
    print(f"  Octos Robot Edge — Nav Demo ({mode_label})")
    print(f"  Robot: {robot_name}")
    if user_command:
        print(f"  Command: {user_command}")
    print(f"  Provider: {provider_name}")
    print("=" * 60)

    node = Node()
    bridge = DoraAgentBridge(node)
    provider = _build_provider(provider_name)
    session = SessionManager()

    # Build tool registry — nav-only tools
    registry = ToolRegistry()
    registry.register(NavigateToTool(bridge))
    registry.register(StopBaseTool(bridge))
    registry.register(GetRobotStateTool(bridge))
    registry.register(GetMapTool(bridge))
    registry.register(ReadLidarTool(bridge))
    registry.register(DoraListTool(DATAFLOW_INFO))
    registry.set_base_tools([
        "navigate_to", "stop_base", "get_robot_state", "get_map", "read_lidar",
    ])

    skills = _load_skills_from_env(skills_dir)

    pipeline = None
    if pipeline_path and os.path.isfile(pipeline_path):
        pipeline = Pipeline.from_dot_file(pipeline_path)

    config = AgentConfig(
        max_iterations=50,
        max_timeout_secs=600.0,
        tool_timeout_secs=600.0,
        temperature=0.1,
    )
    agent = Agent(
        provider=provider,
        registry=registry,
        config=config,
        skills=skills,
        pipeline=pipeline,
    )

    def tool_executor(tool_name, args):
        tool = registry.get(tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        return tool.execute(args).output

    mission_executor = MissionExecutor(agent, tool_executor)

    registration = {
        "robot_name": robot_name,
        "capabilities": ["navigation"],
        "status": "ready",
        "timestamp": time.time(),
    }
    bridge.send_json("agent_registration", registration)

    # --- ONE-SHOT MODE ---
    if not persistent_mode:
        print(f"\nRunning agent (one-shot)...")
        response = agent.process_message(user_command, tool_executor)
        node.send_output("result", pa.array(list(response.encode("utf-8")), type=pa.uint8()))
        print(f"\nAgent complete.")

        deadline = time.time() + 5.0
        try:
            while time.time() < deadline:
                node.next(timeout=1.0)
        except KeyboardInterrupt:
            pass
        subprocess.Popen(["dora", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    # --- PERSISTENT MODE ---
    print("\nPersistent mode — waiting for tasks...")
    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "task":
                data = _deserialize_event(event)
                if not data:
                    continue
                command = data.get("task", "") if isinstance(data, dict) else str(data)
                if not command:
                    continue

                print(f"\n>>> Task: {command}")
                bridge.send_json("agent_status", {
                    "robot_name": robot_name, "status": "busy", "timestamp": time.time()})

                mission = Mission(command=command, max_iterations=config.max_iterations,
                                  max_timeout_secs=config.max_timeout_secs)
                session.enqueue_mission(command)
                active = session.next_mission()
                if active is None:
                    continue

                response = mission_executor.execute(active, session.episode_context_prompt())
                summary = Agent.generate_episode_summary(command, response)
                session.complete_mission(summary)

                result_msg = {"robot_name": robot_name, "task": command,
                              "result": response, "timestamp": time.time()}
                node.send_output("result", pa.array(
                    list(json.dumps(result_msg).encode("utf-8")), type=pa.uint8()))

                bridge.send_json("agent_status", {
                    "robot_name": robot_name, "status": "idle", "timestamp": time.time()})
            else:
                bridge._cache_event(event)
        elif event["type"] == "STOP":
            break

    print("\nRobot edge node stopped.")


if __name__ == "__main__":
    main()
