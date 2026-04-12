"""
Agent — mirrors octos_agent::agent::Agent.

Orchestrates: provider (LLM), tool registry, skills, pipeline,
loop detection, message repair, safety hooks, context compaction.
"""

import json
import re
import time
from dataclasses import dataclass, field

import threading
from collections import deque

from .provider import LlmProvider, ChatConfig, ChatResponse
from .tools import ToolRegistry, ToolResult
from .skills import SkillInfo, skills_to_prompt
from .pipeline import Pipeline, PipelineExecutor, CyclicPipelineExecutor
from .safety import LoopDetector, MessageRepairer, SafetyHook, WatchdogTimer, SafeHoldManager
from .mission import Mission, MissionStatus


@dataclass(frozen=True)
class RealtimeConfig:
    """Configuration for real-time agent loop behavior.

    Mirrors octos_agent::agent::realtime::RealtimeConfig.
    """
    iteration_deadline_ms: int = 5000
    heartbeat_timeout_ms: int = 10000
    llm_timeout_ms: int = 8000
    min_cycle_ms: int = 100
    check_estop: bool = True


class Heartbeat:
    """Atomic heartbeat counter for monitoring agent liveness.

    Mirrors octos_agent::agent::realtime::Heartbeat.
    """

    def __init__(self, timeout_ms: int = 10000):
        self._counter = 0
        self._last_check_value = 0
        self._timeout_secs = timeout_ms / 1000.0
        self._last_beat: float = time.time()
        self._lock = threading.Lock()

    def beat(self) -> None:
        """Record a heartbeat (called each agent loop iteration)."""
        with self._lock:
            self._counter += 1
            self._last_beat = time.time()

    @property
    def count(self) -> int:
        with self._lock:
            return self._counter

    def state(self) -> str:
        """Check heartbeat state. Returns 'alive' or 'stalled'."""
        with self._lock:
            current = self._counter
            prev = self._last_check_value
            self._last_check_value = current

            if current != prev:
                return "alive"

            if time.time() - self._last_beat > self._timeout_secs:
                return "stalled"
            return "alive"


@dataclass
class SensorSnapshot:
    """A timestamped snapshot of sensor data for LLM context injection."""
    sensor_id: str
    value: object
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_context_line(self) -> str:
        age_ms = int(time.time() * 1000) - self.timestamp_ms
        return f"[{self.sensor_id}] {self.value} ({age_ms}ms ago)"


class SensorContextInjector:
    """Ring buffer for sensor snapshots with LLM context formatting.

    Mirrors octos_agent::agent::realtime::SensorContextInjector.
    """

    def __init__(self, capacity: int = 20):
        self._buffer: deque[SensorSnapshot] = deque(maxlen=capacity)
        self._capacity = capacity

    def push(self, snapshot: SensorSnapshot) -> None:
        self._buffer.append(snapshot)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def to_context_block(self) -> str:
        if not self._buffer:
            return ""
        lines = ["## Live Sensor Data"]
        for snap in self._buffer:
            lines.append(snap.to_context_line())
        return "\n".join(lines)

    def latest(self, sensor_id: str) -> SensorSnapshot | None:
        for snap in reversed(self._buffer):
            if snap.sensor_id == sensor_id:
                return snap
        return None


@dataclass
class AgentConfig:
    """Agent configuration — mirrors octos_agent::agent::AgentConfig."""
    max_iterations: int = 50
    max_timeout_secs: float = 600.0
    tool_timeout_secs: float = 600.0
    temperature: float = 0.1
    context_compact_threshold: int = 80_000  # tokens before compaction
    # Persistent mode fields
    episode_memory_size: int = 20
    idle_pipeline_path: str = ""
    watchdog_timeout_secs: float = 120.0


BASE_SYSTEM_PROMPT = """You are a robot control agent operating a UR5e 6-DOF robot arm with a Robotiq 2F-85 gripper in MuJoCo simulation.

## Available Tools
You have 5 tools to interact with the robot:

1. **dora_read** — Read the latest sensor value (joint_positions, scene_state, execution_status, gripper_status)
2. **dora_send** — Send a fire-and-forget command (scene_command)
3. **dora_call** — Send a command and wait for response (plan_request->plan_status, gripper_command->gripper_status)
4. **dora_move** — Move arm directly to joint target (straight-line, no planner). Use for short motions like lowering to grasp/place.
5. **dora_wait** — Wait for a number of seconds (for hold/pause commands)
6. **dora_list** — List available dataflow nodes and their inputs/outputs

## Important
- Always read joint_positions before planning (for start config)
- Wait for each plan+execution to complete before the next move
- For relative rotations: add the angle offset to the appropriate joint of current position
- Use dora_wait for hold/pause commands
"""


class Agent:
    """
    Octos-pattern agent with tool calling loop.

    Features:
    - ToolRegistry with LRU and base-tool pinning
    - Skill loading from SKILL.md files
    - Pipeline execution (DOT-graph)
    - Loop detection
    - Message repair
    - Safety hooks
    - Context compaction
    """

    def __init__(
        self,
        provider: LlmProvider,
        registry: ToolRegistry,
        config: AgentConfig | None = None,
        skills: list[SkillInfo] | None = None,
        pipeline: Pipeline | None = None,
    ):
        self._provider = provider
        self._registry = registry
        self._config = config or AgentConfig()
        self._skills = skills or []
        self._pipeline = pipeline
        self._loop_detector = LoopDetector()
        self._repairer = MessageRepairer()
        self._safety = SafetyHook()
        if pipeline and pipeline.is_cyclic:
            self._pipeline_executor = CyclicPipelineExecutor(pipeline)
        elif pipeline:
            self._pipeline_executor = PipelineExecutor(pipeline)
        else:
            self._pipeline_executor = None
        self._system_prompt = self._build_system_prompt()

    @staticmethod
    def _try_direct_execute(label: str, tool_executor) -> str | None:
        """Parse a pipeline label for an explicit dora_call/dora_send and execute directly.

        Returns the tool result string if successful, None if label can't be parsed.
        """
        # Match dora_read
        read_m = re.search(r"Use dora_read.*?(\w+_\w+)", label)
        if read_m:
            input_id = read_m.group(1)
            try:
                print(f"  [direct] dora_read(input_id={input_id})")
                result = tool_executor("dora_read", {"input_id": input_id})
                return result
            except Exception as e:
                print(f"  [direct] dora_read failed: {e}")
                return None

        # Match dora_move: goal=[...]
        move_m = re.search(r"Use dora_move:\s*goal\s*=\s*(\[[\d\s,.\-]+\])", label)
        if move_m:
            try:
                goal = json.loads(move_m.group(1))
                print(f"  [direct] dora_move(goal={goal})")
                result = tool_executor("dora_move", {"goal": goal})
                return result
            except Exception as e:
                print(f"  [direct] dora_move failed: {e}")
                return None

        # Match: "Use dora_call: output_id=X, data={...}, response_id=Y. Description."
        # Use greedy match + sentence boundary (". " + uppercase) to avoid
        # stopping at decimal points inside numbers like -1.5708
        m = re.search(
            r"Use (dora_call|dora_send):\s*(.+?)(?:\.\s+[A-Z]|\.$|$)", label
        )
        if not m:
            return None

        tool_name = m.group(1)
        params_str = m.group(2).strip().rstrip(".")

        args = {}
        # Parse output_id
        oid = re.search(r"output_id\s*=\s*(\w+)", params_str)
        if oid:
            args["output_id"] = oid.group(1)

        # Parse response_id
        rid = re.search(r"response_id\s*=\s*(\w+)", params_str)
        if rid:
            args["response_id"] = rid.group(1)

        # Parse data={...} — extract the JSON-like object
        data_match = re.search(r"data\s*=\s*(\{.+\})", params_str)
        if data_match:
            raw = data_match.group(1)
            # Convert from DOT label syntax to valid JSON:
            # Add quotes around keys, handle arrays
            try:
                # Try direct JSON parse first
                args["data"] = json.loads(raw)
            except json.JSONDecodeError:
                # Add quotes around bare keys: key: -> "key":
                fixed = re.sub(r"(\w+)\s*:", r'"\1":', raw)
                # Quote bare string values: "key": word -> "key": "word"
                fixed = re.sub(
                    r':\s*([a-zA-Z_]\w*)\s*([,}])',
                    r': "\1"\2',
                    fixed,
                )
                try:
                    args["data"] = json.loads(fixed)
                except json.JSONDecodeError:
                    return None

        if not args.get("output_id"):
            return None

        try:
            print(f"  [direct] {tool_name}({json.dumps(args, separators=(',', ':'))})")
            result = tool_executor(tool_name, args)
            return result
        except Exception as e:
            print(f"  [direct] Failed: {e}")
            return None

    def _build_system_prompt(self) -> str:
        parts = [BASE_SYSTEM_PROMPT]

        # Add skills
        if self._skills:
            skill_text = skills_to_prompt(self._skills, always_only=True)
            if skill_text:
                parts.append(skill_text)

        # Add pipeline info
        if self._pipeline:
            parts.append(f"\n## Active Pipeline\n\n{self._pipeline.describe()}")
            parts.append(
                "\nFollow the pipeline steps in order. Execute each step's instructions "
                "using the available tools. For GATE steps, report status and wait for "
                "confirmation before proceeding."
            )

        return "\n\n".join(parts)

    def process_message(
        self,
        user_message: str,
        tool_executor: "callable",
    ) -> str:
        """
        Run the agent loop for a user message.

        Args:
            user_message: Natural language command
            tool_executor: Callable(tool_name, args) -> str that executes tools

        Returns:
            Final agent text response
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]

        print(f"\n>>> User: {user_message}\n")
        print(f"  Provider: {self._provider.name()}")
        if self._pipeline:
            print(f"  Pipeline: {self._pipeline.name} ({self._pipeline.step_count()} steps)")
        print(f"  Skills: {[s.name for s in self._skills]}")
        print(f"  Tools: {self._registry.names()}")

        self._loop_detector.reset()
        if self._pipeline_executor:
            self._pipeline_executor.resume_from(
                self._pipeline_executor._steps[0].name if self._pipeline_executor._steps else ""
            )
        chat_config = ChatConfig(temperature=self._config.temperature)
        deadline = time.time() + self._config.max_timeout_secs
        final_response = "(no final response)"

        for iteration in range(1, self._config.max_iterations + 1):
            if time.time() > deadline:
                print(f"  [timeout] Agent exceeded {self._config.max_timeout_secs}s")
                break

            # Pipeline step execution: directly execute steps with explicit tool calls
            if self._pipeline_executor and not self._pipeline_executor.is_complete:
                step_node = self._pipeline_executor.advance()
                if step_node:
                    instruction = self._pipeline_executor.current_instruction()
                    if instruction:
                        if step_node.is_gate():
                            print(f"  [pipeline] Gate: {step_node.label}")
                            messages.append({
                                "role": "user",
                                "content": f"Pipeline gate reached: {step_node.label}. Confirming and proceeding.",
                            })
                            self._pipeline_executor.confirm_gate(True)
                        else:
                            step_num = self._pipeline_executor.current_step
                            total = self._pipeline_executor.total_steps
                            print(f"  [pipeline] Step {step_num}/{total}: {step_node.label}")

                            # Direct tool execution from DOT node attributes
                            if step_node.has_direct_tool():
                                tool_name = step_node.tool
                                try:
                                    tool_args = json.loads(step_node.args) if step_node.args else {}
                                except json.JSONDecodeError:
                                    tool_args = {}
                                print(f"  [pipeline] Direct: {tool_name}({tool_args})")
                                try:
                                    result = tool_executor(tool_name, tool_args)
                                    messages.append({
                                        "role": "user",
                                        "content": f"[Step {step_num}] {step_node.label}",
                                    })
                                    messages.append({
                                        "role": "assistant",
                                        "content": f"Executed {tool_name}: {result[:500] if result else 'ok'}",
                                    })
                                    if step_node.checkpoint:
                                        print(f"  [pipeline] Checkpoint: {step_node.checkpoint}")
                                    continue  # skip LLM, go to next pipeline step
                                except Exception as e:
                                    print(f"  [pipeline] Tool {tool_name} failed: {e}")
                                    if step_node.deadline_action == "skip":
                                        continue
                                    break  # abort on failure

                            # Try to parse and directly execute tool call from label
                            direct_result = self._try_direct_execute(
                                step_node.label, tool_executor
                            )
                            if direct_result:
                                # Tool executed directly — add to messages and skip LLM
                                messages.append({
                                    "role": "user",
                                    "content": f"[Step {step_num}] {step_node.label}",
                                })
                                messages.append({
                                    "role": "assistant",
                                    "content": f"Executed: {direct_result}",
                                })
                                continue  # skip LLM call, go to next pipeline step

                            # Fallback: inject as instruction for LLM
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"IMPORTANT — Execute this EXACT tool call now "
                                    f"(Step {step_num}/{total}):\n{instruction}"
                                ),
                            })

            print(f"--- LLM iteration {iteration} ---")

            tools_schema = self._registry.to_openai_schema()

            try:
                response = self._provider.chat(messages, tools_schema, chat_config)
            except Exception as e:
                print(f"  [ERROR] Provider call failed: {e}")
                if hasattr(e, "response"):
                    try:
                        print(f"  [ERROR] Response body: {e.response.text}")
                    except Exception:
                        pass
                break

            # Build assistant message for history
            msg_dict = {"role": "assistant"}
            if response.content:
                msg_dict["content"] = response.content
            if response.tool_calls:
                msg_dict["tool_calls"] = response.tool_calls
            messages.append(msg_dict)

            # Check if done — but don't stop if pipeline has remaining steps
            if response.finish_reason == "stop" or not response.tool_calls:
                if response.content:
                    print(f"\n>>> Agent: {response.content}\n")
                    final_response = response.content

                has_pipeline_steps = (
                    self._pipeline_executor
                    and not self._pipeline_executor.is_complete
                )
                if has_pipeline_steps:
                    print(f"  [pipeline] LLM tried to stop but pipeline has "
                          f"{self._pipeline_executor.total_steps - self._pipeline_executor.current_step} "
                          f"steps remaining — continuing...")
                    continue
                break

            # Execute tool calls
            for tc in response.tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]

                # Message repair
                fn_args = self._repairer.repair_tool_args(raw_args)

                # Loop detection
                is_loop, loop_msg = self._loop_detector.check(fn_name, fn_args)
                if is_loop:
                    print(f"  [LOOP] {loop_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": loop_msg}),
                    })
                    continue

                # Safety hook
                allowed, safety_msg = self._safety.before_tool_call(fn_name, fn_args)
                if not allowed:
                    print(f"  [SAFETY] Blocked: {safety_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": f"Safety check failed: {safety_msg}"}),
                    })
                    continue

                # Execute via bridge
                result = tool_executor(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            # Context compaction check
            messages = self._compact_context(messages)

        # Print metrics
        metrics = self._provider.export_metrics()
        if metrics:
            print(f"  [metrics] {metrics}")

        return final_response

    def process_mission(
        self,
        mission: Mission,
        tool_executor: "callable",
        episode_context: str = "",
    ) -> str:
        """
        Run the agent loop for a mission with per-mission lifecycle.

        Unlike process_message (which runs a single conversation),
        process_mission:
        - Injects episode summaries from prior missions into system prompt
        - Resets loop detector per mission
        - Tracks iteration count on the mission object
        - Generates a 2-3 sentence summary for episode memory

        Args:
            mission: The Mission to execute
            tool_executor: Callable(tool_name, args) -> str
            episode_context: Episode summaries from SessionManager

        Returns:
            Final agent text response
        """
        # Build mission-specific system prompt with episode context
        system_prompt = self._system_prompt
        if episode_context:
            system_prompt = system_prompt + "\n\n" + episode_context

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mission.command},
        ]

        print(f"\n>>> Mission [{mission.mission_id}]: {mission.command}\n")
        print(f"  Provider: {self._provider.name()}")
        if self._pipeline:
            print(f"  Pipeline: {self._pipeline.name} ({self._pipeline.step_count()} steps)")
        print(f"  Skills: {[s.name for s in self._skills]}")
        print(f"  Tools: {self._registry.names()}")
        if episode_context:
            print(f"  Episode context: {len(episode_context)} chars injected")

        self._loop_detector.reset()
        if self._pipeline_executor:
            self._pipeline_executor.resume_from(
                self._pipeline_executor._steps[0].name if self._pipeline_executor._steps else ""
            )
        chat_config = ChatConfig(temperature=self._config.temperature)
        deadline = time.time() + mission.max_timeout_secs
        final_response = "(no final response)"

        for iteration in range(1, mission.max_iterations + 1):
            mission.iteration_count = iteration

            if time.time() > deadline:
                print(f"  [timeout] Mission exceeded {mission.max_timeout_secs}s")
                break

            # Pipeline step execution (same logic as process_message)
            if self._pipeline_executor and not self._pipeline_executor.is_complete:
                step_node = self._pipeline_executor.advance()
                if step_node:
                    instruction = self._pipeline_executor.current_instruction()
                    if instruction:
                        if step_node.is_gate():
                            print(f"  [pipeline] Gate: {step_node.label}")
                            messages.append({
                                "role": "user",
                                "content": f"Pipeline gate reached: {step_node.label}. Confirming and proceeding.",
                            })
                            self._pipeline_executor.confirm_gate(True)
                        else:
                            step_num = self._pipeline_executor.current_step
                            total = self._pipeline_executor.total_steps
                            print(f"  [pipeline] Step {step_num}/{total}: {step_node.label}")

                            direct_result = self._try_direct_execute(
                                step_node.label, tool_executor
                            )
                            if direct_result:
                                messages.append({
                                    "role": "user",
                                    "content": f"[Step {step_num}] {step_node.label}",
                                })
                                messages.append({
                                    "role": "assistant",
                                    "content": f"Executed: {direct_result}",
                                })
                                continue

                            messages.append({
                                "role": "user",
                                "content": (
                                    f"IMPORTANT — Execute this EXACT tool call now "
                                    f"(Step {step_num}/{total}):\n{instruction}"
                                ),
                            })

            print(f"--- LLM iteration {iteration} (mission {mission.mission_id}) ---")

            tools_schema = self._registry.to_openai_schema()

            try:
                response = self._provider.chat(messages, tools_schema, chat_config)
            except Exception as e:
                print(f"  [ERROR] Provider call failed: {e}")
                break

            msg_dict = {"role": "assistant"}
            if response.content:
                msg_dict["content"] = response.content
            if response.tool_calls:
                msg_dict["tool_calls"] = response.tool_calls
            messages.append(msg_dict)

            if response.finish_reason == "stop" or not response.tool_calls:
                if response.content:
                    print(f"\n>>> Agent: {response.content}\n")
                    final_response = response.content

                has_pipeline_steps = (
                    self._pipeline_executor
                    and not self._pipeline_executor.is_complete
                )
                if has_pipeline_steps:
                    continue
                break

            for tc in response.tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                fn_args = self._repairer.repair_tool_args(raw_args)

                is_loop, loop_msg = self._loop_detector.check(fn_name, fn_args)
                if is_loop:
                    print(f"  [LOOP] {loop_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": loop_msg}),
                    })
                    continue

                allowed, safety_msg = self._safety.before_tool_call(fn_name, fn_args)
                if not allowed:
                    print(f"  [SAFETY] Blocked: {safety_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": f"Safety check failed: {safety_msg}"}),
                    })
                    continue

                result = tool_executor(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            messages = self._compact_context(messages)

        metrics = self._provider.export_metrics()
        if metrics:
            print(f"  [metrics] {metrics}")

        return final_response

    @staticmethod
    def generate_episode_summary(command: str, response: str) -> str:
        """Generate a 2-3 sentence summary of a mission for episode memory."""
        # Truncate long responses for summary
        resp_preview = response[:300] if len(response) > 300 else response
        # Simple extractive summary — the LLM response usually contains
        # a natural summary of what it did
        if len(response) > 200:
            # Take first two sentences
            sentences = response.split(". ")
            summary = ". ".join(sentences[:2])
            if not summary.endswith("."):
                summary += "."
            return summary
        return resp_preview

    def _compact_context(self, messages: list[dict]) -> list[dict]:
        """
        Token-aware context compaction.

        Estimates token count and removes old tool call/result pairs
        when approaching the context window limit.
        """
        estimated_tokens = sum(
            len(json.dumps(m, default=str)) // 4
            for m in messages
        )

        if estimated_tokens < self._config.context_compact_threshold:
            return messages

        print(f"  [compact] ~{estimated_tokens} tokens, compacting old tool calls...")

        # Keep system prompt, last N messages, remove middle tool exchanges
        system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        # Keep first user message + last 20 messages, summarize middle
        if len(rest) > 22:
            first_user = rest[0]
            recent = rest[-20:]
            removed_count = len(rest) - 21

            summary = {
                "role": "user",
                "content": f"[Context compacted: {removed_count} earlier messages removed. "
                           f"The agent has been executing tool calls for the current task.]",
            }
            rest = [first_user, summary] + recent

        return system + rest
