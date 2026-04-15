"""
DOT-graph pipeline parser and executor — unique to Octos.

Parses DOT files into a topologically-sorted list of pipeline nodes.
Each node becomes a sub-agent call with its own context.

Node types:
  - codergen: The agent executes the label as instructions
  - gate: Human confirmation required before proceeding
  - dynamic_parallel: Fan-out to parallel sub-pipelines (future)
"""

import re
from dataclasses import dataclass, field


@dataclass
class PipelineNode:
    name: str
    node_type: str  # "codergen", "gate", "dynamic_parallel", "sensor_check", "motion", "grasp", "safety_gate", "wait_for_event"
    label: str  # Instructions for the agent
    tool: str = ""  # Tool name to call directly (bypasses LLM)
    args: str = ""  # JSON-encoded tool arguments
    deadline_secs: float | None = None
    deadline_action: str = "abort"  # "abort", "skip", "emergency_stop"
    invariants: list = field(default_factory=list)
    checkpoint: str = ""  # Checkpoint name for recovery
    invariant: str = ""  # Post-condition expression

    def is_gate(self) -> bool:
        return self.node_type in ("gate", "safety_gate")

    def has_direct_tool(self) -> bool:
        """True if this node has an explicit tool call (no LLM needed)."""
        return bool(self.tool)


@dataclass
class Pipeline:
    """A parsed DOT-graph pipeline with topological ordering."""
    name: str
    nodes: dict[str, PipelineNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    is_cyclic: bool = False
    max_cycles: int = 0  # 0 = infinite

    @classmethod
    def from_dot_file(cls, path: str) -> "Pipeline":
        with open(path) as f:
            text = f.read()
        return cls.from_dot_string(text, name=path)

    @classmethod
    def from_dot_string(cls, text: str, name: str = "pipeline") -> "Pipeline":
        pipeline = cls(name=name)

        # Extract graph name
        graph_match = re.search(r"digraph\s+(\w+)\s*\{", text)
        if graph_match:
            pipeline.name = graph_match.group(1)

        # Parse graph-level attributes for cycle support
        graph_attrs_match = re.search(r"graph\s*\[([^\]]*)\]", text)
        if graph_attrs_match:
            attrs_str = graph_attrs_match.group(1)
            cycle_match = re.search(r"cycle\s*=\s*(\w+)", attrs_str)
            if cycle_match:
                pipeline.is_cyclic = cycle_match.group(1).lower() == "true"
            max_cycles_match = re.search(r"max_cycles\s*=\s*(\d+)", attrs_str)
            if max_cycles_match:
                pipeline.max_cycles = int(max_cycles_match.group(1))

        # Parse node definitions: name [attr=val ...]
        # Matches both: name [type=X label="..."] and name [label="..." tool="..."]
        node_block_pattern = re.compile(r"(\w+)\s*\[([^\]]+)\]")

        def _parse_attr(attrs_str, key):
            """Extract attribute value, handling both quoted and unquoted."""
            # Quoted: key="value" or key='value'
            m = re.search(rf'{key}\s*=\s*"([^"]*)"', attrs_str)
            if m:
                return m.group(1)
            m = re.search(rf"{key}\s*=\s*'([^']*)'", attrs_str)
            if m:
                return m.group(1)
            # Unquoted: key=value (word chars)
            m = re.search(rf'{key}\s*=\s*(\w+)', attrs_str)
            if m:
                return m.group(1)
            return ""

        def _parse_json_attr(attrs_str, key):
            """Extract JSON object attribute like args='{...}'."""
            m = re.search(rf"""{key}\s*=\s*["'](\{{[^}}]*\}})["']""", attrs_str)
            if m:
                # Unescape DOT-escaped quotes: \" → "
                return m.group(1).replace('\\"', '"')
            return ""

        for match in node_block_pattern.finditer(text):
            node_name = match.group(1)
            attrs = match.group(2)

            # Skip graph-level attributes
            if node_name in ("graph", "node", "edge", "digraph"):
                continue

            label = _parse_attr(attrs, "label") or node_name
            node_type = _parse_attr(attrs, "type") or "codergen"
            tool = _parse_attr(attrs, "tool")
            args = _parse_json_attr(attrs, "args")
            deadline_str = _parse_attr(attrs, "deadline")
            deadline_action = _parse_attr(attrs, "deadline_action") or "abort"
            invariant = _parse_attr(attrs, "invariant")
            checkpoint = _parse_attr(attrs, "checkpoint")

            deadline_secs = float(deadline_str) if deadline_str else None

            pipeline.nodes[node_name] = PipelineNode(
                name=node_name,
                node_type=node_type,
                label=label,
                tool=tool,
                args=args,
                deadline_secs=deadline_secs,
                deadline_action=deadline_action,
                invariant=invariant,
                checkpoint=checkpoint,
            )

        # Parse edges: handles chains like "a -> b -> c -> d"
        # Find lines with -> and split into node chains
        for line in text.split("\n"):
            line = line.strip().rstrip(";")
            if "->" not in line:
                continue
            # Skip lines that are node definitions (contain [])
            if "[" in line and "]" in line:
                continue
            parts = [p.strip() for p in line.split("->")]
            for i in range(len(parts) - 1):
                src = parts[i].strip()
                dst = parts[i + 1].strip()
                if src in pipeline.nodes and dst in pipeline.nodes:
                    pipeline.edges.append((src, dst))

        return pipeline

    def topological_order(self) -> list[PipelineNode]:
        """Return nodes in topological order (respecting edge dependencies)."""
        in_degree: dict[str, int] = {name: 0 for name in self.nodes}
        adj: dict[str, list[str]] = {name: [] for name in self.nodes}

        for src, dst in self.edges:
            adj[src].append(dst)
            in_degree[dst] = in_degree.get(dst, 0) + 1

        queue = [n for n in self.nodes if in_degree.get(n, 0) == 0]
        result = []

        while queue:
            node_name = queue.pop(0)
            result.append(self.nodes[node_name])
            for neighbor in adj.get(node_name, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def step_count(self) -> int:
        return len(self.nodes)

    def describe(self) -> str:
        """Human-readable description of the pipeline."""
        ordered = self.topological_order()
        lines = [f"Pipeline: {self.name} ({len(ordered)} steps)"]
        for i, node in enumerate(ordered):
            prefix = "GATE" if node.is_gate() else f"Step"
            lines.append(f"  {prefix} {i + 1}: [{node.name}] {node.label}")
        return "\n".join(lines)


class PipelineExecutor:
    """
    Programmatic pipeline step-through executor.

    Steps through pipeline nodes in topological order:
    - codergen nodes: generates tool call instructions for the agent
    - gate nodes: pauses and requires confirmation before proceeding
    - Supports checkpoint recovery (resume from a given step)
    """

    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline
        self._steps = pipeline.topological_order()
        self._current_step = 0
        self._completed: list[str] = []
        self._gate_pending: PipelineNode | None = None

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    @property
    def is_complete(self) -> bool:
        return self._current_step >= len(self._steps)

    @property
    def gate_pending(self) -> PipelineNode | None:
        return self._gate_pending

    def advance(self) -> PipelineNode | None:
        """
        Advance to the next pipeline step.

        Returns the next PipelineNode to execute, or None if complete.
        If a gate node is reached, sets gate_pending and returns it
        (caller must call confirm_gate() before further advance).
        """
        if self._gate_pending:
            return self._gate_pending

        if self._current_step >= len(self._steps):
            return None

        node = self._steps[self._current_step]

        if node.is_gate():
            self._gate_pending = node
            return node

        self._current_step += 1
        self._completed.append(node.name)
        return node

    def confirm_gate(self, confirmed: bool = True) -> bool:
        """
        Confirm or reject a gate node.

        Returns True if gate was confirmed and pipeline advances,
        False if rejected (pipeline stays at gate).
        """
        if not self._gate_pending:
            return False

        if confirmed:
            self._completed.append(self._gate_pending.name)
            self._gate_pending = None
            self._current_step += 1
            return True
        return False

    def resume_from(self, step_name: str) -> bool:
        """
        Resume pipeline execution from a named step (checkpoint recovery).

        Returns True if the step was found and pipeline was repositioned.
        """
        for i, node in enumerate(self._steps):
            if node.name == step_name:
                self._current_step = i
                self._completed = [s.name for s in self._steps[:i]]
                self._gate_pending = None
                return True
        return False

    def current_instruction(self) -> str | None:
        """Get the instruction text for the current step, or None if complete."""
        if self._gate_pending:
            return f"[GATE] {self._gate_pending.label} — Awaiting confirmation."

        if self._current_step >= len(self._steps):
            return None

        node = self._steps[self._current_step]
        return f"[Step {self._current_step + 1}/{len(self._steps)}] {node.label}"

    def status(self) -> dict:
        """Return current executor status as a dict."""
        return {
            "pipeline": self._pipeline.name,
            "step": self._current_step,
            "total": len(self._steps),
            "completed": self._completed,
            "gate_pending": self._gate_pending.name if self._gate_pending else None,
            "is_complete": self.is_complete,
        }


class CyclicPipelineExecutor:
    """
    Pipeline executor that supports back-edges (cycles).

    Unlike PipelineExecutor which uses topological sort (DAG-only),
    this follows edges in declaration order and loops back to start
    when reaching the end. GATE nodes can call break_cycle() to exit.

    Used for industrial patterns like inspection patrols that loop
    indefinitely until an anomaly is detected.
    """

    def __init__(self, pipeline: Pipeline, max_cycles: int = 0):
        """
        Args:
            pipeline: The pipeline to execute
            max_cycles: Maximum number of cycles (0 = infinite)
        """
        self._pipeline = pipeline
        self._order = self._edge_order(pipeline)
        self._steps = self._order  # alias for agent compatibility
        self._max_cycles = max_cycles or pipeline.max_cycles
        self._current_idx = 0
        self._cycle_count = 0
        self._break_requested = False
        self._gate_pending: PipelineNode | None = None
        self._completed: list[str] = []

    def resume_from(self, step_name: str) -> bool:
        """Reset to the start of the cycle (cyclic pipelines always restart)."""
        self._current_idx = 0
        self._cycle_count = 0
        self._break_requested = False
        self._gate_pending = None
        self._completed = []
        return True

    @staticmethod
    def _edge_order(pipeline: Pipeline) -> list[PipelineNode]:
        """Follow edges in declaration order to build execution sequence."""
        if not pipeline.edges:
            return list(pipeline.nodes.values())

        # Build adjacency from edges (first edge per source wins)
        adj: dict[str, str] = {}
        for src, dst in pipeline.edges:
            if src not in adj:
                adj[src] = dst

        # Find start node (appears as source but not as any destination)
        dsts = {dst for _, dst in pipeline.edges}
        srcs = {src for src, _ in pipeline.edges}
        starts = srcs - dsts
        if not starts:
            # Fully cyclic — pick the first source in edge list
            start = pipeline.edges[0][0]
        else:
            start = next(iter(starts))

        # Walk the chain
        visited = set()
        order = []
        current = start
        while current and current not in visited:
            if current in pipeline.nodes:
                order.append(pipeline.nodes[current])
                visited.add(current)
            current = adj.get(current)

        return order

    @property
    def current_step(self) -> int:
        return self._current_idx

    @property
    def total_steps(self) -> int:
        return len(self._order)

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def is_complete(self) -> bool:
        if self._break_requested:
            return True
        if self._max_cycles > 0 and self._cycle_count >= self._max_cycles:
            return True
        return False

    @property
    def gate_pending(self) -> PipelineNode | None:
        return self._gate_pending

    def advance(self) -> PipelineNode | None:
        """Advance to next step, looping back to start at end of sequence."""
        if self.is_complete:
            return None

        if self._gate_pending:
            return self._gate_pending

        if self._current_idx >= len(self._order):
            # Loop back to start
            self._current_idx = 0
            self._cycle_count += 1
            print(f"  [cyclic] Starting cycle {self._cycle_count}")

            if self._max_cycles > 0 and self._cycle_count >= self._max_cycles:
                return None

        node = self._order[self._current_idx]

        if node.is_gate():
            self._gate_pending = node
            return node

        self._current_idx += 1
        self._completed.append(node.name)
        return node

    def confirm_gate(self, confirmed: bool = True) -> bool:
        """Confirm or reject a gate node."""
        if not self._gate_pending:
            return False

        if confirmed:
            self._completed.append(self._gate_pending.name)
            self._gate_pending = None
            self._current_idx += 1
            return True
        return False

    def break_cycle(self) -> None:
        """Request the executor to stop cycling after current step."""
        self._break_requested = True
        print(f"  [cyclic] Break requested after {self._cycle_count} cycles")

    def current_instruction(self) -> str | None:
        if self._gate_pending:
            return f"[GATE] {self._gate_pending.label} — Awaiting confirmation."
        if self._current_idx >= len(self._order):
            return None
        node = self._order[self._current_idx]
        cycle_info = f" (cycle {self._cycle_count})" if self._cycle_count > 0 else ""
        return f"[Step {self._current_idx + 1}/{len(self._order)}{cycle_info}] {node.label}"

    def status(self) -> dict:
        return {
            "pipeline": self._pipeline.name,
            "step": self._current_idx,
            "total": len(self._order),
            "cycle": self._cycle_count,
            "max_cycles": self._max_cycles,
            "break_requested": self._break_requested,
            "is_complete": self.is_complete,
            "completed": self._completed[-10:],  # last 10 for brevity
        }


def load_pipeline(path: str) -> Pipeline:
    """Load a pipeline from a DOT file."""
    pipeline = Pipeline.from_dot_file(path)
    print(f"  [pipeline] Loaded '{pipeline.name}' with {pipeline.step_count()} steps")
    if pipeline.is_cyclic:
        print(f"  [pipeline] Cyclic pipeline (max_cycles={pipeline.max_cycles})")
    return pipeline
