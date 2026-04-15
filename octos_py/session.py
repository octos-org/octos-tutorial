"""
Session manager — state across missions for infinite-lifetime operation.

Manages episode memory (compressed summaries of past missions),
perception state (latest sensor data), and mission queue.
"""

import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .mission import Mission, MissionStatus


@dataclass
class EpisodeSummary:
    """Compressed summary of a completed mission for context injection."""
    mission_id: str
    command: str
    outcome: str  # "completed" or "failed"
    summary: str  # 2-3 sentence summary
    timestamp: float = field(default_factory=time.time)

    def to_prompt_line(self) -> str:
        age_mins = (time.time() - self.timestamp) / 60
        return (
            f"- [{self.outcome.upper()}] \"{self.command}\" "
            f"({age_mins:.0f}m ago): {self.summary}"
        )


class SessionManager:
    """
    Manages state across missions for persistent agent operation.

    Key insight: the current _compact_context tries to manage one ever-growing
    conversation. The mission model resets context per mission but preserves
    knowledge via episode summaries injected into the system prompt.
    """

    def __init__(self, max_episodes: int = 20):
        self._max_episodes = max_episodes
        self.episode_memory: list[EpisodeSummary] = []
        self.perception_state: dict[str, object] = {}
        self.active_mission: Mission | None = None
        self.mission_queue: deque[Mission] = deque()
        self.total_missions: int = 0
        self.start_time: float = time.time()

    def enqueue_mission(self, command: str, **kwargs) -> Mission:
        """Create and enqueue a new mission."""
        mission = Mission(command=command, **kwargs)
        self.mission_queue.append(mission)
        return mission

    def next_mission(self) -> Mission | None:
        """Dequeue and activate the next pending mission."""
        if not self.mission_queue:
            return None
        mission = self.mission_queue.popleft()
        self.active_mission = mission
        self.total_missions += 1
        return mission

    def complete_mission(self, summary: str) -> None:
        """Record mission completion and store episode summary."""
        if self.active_mission is None:
            return

        mission = self.active_mission
        episode = EpisodeSummary(
            mission_id=mission.mission_id,
            command=mission.command,
            outcome=mission.status.value,
            summary=summary,
        )
        self.episode_memory.append(episode)

        # Trim to max episodes (keep most recent)
        if len(self.episode_memory) > self._max_episodes:
            self.episode_memory = self.episode_memory[-self._max_episodes:]

        self.active_mission = None

    def update_perception(self, input_id: str, data: object) -> None:
        """Update perception state from a dora input event."""
        self.perception_state[input_id] = {
            "data": data,
            "timestamp": time.time(),
        }

    def episode_context_prompt(self) -> str:
        """Build episode memory context for injection into system prompt."""
        if not self.episode_memory:
            return ""

        lines = [
            "\n## Mission History",
            f"Session uptime: {(time.time() - self.start_time) / 60:.0f} minutes, "
            f"{self.total_missions} missions completed.\n",
            "Recent missions (newest last):",
        ]
        for ep in self.episode_memory:
            lines.append(ep.to_prompt_line())

        lines.append(
            "\nUse this history to avoid repeating mistakes and to build "
            "on prior knowledge about the workspace state."
        )
        return "\n".join(lines)

    def perception_summary(self) -> str:
        """Text summary of current perception state for LLM context."""
        if not self.perception_state:
            return "No perception data available."

        lines = ["Current perception state:"]
        for input_id, info in self.perception_state.items():
            age_ms = int((time.time() - info["timestamp"]) * 1000)
            data = info["data"]
            if isinstance(data, list) and len(data) > 6:
                preview = f"[{len(data)} elements]"
            elif isinstance(data, dict):
                preview = str(data)[:200]
            else:
                preview = str(data)[:200]
            lines.append(f"  {input_id}: {preview} (age: {age_ms}ms)")
        return "\n".join(lines)

    def status(self) -> dict:
        return {
            "uptime_secs": round(time.time() - self.start_time, 1),
            "total_missions": self.total_missions,
            "queued_missions": len(self.mission_queue),
            "active_mission": self.active_mission.mission_id if self.active_mission else None,
            "episode_count": len(self.episode_memory),
            "perception_inputs": list(self.perception_state.keys()),
        }


@dataclass(frozen=True)
class RobotContentBlock:
    """Structured content block for robot-specific data in recordings."""

    block_type: str  # "sensor", "motion", "force", "image"
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": self.block_type,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class BlackBoxRecorder:
    """
    JSONL flight-data recorder for post-incident analysis.

    Uses a background thread and bounded queue. If the queue is full,
    new entries are silently dropped (never blocks the agent loop).
    """

    def __init__(self, path: str, buffer_size: int = 1024) -> None:
        self._path = path
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._active = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self) -> None:
        with open(self._path, "a") as f:
            while self._active or not self._queue.empty():
                try:
                    entry = self._queue.get(timeout=0.1)
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                except queue.Empty:
                    continue

    def record(self, event: str, data: dict | None = None) -> bool:
        """Record an event. Returns True if queued, False if dropped."""
        entry = {
            "timestamp": time.time(),
            "event": event,
            "data": data or {},
        }
        try:
            self._queue.put_nowait(entry)
            return True
        except queue.Full:
            return False

    def record_block(self, event: str, block: "RobotContentBlock") -> bool:
        """Record an event with a RobotContentBlock."""
        entry = {
            "timestamp": time.time(),
            "event": event,
            "data": block.to_dict(),
        }
        try:
            self._queue.put_nowait(entry)
            return True
        except queue.Full:
            return False

    @property
    def is_active(self) -> bool:
        return self._active and self._thread.is_alive()

    def close(self) -> None:
        """Signal the writer thread to stop and wait for it to finish."""
        self._active = False
        self._thread.join(timeout=2.0)
