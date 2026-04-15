"""
Mission — bounded unit of work within an infinite agent session.

Replaces the one-shot model where agent lifetime = one command.
Each mission has its own iteration/timeout budget and lifecycle.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class MissionCheckpoint:
    """Checkpoint saved during mission execution for recovery."""
    node_id: str
    timestamp: float = field(default_factory=time.time)
    state: dict = field(default_factory=dict)
    resumable: bool = True

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "state": self.state,
            "resumable": self.resumable,
        }


class MissionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Mission:
    """A bounded unit of work within a persistent agent session."""
    command: str
    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: MissionStatus = MissionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: str | None = None
    max_iterations: int = 50
    max_timeout_secs: float = 600.0
    iteration_count: int = 0

    def start(self) -> None:
        self.status = MissionStatus.ACTIVE

    def complete(self, result: str) -> None:
        self.status = MissionStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result

    def fail(self, reason: str) -> None:
        self.status = MissionStatus.FAILED
        self.completed_at = time.time()
        self.result = reason

    @property
    def is_active(self) -> bool:
        return self.status == MissionStatus.ACTIVE

    @property
    def elapsed_secs(self) -> float:
        return time.time() - self.created_at

    @property
    def is_timed_out(self) -> bool:
        return self.elapsed_secs > self.max_timeout_secs

    @property
    def is_iteration_limited(self) -> bool:
        return self.iteration_count >= self.max_iterations

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "command": self.command,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "iteration_count": self.iteration_count,
            "elapsed_secs": round(self.elapsed_secs, 1),
        }


class MissionExecutor:
    """
    Executes a mission using the agent's process_message loop.

    Wraps the existing agent loop with mission lifecycle:
    - Sets iteration/timeout limits from mission config
    - Tracks iteration count on the mission
    - Returns mission result
    """

    def __init__(self, agent, tool_executor: "callable"):
        self._agent = agent
        self._tool_executor = tool_executor
        self.checkpoints: list[MissionCheckpoint] = []

    def save_checkpoint(self, node_id: str, state: dict | None = None, resumable: bool = True) -> MissionCheckpoint:
        """Save a checkpoint for the given node.

        Args:
            node_id: The pipeline node ID at which to checkpoint.
            state: Optional serializable state dict to persist.
            resumable: Whether execution can be resumed from this checkpoint.

        Returns:
            The created MissionCheckpoint.
        """
        checkpoint = MissionCheckpoint(
            node_id=node_id,
            timestamp=time.time(),
            state=state or {},
            resumable=resumable,
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def execute(self, mission: Mission, episode_context: str = "") -> str:
        """
        Execute a mission, returning the agent's final response.

        Args:
            mission: The mission to execute
            episode_context: Optional episode summaries to inject into system prompt
        """
        mission.start()
        print(f"\n{'=' * 60}")
        print(f"  Mission {mission.mission_id}: {mission.command}")
        print(f"  Limits: {mission.max_iterations} iterations, {mission.max_timeout_secs}s timeout")
        print(f"{'=' * 60}")

        try:
            response = self._agent.process_mission(
                mission=mission,
                tool_executor=self._tool_executor,
                episode_context=episode_context,
            )
            mission.complete(response)
            print(f"\n  Mission {mission.mission_id} completed.")
            return response
        except Exception as e:
            reason = f"Mission failed: {e}"
            mission.fail(reason)
            print(f"\n  Mission {mission.mission_id} failed: {e}")
            return reason
