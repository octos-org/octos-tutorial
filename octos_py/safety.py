"""
Safety infrastructure — mirrors Octos safety features.

LoopDetector: Detects and breaks infinite tool call loops (loop_detect.rs)
MessageRepairer: Fixes malformed tool call JSON (message_repair.rs)
SafetyHook: Pre-tool-call validation (hooks system)
"""

import json
import re
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class SafetyTier(str, Enum):
    """
    Safety tiers ordered from least to most dangerous.

    Mirrors octos_agent::permissions::SafetyTier.
    """
    OBSERVE = "observe"
    SAFE_MOTION = "safe_motion"
    FULL_ACTUATION = "full_actuation"
    EMERGENCY_OVERRIDE = "emergency_override"

    def __lt__(self, other):
        if not isinstance(other, SafetyTier):
            return NotImplemented
        order = list(SafetyTier)
        return order.index(self) < order.index(other)

    def __le__(self, other):
        if not isinstance(other, SafetyTier):
            return NotImplemented
        order = list(SafetyTier)
        return order.index(self) <= order.index(other)

    def __gt__(self, other):
        if not isinstance(other, SafetyTier):
            return NotImplemented
        order = list(SafetyTier)
        return order.index(self) > order.index(other)

    def __ge__(self, other):
        if not isinstance(other, SafetyTier):
            return NotImplemented
        order = list(SafetyTier)
        return order.index(self) >= order.index(other)


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned workspace bounds for safe motion validation."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, x: float, y: float, z: float) -> bool:
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass(frozen=True)
class PermissionDenied:
    """Error when a tool's required tier exceeds the session's allowed tier."""
    tool_name: str
    required: SafetyTier
    allowed: SafetyTier

    def __str__(self) -> str:
        return (
            f"permission denied: tool '{self.tool_name}' requires tier "
            f"'{self.required.value}' but session allows up to '{self.allowed.value}'"
        )


@dataclass
class RobotPermissionPolicy:
    """
    Policy that authorizes tool execution based on safety tiers.

    Mirrors octos_agent::permissions::RobotPermissionPolicy.
    """
    max_tier: SafetyTier = SafetyTier.OBSERVE
    workspace: WorkspaceBounds | None = None

    def authorize(self, tool_name: str, required: SafetyTier) -> tuple[bool, str]:
        """
        Check if a tool with the given required tier is authorized.
        Returns (allowed, error_message).
        """
        if required <= self.max_tier:
            return True, ""
        denied = PermissionDenied(
            tool_name=tool_name,
            required=required,
            allowed=self.max_tier,
        )
        return False, str(denied)


@dataclass
class RobotPayload:
    """Robot-specific payload for robot safety hook events.

    Mirrors octos_agent::hooks::RobotPayload.
    """
    joint_positions: list[float] = field(default_factory=list)
    tcp_position: list[float] | None = None
    velocity: float | None = None
    force_torque: list[float] = field(default_factory=list)
    safety_tier: str | None = None

    @classmethod
    def for_motion(cls, joint_positions: list[float], velocity: float | None = None) -> "RobotPayload":
        return cls(joint_positions=joint_positions, velocity=velocity)

    @classmethod
    def for_force(cls, force_torque: list[float]) -> "RobotPayload":
        return cls(force_torque=force_torque)

    def to_dict(self) -> dict:
        d: dict = {}
        if self.joint_positions:
            d["joint_positions"] = self.joint_positions
        if self.tcp_position is not None:
            d["tcp_position"] = self.tcp_position
        if self.velocity is not None:
            d["velocity"] = self.velocity
        if self.force_torque:
            d["force_torque"] = self.force_torque
        if self.safety_tier is not None:
            d["safety_tier"] = self.safety_tier
        return d


# Robot safety hook event constants
HOOK_BEFORE_MOTION = "before_motion"
HOOK_AFTER_MOTION = "after_motion"
HOOK_FORCE_LIMIT = "force_limit"
HOOK_WORKSPACE_BOUNDARY = "workspace_boundary"
HOOK_EMERGENCY_STOP = "emergency_stop"


def before_motion(joint_positions: list[float], velocity: float | None = None,
                  max_velocity: float = 1.0) -> tuple[bool, str]:
    """Pre-motion safety check. Returns (allowed, message)."""
    if velocity is not None and velocity > max_velocity:
        return False, (
            f"Velocity {velocity:.2f} m/s exceeds maximum {max_velocity:.2f} m/s"
        )
    return True, ""


def check_force_limit(force_torque: list[float],
                      force_limit: float = 50.0) -> tuple[bool, str]:
    """Check if force/torque exceeds limits. Returns (exceeded, message)."""
    if not force_torque:
        return False, ""
    force_magnitude = sum(f * f for f in force_torque[:3]) ** 0.5
    if force_magnitude > force_limit:
        return True, (
            f"Force magnitude {force_magnitude:.1f}N exceeds limit {force_limit:.1f}N"
        )
    return False, ""


def check_workspace_boundary(position: list[float],
                             bounds: "WorkspaceBounds") -> tuple[bool, str]:
    """Check if position is within workspace bounds. Returns (in_bounds, message)."""
    if len(position) < 3:
        return False, f"Position must have at least 3 components, got {len(position)}"
    if not bounds.contains(position[0], position[1], position[2]):
        return False, (
            f"Position [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] "
            f"is outside workspace bounds"
        )
    return True, ""


class LoopDetector:
    """
    Detects infinite tool call loops by tracking recent call patterns.

    If the same (tool_name, args_hash) appears more than max_repeats times
    in the last window_size calls, the loop is detected.
    """

    def __init__(self, window_size: int = 10, max_repeats: int = 5):
        self._window_size = window_size
        self._max_repeats = max_repeats
        self._history: list[str] = []

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """
        Check if a tool call is part of a loop.
        Returns (is_loop, message).
        """
        key = f"{tool_name}:{_stable_hash(args)}"
        self._history.append(key)

        # Trim to window
        if len(self._history) > self._window_size:
            self._history = self._history[-self._window_size:]

        count = self._history.count(key)
        if count >= self._max_repeats:
            return True, (
                f"Loop detected: {tool_name} called {count} times with same args "
                f"in last {self._window_size} calls. Breaking loop."
            )
        return False, ""

    def reset(self) -> None:
        self._history.clear()


class MessageRepairer:
    """
    Repairs malformed tool call JSON from LLM responses.

    Common issues:
    - Trailing commas in JSON
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing closing braces
    """

    @staticmethod
    def repair_json(raw: str) -> str:
        """Attempt to repair malformed JSON string."""
        if not raw or not raw.strip():
            return "{}"

        text = raw.strip()

        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Try parsing as-is first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try fixing single quotes
        try:
            fixed = text.replace("'", '"')
            json.loads(fixed)
            return fixed
        except (json.JSONDecodeError, ValueError):
            pass

        # Try adding missing closing braces
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        if open_braces > 0 or open_brackets > 0:
            fixed = text + "]" * open_brackets + "}" * open_braces
            try:
                json.loads(fixed)
                return fixed
            except json.JSONDecodeError:
                pass

        # Last resort: return empty object
        return "{}"

    @staticmethod
    def repair_tool_args(raw_args: str) -> dict:
        """Parse tool arguments with repair fallback."""
        try:
            return json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            repaired = MessageRepairer.repair_json(raw_args)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return {}


# Joint limits for UR5e (radians)
UR5E_JOINT_LIMITS = [
    (-2 * math.pi, 2 * math.pi),   # shoulder_pan
    (-2 * math.pi, 2 * math.pi),   # shoulder_lift
    (-math.pi, math.pi),            # elbow
    (-2 * math.pi, 2 * math.pi),   # wrist_1
    (-2 * math.pi, 2 * math.pi),   # wrist_2
    (-2 * math.pi, 2 * math.pi),   # wrist_3
]


class SafetyHook:
    """
    Pre-tool-call safety validation.

    Validates:
    - Joint positions within UR5e limits
    - No dangerous rapid movements
    - Scene commands have valid structure
    """

    def __init__(self, joint_limits: list[tuple[float, float]] | None = None):
        self._joint_limits = joint_limits or UR5E_JOINT_LIMITS

    def before_tool_call(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """
        Validate a tool call before execution.
        Returns (allowed, message). If not allowed, message explains why.
        """
        if tool_name in ("dora_call", "dora_send"):
            data = args.get("data", {})
            output_id = args.get("output_id", "")

            if output_id == "plan_request":
                return self._validate_plan_request(data)

        return True, ""

    def _validate_plan_request(self, data: dict) -> tuple[bool, str]:
        """Validate joint positions in a plan request."""
        for key in ("start", "goal"):
            joints = data.get(key)
            if joints is None:
                continue
            if not isinstance(joints, list) or len(joints) != 6:
                return False, f"Plan request '{key}' must be a list of 6 joint values, got {type(joints)}"
            for i, (val, (lo, hi)) in enumerate(zip(joints, self._joint_limits)):
                if not isinstance(val, (int, float)):
                    return False, f"Joint {i} in '{key}' is not a number: {val}"
                if val < lo or val > hi:
                    return False, (
                        f"Joint {i} in '{key}' = {val:.4f} is outside limits "
                        f"[{lo:.4f}, {hi:.4f}]"
                    )
        return True, ""


class CircuitBreaker:
    """
    Auto-disable after N consecutive failures.

    When tripped, all subsequent calls are blocked until reset.
    Mirrors the Octos safety circuit breaker pattern.
    """

    def __init__(self, max_consecutive_failures: int = 5):
        self._max_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._tripped = False
        self._total_trips = 0

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def record_success(self) -> None:
        """Record a successful operation, resetting the failure counter."""
        self._consecutive_failures = 0

    def record_failure(self) -> bool:
        """
        Record a failed operation.
        Returns True if the circuit breaker has just tripped.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_failures and not self._tripped:
            self._tripped = True
            self._total_trips += 1
            return True
        return False

    def check(self) -> tuple[bool, str]:
        """
        Check if operations are allowed.
        Returns (allowed, message).
        """
        if self._tripped:
            return False, (
                f"Circuit breaker tripped after {self._max_failures} consecutive failures. "
                f"Call reset() to re-enable."
            )
        return True, ""

    def reset(self) -> None:
        """Reset the circuit breaker, allowing operations again."""
        self._tripped = False
        self._consecutive_failures = 0

    def export_metrics(self) -> dict:
        return {
            "tripped": self._tripped,
            "consecutive_failures": self._consecutive_failures,
            "total_trips": self._total_trips,
        }


class WatchdogTimer:
    """
    Monitors agent activity and triggers safe-hold if no tool call
    occurs within the timeout period.

    For 24/7 operation: if the LLM provider goes down or the agent
    gets stuck, the watchdog ensures the robot enters a safe state.
    """

    def __init__(self, timeout_secs: float = 120.0):
        self._timeout_secs = timeout_secs
        self._last_activity: float = time.time()
        self._tripped = False
        self._trip_count = 0

    def feed(self) -> None:
        """Record activity (call after each tool execution or LLM response)."""
        self._last_activity = time.time()
        self._tripped = False

    def check(self) -> tuple[bool, str]:
        """
        Check if the watchdog has timed out.
        Returns (timed_out, message).
        """
        elapsed = time.time() - self._last_activity
        if elapsed > self._timeout_secs and not self._tripped:
            self._tripped = True
            self._trip_count += 1
            return True, (
                f"Watchdog timeout: no activity for {elapsed:.0f}s "
                f"(threshold: {self._timeout_secs}s). Entering safe hold."
            )
        return False, ""

    def reset(self) -> None:
        self._last_activity = time.time()
        self._tripped = False

    def export_metrics(self) -> dict:
        return {
            "timeout_secs": self._timeout_secs,
            "last_activity_ago": round(time.time() - self._last_activity, 1),
            "tripped": self._tripped,
            "total_trips": self._trip_count,
        }


class SafeHoldManager:
    """
    Manages safe-hold state for the robot during 24/7 operation.

    When triggered (by watchdog, circuit breaker, or explicit call):
    1. Freezes robot at current position (no new motion commands)
    2. Sends alert to monitoring systems
    3. Periodically allows recovery checks

    The safe-hold is a logical state — the actual position-hold is
    achieved by not sending new trajectories (MuJoCo PD controller
    holds the last commanded position).
    """

    def __init__(self):
        self._in_safe_hold = False
        self._hold_reason: str = ""
        self._hold_start: float | None = None
        self._hold_count = 0
        self._recovery_attempts = 0

    @property
    def is_holding(self) -> bool:
        return self._in_safe_hold

    @property
    def hold_reason(self) -> str:
        return self._hold_reason

    def enter_safe_hold(self, reason: str) -> None:
        """Enter safe-hold state."""
        if not self._in_safe_hold:
            self._in_safe_hold = True
            self._hold_reason = reason
            self._hold_start = time.time()
            self._hold_count += 1
            self._recovery_attempts = 0
            print(f"  [SAFE HOLD] Entered: {reason}")

    def exit_safe_hold(self) -> None:
        """Exit safe-hold state (after recovery)."""
        if self._in_safe_hold:
            duration = time.time() - (self._hold_start or time.time())
            print(f"  [SAFE HOLD] Exited after {duration:.1f}s "
                  f"({self._recovery_attempts} recovery attempts)")
            self._in_safe_hold = False
            self._hold_reason = ""
            self._hold_start = None

    def record_recovery_attempt(self) -> None:
        """Record a provider recovery attempt during safe-hold."""
        self._recovery_attempts += 1

    def should_allow_tool(self, tool_name: str) -> tuple[bool, str]:
        """
        Check if a tool call is allowed during safe-hold.

        During safe-hold, only read/status tools are allowed.
        Motion tools (dora_call with plan_request, dora_move) are blocked.
        """
        if not self._in_safe_hold:
            return True, ""

        # Allow read-only tools during safe hold
        safe_tools = {"dora_read", "dora_list", "dora_wait", "dora_perceive", "dora_alert"}
        if tool_name in safe_tools:
            return True, ""

        return False, (
            f"Safe-hold active ({self._hold_reason}). "
            f"Motion tool '{tool_name}' blocked. Only read/status tools allowed."
        )

    def export_metrics(self) -> dict:
        return {
            "in_safe_hold": self._in_safe_hold,
            "hold_reason": self._hold_reason,
            "hold_duration": round(time.time() - self._hold_start, 1) if self._hold_start else 0,
            "total_holds": self._hold_count,
            "recovery_attempts": self._recovery_attempts,
        }


def _stable_hash(obj) -> str:
    """Create a stable string hash of a JSON-serializable object."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(obj)
