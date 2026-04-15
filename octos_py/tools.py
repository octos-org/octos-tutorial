"""
Tool trait and ToolRegistry — mirrors octos_agent::tools.

Tool: abstract base with name, description, input_schema, tags, execute.
ToolRegistry: register tools, pin base tools (prevent LRU eviction),
              export to OpenAI function-calling schema.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Result of a tool execution."""
    output: str
    success: bool = True
    file_modified: str = ""
    tokens_used: int = 0


class Tool(ABC):
    """Abstract tool — mirrors octos_agent::tools::Tool trait."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def input_schema(self) -> dict:
        ...

    @abstractmethod
    def tags(self) -> list[str]:
        ...

    @abstractmethod
    def execute(self, args: dict) -> ToolResult:
        ...

    def required_safety_tier(self) -> str:
        """Minimum safety tier required to execute this tool.

        Override to return "safe_motion", "full_actuation", or
        "emergency_override" for tools that perform actuation.
        Default: "observe" (read-only, safe for all sessions).
        """
        return "observe"


class ToolRegistry:
    """
    Registry for tools with LRU lifecycle and base-tool pinning.

    Mirrors octos_agent::tools::ToolRegistry:
    - Register tools by name
    - Pin base tools (prevent LRU eviction)
    - Export to OpenAI function-calling schema
    - Tag-based filtering
    """

    def __init__(self, max_active: int = 15):
        self._tools: dict[str, Tool] = {}
        self._base_tools: set[str] = set()
        self._usage_order: list[str] = []
        self._max_active = max_active

    def register(self, tool: Tool) -> None:
        self._tools[tool.name()] = tool

    def set_base_tools(self, names: list[str]) -> None:
        self._base_tools = set(names)

    def get(self, name: str) -> Tool | None:
        tool = self._tools.get(name)
        if tool is not None:
            self._touch(name)
        return tool

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def _touch(self, name: str) -> None:
        if name in self._usage_order:
            self._usage_order.remove(name)
        self._usage_order.append(name)

    def active_tools(self) -> list[Tool]:
        """Return active tools, respecting LRU eviction and base pinning."""
        active = list(self._base_tools)
        for name in reversed(self._usage_order):
            if name not in active and len(active) < self._max_active:
                active.append(name)
        return [self._tools[n] for n in active if n in self._tools]

    def to_openai_schema(self) -> list[dict]:
        """Export active tools as OpenAI function-calling tool definitions."""
        result = []
        for tool in self.active_tools():
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.input_schema(),
                },
            })
        return result
