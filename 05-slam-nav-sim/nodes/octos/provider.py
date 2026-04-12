"""
LLM Provider abstraction — mirrors octos_llm::provider.

3-layer failover: RetryProvider -> ProviderChain -> AdaptiveRouter.
Supports: OpenAI, Mock (deterministic for CI).
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ChatConfig:
    temperature: float = 0.1
    max_tokens: int | None = None


@dataclass
class ChatResponse:
    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"


class LlmProvider(ABC):
    """Abstract LLM provider — mirrors octos_llm::provider::LlmProvider trait."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        config: ChatConfig,
    ) -> ChatResponse:
        ...

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        config: ChatConfig,
    ) -> Iterator[ChatResponse]:
        """Streaming chat — yields partial responses. Default: single yield."""
        yield self.chat(messages, tools, config)

    def context_window(self) -> int:
        return 128_000

    def export_metrics(self) -> dict:
        return {}

    def report_late_failure(self, error: str) -> None:
        """Report a failure discovered after the call returned (e.g. invalid JSON)."""
        pass


# ---------------------------------------------------------------------------
# Layer 1: RetryProvider — wraps a single provider with exponential backoff
# ---------------------------------------------------------------------------

class RetryProvider(LlmProvider):
    """
    Retry wrapper with exponential backoff.
    Mirrors octos RetryProvider — first layer of 3-layer failover.
    """

    def __init__(self, inner: LlmProvider, max_retries: int = 3, base_delay: float = 1.0):
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._retry_count = 0

    def name(self) -> str:
        return f"retry({self._inner.name()})"

    def chat(self, messages, tools, config):
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                result = self._inner.chat(messages, tools, config)
                return result
            except Exception as e:
                last_error = e
                self._retry_count += 1
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    print(f"  [retry] Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        raise last_error

    def chat_stream(self, messages, tools, config):
        yield self.chat(messages, tools, config)

    def context_window(self) -> int:
        return self._inner.context_window()

    def export_metrics(self):
        inner_metrics = self._inner.export_metrics()
        return {**inner_metrics, "retries": self._retry_count}

    def report_late_failure(self, error):
        self._inner.report_late_failure(error)


# ---------------------------------------------------------------------------
# Layer 2: ProviderChain — falls through a list of providers on failure
# ---------------------------------------------------------------------------

class ProviderChain(LlmProvider):
    """
    Sequential failover chain.
    Mirrors octos ProviderChain — second layer of 3-layer failover.
    Tries each provider in order until one succeeds.
    """

    def __init__(self, providers: list[LlmProvider]):
        if not providers:
            raise ValueError("ProviderChain requires at least one provider")
        self._providers = providers
        self._active_index = 0
        self._failover_count = 0

    def name(self) -> str:
        names = [p.name() for p in self._providers]
        return f"chain({' -> '.join(names)})"

    def chat(self, messages, tools, config):
        errors = []
        for i, provider in enumerate(self._providers):
            try:
                result = provider.chat(messages, tools, config)
                if i != self._active_index:
                    self._failover_count += 1
                    print(f"  [chain] Failed over to {provider.name()}")
                self._active_index = i
                return result
            except Exception as e:
                errors.append((provider.name(), str(e)))
                print(f"  [chain] {provider.name()} failed: {e}")
                continue
        error_summary = "; ".join(f"{n}: {e}" for n, e in errors)
        raise RuntimeError(f"All providers in chain failed: {error_summary}")

    def chat_stream(self, messages, tools, config):
        yield self.chat(messages, tools, config)

    def context_window(self) -> int:
        return min(p.context_window() for p in self._providers)

    def export_metrics(self):
        return {
            "active_provider": self._providers[self._active_index].name(),
            "failovers": self._failover_count,
            "providers": [p.export_metrics() for p in self._providers],
        }

    def report_late_failure(self, error):
        self._providers[self._active_index].report_late_failure(error)


# ---------------------------------------------------------------------------
# Layer 3: AdaptiveRouter — routes based on latency/error scoring
# ---------------------------------------------------------------------------

@dataclass
class _ProviderStats:
    total_calls: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    late_failures: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.total_errors + self.late_failures) / self.total_calls

    def score(self) -> float:
        """Lower is better. Combines error rate and latency."""
        return self.error_rate * 1000 + self.avg_latency_ms


class AdaptiveRouter(LlmProvider):
    """
    Adaptive routing based on latency and error scoring.
    Mirrors octos AdaptiveRouter — third layer of 3-layer failover.
    Routes to the best-performing provider based on recent metrics.
    """

    def __init__(self, providers: list[LlmProvider]):
        if not providers:
            raise ValueError("AdaptiveRouter requires at least one provider")
        self._providers = providers
        self._stats: dict[str, _ProviderStats] = {
            p.name(): _ProviderStats() for p in providers
        }
        self._last_used: str = providers[0].name()

    def name(self) -> str:
        return f"adaptive({', '.join(p.name() for p in self._providers)})"

    def _best_provider(self) -> LlmProvider:
        """Select the provider with the lowest score."""
        best = self._providers[0]
        best_score = float("inf")
        for p in self._providers:
            stats = self._stats.get(p.name(), _ProviderStats())
            s = stats.score()
            if s < best_score:
                best_score = s
                best = p
        return best

    def chat(self, messages, tools, config):
        provider = self._best_provider()
        stats = self._stats[provider.name()]
        stats.total_calls += 1

        start = time.time()
        try:
            result = provider.chat(messages, tools, config)
            elapsed_ms = (time.time() - start) * 1000
            stats.total_latency_ms += elapsed_ms
            self._last_used = provider.name()
            return result
        except Exception as e:
            stats.total_errors += 1
            # Try next best
            for fallback in self._providers:
                if fallback.name() == provider.name():
                    continue
                try:
                    result = fallback.chat(messages, tools, config)
                    self._last_used = fallback.name()
                    print(f"  [adaptive] Routed to {fallback.name()} after {provider.name()} failed")
                    return result
                except Exception:
                    self._stats[fallback.name()].total_errors += 1
                    continue
            raise e

    def chat_stream(self, messages, tools, config):
        yield self.chat(messages, tools, config)

    def context_window(self) -> int:
        return min(p.context_window() for p in self._providers)

    def export_metrics(self):
        return {
            "last_used": self._last_used,
            "provider_stats": {
                name: {
                    "calls": s.total_calls,
                    "errors": s.total_errors,
                    "avg_latency_ms": round(s.avg_latency_ms, 1),
                    "error_rate": round(s.error_rate, 3),
                    "score": round(s.score(), 1),
                }
                for name, s in self._stats.items()
            },
        }

    def report_late_failure(self, error):
        stats = self._stats.get(self._last_used)
        if stats:
            stats.late_failures += 1


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------

class OpenAIProvider(LlmProvider):
    """OpenAI GPT provider."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None,
                 api_base: str | None = None):
        from openai import OpenAI
        self._model = model
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base = api_base or os.environ.get("OPENAI_API_BASE")
        client_kwargs = {"api_key": key}
        if base:
            client_kwargs["base_url"] = base
        self._client = OpenAI(**client_kwargs)
        self._total_tokens = 0
        self._late_failures = 0

    def name(self) -> str:
        return f"openai/{self._model}"

    def chat(self, messages, tools, config):
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": config.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens

        response = self._client.chat.completions.create(**kwargs)
        self._total_tokens += response.usage.total_tokens if response.usage else 0

        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        return ChatResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )

    def chat_stream(self, messages, tools, config):
        # For simplicity, yield full response (OpenAI streaming would use stream=True)
        yield self.chat(messages, tools, config)

    def export_metrics(self):
        return {
            "total_tokens": self._total_tokens,
            "model": self._model,
            "late_failures": self._late_failures,
        }

    def report_late_failure(self, error):
        self._late_failures += 1
        print(f"  [provider] Late failure reported: {error}")


class MockProvider(LlmProvider):
    """
    Deterministic mock provider for CI-compatible testing.

    Executes a scripted sequence of tool calls, then returns a final message.
    """

    def __init__(self, script: list[dict] | None = None):
        self._script = script or self._default_pick_place_script()
        self._step = 0

    def name(self) -> str:
        return "mock"

    def chat(self, messages, tools, config):
        if self._step >= len(self._script):
            return ChatResponse(content="Mock agent: all scripted steps complete.")

        entry = self._script[self._step]
        self._step += 1

        if "tool_calls" in entry:
            return ChatResponse(
                tool_calls=entry["tool_calls"],
                finish_reason="tool_calls",
            )
        return ChatResponse(
            content=entry.get("content", "Done."),
            finish_reason="stop",
        )

    @staticmethod
    def _default_pick_place_script():
        """Default pick-and-place script for UR5e demo (from pick_and_place_demo.py)."""
        # Proven working joints from pick_and_place_demo.py:
        #   above_ball:  [2.2045, -1.6635, 2.1416, -2.049, -1.5708, 0.0]  → EE near ball
        #   grasp_ball:  [2.2045, -1.4535, 2.3026, -2.4199, -1.5708, 0.0] → EE at ball
        #   lift:        [2.2045, -1.7505, 1.944, -1.7642, -1.5708, 0.0]  → lifted
        #   above_plate: [3.445, -1.7505, 1.944, -1.7642, -1.5708, 0.0]   → EE near plate
        #   place_plate: [3.445, -1.4869, 2.326, -2.4099, -1.5708, 0.0]  → EE at plate (Z=0.039)
        home = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        above_ball = [2.2045, -1.6635, 2.1416, -2.049, -1.5708, 0.0]
        grasp_ball = [2.2045, -1.4535, 2.3026, -2.4199, -1.5708, 0.0]
        lift = [2.2045, -1.7505, 1.944, -1.7642, -1.5708, 0.0]
        above_plate = [3.445, -1.7505, 1.944, -1.7642, -1.5708, 0.0]
        place_plate = [3.445, -1.4869, 2.326, -2.4099, -1.5708, 0.0]

        return [
            # 1. Read current state
            {"tool_calls": [_mock_tc("call_1", "dora_read", {"input_id": "joint_positions"})]},
            # 2. Move to home
            {"tool_calls": [_mock_tc("call_2", "dora_call", {
                "output_id": "plan_request",
                "data": {"goal": home},
                "response_id": "plan_status",
            })]},
            # 3. Add objects to scene
            {"tool_calls": [
                _mock_tc("call_3a", "dora_send", {
                    "output_id": "scene_command",
                    "data": {"action": "add", "object": {"name": "red_ball", "type": "box",
                             "position": [0.35, -0.25, 0.028], "dimensions": [0.04, 0.04, 0.04],
                             "color": [1.0, 0.0, 0.0, 1.0]}},
                }),
                _mock_tc("call_3b", "dora_send", {
                    "output_id": "scene_command",
                    "data": {"action": "add", "object": {"name": "green_plate", "type": "box",
                             "position": [0.35, 0.25, 0.001], "dimensions": [0.12, 0.12, 0.01],
                             "color": [0.0, 1.0, 0.0, 1.0]}},
                }),
            ]},
            # 4. Open gripper
            {"tool_calls": [_mock_tc("call_4", "dora_call", {
                "output_id": "gripper_command", "data": {"action": "open"},
                "response_id": "gripper_status",
            })]},
            # 5. Move above red ball
            {"tool_calls": [_mock_tc("call_5", "dora_call", {
                "output_id": "plan_request",
                "data": {"goal": above_ball},
                "response_id": "plan_status",
            })]},
            # 6. Lower to grasp
            {"tool_calls": [_mock_tc("call_6", "dora_call", {
                "output_id": "plan_request",
                "data": {"goal": grasp_ball},
                "response_id": "plan_status",
            })]},
            # 7. Close gripper
            {"tool_calls": [_mock_tc("call_7", "dora_call", {
                "output_id": "gripper_command", "data": {"action": "close"},
                "response_id": "gripper_status",
            })]},
            # 8. Attach object
            {"tool_calls": [_mock_tc("call_8", "dora_send", {
                "output_id": "scene_command",
                "data": {"action": "attach", "name": "red_ball", "link": "link6"},
            })]},
            # 9. Lift up
            {"tool_calls": [_mock_tc("call_9", "dora_call", {
                "output_id": "plan_request",
                "data": {"goal": lift},
                "response_id": "plan_status",
            })]},
            # 10. Move above green plate
            {"tool_calls": [_mock_tc("call_10", "dora_call", {
                "output_id": "plan_request",
                "data": {"goal": above_plate},
                "response_id": "plan_status",
            })]},
            # 11. Lower to place
            {"tool_calls": [_mock_tc("call_11", "dora_call", {
                "output_id": "plan_request",
                "data": {"goal": place_plate},
                "response_id": "plan_status",
            })]},
            # 12. Open gripper
            {"tool_calls": [_mock_tc("call_12", "dora_call", {
                "output_id": "gripper_command", "data": {"action": "open"},
                "response_id": "gripper_status",
            })]},
            # 13. Detach object
            {"tool_calls": [_mock_tc("call_13", "dora_send", {
                "output_id": "scene_command",
                "data": {"action": "detach", "name": "red_ball"},
            })]},
            # 14. Return home
            {"tool_calls": [_mock_tc("call_14", "dora_call", {
                "output_id": "plan_request",
                "data": {"goal": home},
                "response_id": "plan_status",
            })]},
            {"content": "Pick and place complete. Red ball picked from (0.35, -0.25) and placed on green plate at (0.35, 0.25)."},
        ]


def _mock_tc(call_id: str, fn_name: str, fn_args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": fn_name, "arguments": json.dumps(fn_args)},
    }
