# Octos Architecture

## Overview

Octos is a Rust-native, API-first agentic OS for robots. It orchestrates LLM-driven tool execution with safety controls, pipeline workflows, and multi-channel communication.

```
┌─────────────────────────────────────────────────────────────┐
│                        octos-cli                            │
│  Commands: chat, serve, gateway, init, status, clean        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                       octos-agent                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Agent Loop   │  │  Tool System  │  │  Safety & Hooks  │  │
│  │              │  │              │  │                  │  │
│  │  LLM call    │  │  Registry    │  │  SafetyTier      │  │
│  │  Tool exec   │  │  Policy      │  │  Permissions     │  │
│  │  Compaction  │  │  LRU defer   │  │  WorkspaceBounds │  │
│  │  Budget      │  │  MCP bridge  │  │  HookExecutor    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│  ┌──────┴─────────────────┴────────────────────┴─────────┐  │
│  │                    Pipeline Engine                     │  │
│  │  DOT graph → topological sort → per-node execution    │  │
│  │  Checkpoints, deadlines, invariants, cyclic loops     │  │
│  │  Human gates, parallel fan-out, model selection       │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
┌─────────────┴──────┐   ┌─────────────┴──────┐
│     octos-llm      │   │    octos-memory     │
│                    │   │                    │
│  OpenAI, Anthropic │   │  EpisodeStore      │
│  Gemini, DeepSeek  │   │  HybridSearch      │
│  OpenRouter        │   │  (BM25 + vector)   │
│  RetryProvider     │   │  HNSW index        │
│  ProviderChain     │   │                    │
│  AdaptiveRouter    │   │                    │
└────────────────────┘   └────────────────────┘
```

## Agent Loop

The core execution cycle in `octos-agent/src/agent/loop_runner.rs`:

```
1. Build messages (system prompt + history + memory context)
2. Call LLM with tool schemas (filtered by policy + provider)
3. If tool_calls returned:
   a. Execute each tool (with safety tier check)
   b. Append tool results to messages
   c. Go to step 2
4. If EndTurn or budget exceeded:
   a. Return final response
5. Context compaction if token budget fills
```

### Key behaviors:
- **Max iterations**: Default 50, configurable per agent
- **Tool timeout**: Default 30s, max 600s per tool call
- **Compaction**: When context approaches limit, summarize older messages
- **Loop detection**: Detects repeated tool calls and breaks the cycle
- **Budget tracking**: Counts tokens across iterations

## Tool System

Tools implement the `Tool` trait:

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> serde_json::Value;
    fn required_safety_tier(&self) -> SafetyTier { SafetyTier::Observe }
    async fn execute(&self, args: &Value) -> Result<ToolResult>;
}
```

### Tool lifecycle:
- **Registration**: Tools registered in `ToolRegistry` (HashMap by name)
- **Schema generation**: `to_openai_schema()` for LLM function calling
- **Policy filtering**: `ToolPolicy` allow/deny lists with wildcard support
- **LRU deferral**: 15 active tools, idle tools auto-evict, `spawn_only` pinned
- **Safety check**: `RobotPermissionPolicy::authorize()` before execution

## Safety Tiers

Ordered from least to most dangerous:

```
Observe < SafeMotion < FullActuation < EmergencyOverride
```

Each tool declares its `required_safety_tier()`. Each session has a `max_tier`. Authorization: `required <= max_tier → allowed`.

## Dora Integration

The `octos-dora-mcp` crate bridges octos tools to dora-rs dataflows:

```
┌──────────────────┐     bridge channel     ┌──────────────────┐
│  octos agent     │ ─────────────────────→ │  dora event loop │
│  (tool.execute)  │ ←───────────────────── │  (send/recv)     │
└──────────────────┘     ToolRequest        └──────────────────┘
                         + reply_tx              │
                                            skill_request / skill_result
                                                 │
                                            ┌────┴─────────────┐
                                            │  robot nodes     │
                                            └──────────────────┘
```

### Two deployment patterns:

**Pattern A: Embedded (octos-dora-agent binary)**
- Rust binary runs as a dora node
- Pipeline executor built-in (DOT → tool calls → dora)
- No external process needed
- Best for: deterministic pipelines, CI/CD, edge deployment

**Pattern B: Serve + MCP bridge (production)**
- `octos serve` runs the full agent (all features)
- MCP stdio bridge connects to dora via Unix socket
- Best for: LLM reasoning, web dashboard, multi-channel

## LLM Providers

3-layer failover architecture:

```
Layer 1: RetryProvider (exponential backoff on 429/5xx)
    ↓
Layer 2: ProviderChain (sequential failover across providers)
    ↓
Layer 3: AdaptiveRouter (hedge racing, lane scoring, circuit breakers)
```

Supported: OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, + 8 OpenAI-compatible via `with_base_url()`.

## Pipeline Engine

DOT-graph based multi-step workflows (see [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)):

- Per-node model selection via `ModelStylesheet`
- Parallel fan-out spawns N concurrent workers
- Checkpoints for crash recovery
- Deadlines with configurable actions (abort, skip, emergency_stop)
- Invariant conditions checked post-execution
- Human gates requiring confirmation
- Cyclic execution with `max_cycles`

## Memory

- **EpisodeStore**: redb database, stores task completion summaries
- **HybridSearch**: BM25 + vector (cosine similarity) with HNSW index
- Configurable weights (default 0.7 vector / 0.3 BM25)
- 7-day recent memory window
