# Octos Tutorial

Hands-on examples for building industrial robot agents with [octos](https://github.com/octos-org/octos) and [dora-rs](https://github.com/dora-rs/dora).

## Examples

| # | Example | What You'll Learn | Dependencies |
|---|---------|-------------------|-------------|
| 01 | [Pipeline Basics](01-pipeline-basics/) | DOT pipeline engine, deterministic tool execution | dora-rs only |
| 02 | [Safety Tiers](02-safety-tiers/) | Permission gating, tier-based tool authorization | dora-rs only |
| 03 | [LLM Agent](03-llm-agent/) | Free-form LLM reasoning, replanning on failure | dora-rs + Ollama |
| 05 | [SLAM Nav Sim](05-slam-nav-sim/) | Visual MuJoCo simulation, full navigation pipeline | dora-rs + MuJoCo + Rerun |

Examples 01-02 run in under 5 minutes with just `pip install dora-rs pyarrow numpy`.

## Quick Start

```bash
# Install dependencies
pip install dora-rs pyarrow numpy

# Run the simplest example
cd 01-pipeline-basics
dora up && dora start dataflow.yaml --attach
```

## Setup

```bash
git clone https://github.com/octos-org/octos-tutorial.git
cd octos-tutorial
pip install -r requirements.txt
```

For visual simulation (example 05), also install:
```bash
pip install mujoco rerun-sdk
```

For LLM examples (03), install [Ollama](https://ollama.ai) and pull a model:
```bash
ollama pull qwen3:30b
```

## Architecture

Octos is an agentic OS for robots. These tutorials use the Python SDK (`octos_py/`) which mirrors the Rust `octos-agent` crate.

```
┌─────────────────────────────────────┐
│           Octos Agent               │
│  ┌───────────┐  ┌────────────────┐  │
│  │  Pipeline  │  │  LLM Provider  │  │
│  │  Engine    │  │  (OpenAI/Mock) │  │
│  └─────┬─────┘  └───────┬────────┘  │
│        │    Tool Registry    │       │
│        └────────┬────────────┘       │
│          ┌──────┴──────┐             │
│          │ Safety Tier │             │
│          │   Policy    │             │
│          └──────┬──────┘             │
└─────────────────┼───────────────────┘
                  │ skill_request / skill_result
                  ▼
         ┌────────────────┐
         │  Robot Bridge  │ (dora dataflow node)
         └────────────────┘
```

## Octos Python SDK

The `octos_py/` directory contains a standalone Python implementation of the octos agent framework:

| Module | Purpose |
|--------|---------|
| `agent.py` | Agent loop, tool dispatch, pipeline integration |
| `pipeline.py` | DOT graph parser, PipelineExecutor, CyclicPipelineExecutor |
| `safety.py` | SafetyTier, RobotPermissionPolicy, WorkspaceBounds |
| `provider.py` | LLM providers (OpenAI, Mock, Retry, ProviderChain) |
| `tools.py` | Tool trait, ToolRegistry, ToolResult |
| `mission.py` | Mission lifecycle, MissionExecutor, checkpoints |
| `session.py` | Session management, episode memory |
| `skills.py` | SKILL.md loader, skill injection |
| `perception.py` | Sensor aggregation |

## License

Apache-2.0
