# 03 — LLM Agent

An octos agent powered by a real LLM that reasons about tool results and replans when navigation fails.

## Background

### The Problem

Your patrol pipeline (`01-pipeline-basics`) works perfectly — until a forklift parks in aisle 3. The robot tries `navigate_to(B)`, gets "path blocked," and the pipeline aborts. An operator has to SSH in, figure out what happened, manually skip station B, and restart. This happens 2-3 times per week. Each incident costs 20 minutes of downtime and an operator's attention.

### Why This Matters

Real warehouses are messy. Pallets move, doors close, forklifts block aisles, spills happen. A pipeline that aborts on the first unexpected event is brittle. But you also can't pre-script every possible failure mode — there are too many combinations.

An LLM agent closes this gap: it **reads the failure message, understands the context, and decides what to do** — just like a human operator would. "Station A is blocked? Skip it, try B. B is also blocked? Return home and report." The agent doesn't need a rule for every scenario; it reasons about the situation.

### Before vs After

| | Before (fixed pipeline) | After (LLM agent) |
|---|---|---|
| Forklift blocks aisle 3 | Pipeline aborts. Operator intervenes. 20min downtime. | Agent skips blocked station, continues patrol, reports. 0 downtime. |
| Unknown obstacle type | `navigate_to` returns error. Pipeline doesn't know what to do. | Agent reads error message, decides based on severity. |
| Operator workload | 2-3 manual interventions per week per robot | Agent handles routine failures autonomously |
| Failure transparency | Log says "navigation failed." Why? Unknown. | Agent explains: "Obstacle at y=8.0, skipped station A, completed B and home." |
| Scaling to 10 robots | 10x operator interventions | Each robot handles its own failures. Operator reviews summaries. |

## What You'll Learn

- OpenAI-compatible LLM provider (works with Ollama, vLLM, OpenAI API)
- Free-form LLM reasoning vs deterministic pipeline execution
- How agents detect failures and adapt their plan
- Tool call → result → reasoning loop

## Scenario

An obstacle at y=8.0 blocks the path to station A (y=10). The LLM agent must:
1. Attempt navigation to A
2. Detect the failure from the tool result
3. Decide to skip A and try B instead
4. Produce a summary of what happened

## Prerequisites

Install [Ollama](https://ollama.ai) and pull a model with good tool-calling support:
```bash
ollama pull qwen3:30b    # recommended (slow but reliable tool calls)
ollama pull qwen3:8b     # faster but may skip tool calls on some prompts
```

Or use any OpenAI-compatible endpoint (GPT-4o recommended for best tool calling).

## Run

```bash
cd 03-llm-agent
dora up && dora start dataflow.yaml --attach
```

### With OpenAI API

```bash
OPENAI_API_BASE=https://api.openai.com/v1 \
OPENAI_API_KEY=sk-... \
OPENAI_MODEL=gpt-4o-mini \
dora up && dora start dataflow.yaml --attach
```

## Expected Output

```
LLM iteration 1: get_robot_state() → Robot at home
LLM iteration 2: navigate_to(A) → FAILED: obstacle at y=8.0
LLM iteration 3: navigate_to(B) → FAILED: obstacle blocks path
LLM iteration 4: navigate_to(home) → Arrived at home

Agent: Patrol report — Station A: blocked by obstacle at y=8.0.
Station B: also blocked. Returned home successfully.
```

## Comparison: Pipeline vs LLM

| Feature | 01-pipeline-basics | 03-llm-agent |
|---------|-------------------|-------------|
| Decision making | DOT graph (fixed order) | LLM (free-form) |
| Failure handling | Abort or skip (configured) | Reason and adapt |
| LLM required | No | Yes |
| Deterministic | Yes | No |
| Best for | Known workflows | Unknown/dynamic situations |

## Next

Try [04-human-gate](../04-human-gate/) for operator approval gates.
