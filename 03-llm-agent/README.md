# 03 — LLM Agent

An octos agent powered by a real LLM that reasons about tool results and replans when navigation fails.

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

Try [05-slam-nav-sim](../05-slam-nav-sim/) for visual MuJoCo simulation.
