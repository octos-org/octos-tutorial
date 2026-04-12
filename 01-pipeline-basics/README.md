# 01 — Pipeline Basics

The simplest octos example: a 2-node dataflow where an agent executes a DOT pipeline to patrol stations.

## Background

### The Problem

A warehouse runs 3 patrol shifts daily. Each shift, an operator manually drives the robot through stations A→B→home, typing navigation commands into a terminal. Different operators sequence the stations differently. When a robot stalls mid-patrol, nobody knows which stations were completed — the operator has to start over or guess.

### Why This Matters

In production, **robots need repeatable missions, not ad hoc commands**. A gas pipeline inspection that skips station 3 because an operator forgot is a compliance violation. A factory patrol that visits stations in random order makes anomaly tracking impossible. You need a mission definition that is:

- **Auditable** — anyone can read the DOT file and know exactly what the robot will do
- **Resumable** — checkpoints record progress, so a restart picks up where it left off
- **Deterministic** — same pipeline, same execution, every time

### Before vs After

| | Before (ad hoc scripts) | After (DOT pipeline) |
|---|---|---|
| Mission definition | Scattered across Python scripts, bash aliases, operator notes | Single `patrol.dot` file — 15 lines, version-controlled |
| Execution order | Depends on who's operating | Guaranteed: check→A→B→home, every run |
| Failure recovery | "Where did it stop? Let me SSH in and check..." | Checkpoints: `at_A`, `at_B`, `at_home` — resume from last |
| Onboarding | New operator needs training on the script | New operator reads the DOT graph in 30 seconds |
| Testing | Run the whole thing and watch | Pipeline engine validates graph at load time |

## What You'll Learn

- How octos DOT pipelines define tool execution sequences
- Direct tool execution from pipeline nodes (no LLM)
- `tool=` and `args=` attributes in DOT nodes
- Checkpoints for tracking progress

## Architecture

```
┌─────────┐  skill_request  ┌────────────┐
│  agent  │ ──────────────→ │ mock-robot  │
│         │ ←────────────── │            │
└─────────┘  skill_result   └────────────┘
```

The agent reads `patrol.dot` and executes each step:
1. `get_robot_state()` — check current position
2. `navigate_to(A)` — go to station A (checkpoint: at_A)
3. `navigate_to(B)` — go to station B (checkpoint: at_B)
4. `navigate_to(home)` — return home (checkpoint: at_home)

## Run

```bash
pip install dora-rs pyarrow numpy   # one-time setup
cd 01-pipeline-basics
dora up && dora start dataflow.yaml --attach
```

## Expected Output

```
Octos Agent — Pipeline Basics
Pipeline: patrol (4 steps)

[pipeline] Step 1/4: check_state
[pipeline] Direct: get_robot_state({})
[mock-robot] Arrived at home
[pipeline] Step 2/4: navigate_to_A
[pipeline] Direct: navigate_to({'waypoint': 'A'})
[mock-robot] Navigating to A...
[mock-robot] Arrived at A (-0.45, 10.00)
[pipeline] Checkpoint: at_A
[pipeline] Step 3/4: navigate_to_B
[pipeline] Direct: navigate_to({'waypoint': 'B'})
[mock-robot] Arrived at B (-0.70, 18.00)
[pipeline] Checkpoint: at_B
[pipeline] Step 4/4: navigate_home
[pipeline] Direct: navigate_to({'waypoint': 'home'})
[mock-robot] Arrived at home (-0.21, 0.28)
[pipeline] Checkpoint: at_home

Complete!
```

## Files

| File | Purpose |
|------|---------|
| `dataflow.yaml` | 2-node dora dataflow |
| `patrol.dot` | 4-step patrol pipeline |
| `nodes/agent_node.py` | Octos agent with pipeline engine |
| `nodes/mock_robot_node.py` | Simulated robot (no hardware needed) |

## Next

Try [02-safety-tiers](../02-safety-tiers/) to learn about permission gating.
