# 02 — Safety Tiers

Demonstrates octos safety tier authorization — tools are gated by permission level.

## Background

### The Problem

Your warehouse has three roles: **monitoring staff** who watch dashboards, **navigation operators** who drive robots, and **maintenance engineers** who control the arm. Last month, a monitoring intern accidentally sent a `navigate_to` command during a shift change — the robot drove into a pallet stack. The root cause: every session had full tool access regardless of the operator's role.

### Why This Matters

Industrial robots are not laptops — **wrong tool at the wrong time causes physical damage**. An LLM agent with unrestricted tool access is even riskier: it might call `navigate_to` during an arm inspection, or trigger `emergency_stop` when the arm is holding a fragile part mid-air. Safety tiers solve this:

- **Observe** — monitoring staff can read sensors, check state, view maps. Cannot move anything.
- **Safe motion** — operators can navigate the base. Cannot control the arm.
- **Full actuation** — engineers can move the arm, scan stations. Requires explicit escalation.
- **Emergency override** — supervisors only. Bypasses all restrictions.

The key insight: **the agent doesn't decide its own permissions**. The deployment configuration (env var, config file, or auth token) sets the tier. The agent discovers its limits at runtime and works within them.

### Before vs After

| | Before (no tiers) | After (safety tiers) |
|---|---|---|
| Intern on monitoring shift | Can send `navigate_to` — robot moves unexpectedly | Gets `PERMISSION DENIED` — robot stays put |
| LLM agent hallucinating | Calls any tool it wants, including dangerous ones | Blocked at the policy layer before execution |
| Compliance audit | "Who could have moved the robot?" — "Anyone." | Tier config + logs prove which session had which access |
| New deployment | Ship and pray | Start at `observe`, promote to `safe_motion` after validation |
| Incident response | Scramble to figure out who did what | Tier boundaries create clear responsibility zones |

## What You'll Learn

- Safety tier ordering: `observe < safe_motion < full_actuation < emergency_override`
- `RobotPermissionPolicy` blocks tools above the session's max tier
- How to configure tier via `SAFETY_TIER` env var
- How agents discover and report permission boundaries

## Safety Tiers

| Tier | Tools Allowed | Use Case |
|------|--------------|----------|
| `observe` | get_robot_state, get_map, read_lidar | Monitoring, inspection readout |
| `safe_motion` | + navigate_to, stop_base | Autonomous navigation |
| `full_actuation` | + scan_station | Arm control, physical interaction |
| `emergency_override` | All tools | Emergency response |

## Run

### Default: Observe-only (motion blocked)

```bash
cd 02-safety-tiers
dora up && dora start dataflow.yaml --attach
```

Output shows 3 tools allowed, 3 denied:
```
[safe-robot] ALLOWED: get_robot_state
[safe-robot] ALLOWED: get_map
[safe-robot] ALLOWED: read_lidar
[safe-robot] DENIED: navigate_to (needs safe_motion, have observe)
[safe-robot] DENIED: stop_base (needs safe_motion, have observe)
[safe-robot] DENIED: scan_station (needs full_actuation, have observe)
```

### Unlock navigation

Edit `dataflow.yaml` and change `SAFETY_TIER: "safe_motion"`, then rerun.
Now navigate_to and stop_base succeed, but scan_station is still denied.

## Files

| File | Purpose |
|------|---------|
| `dataflow.yaml` | 2-node dataflow with configurable SAFETY_TIER |
| `test_all_tools.dot` | 6-step pipeline testing every tool |
| `nodes/safe_robot_node.py` | Mock robot with tier enforcement |
| `nodes/agent_node.py` | Pipeline-driven agent |

## Next

Try [03-llm-agent](../03-llm-agent/) to see LLM reasoning about failures.
