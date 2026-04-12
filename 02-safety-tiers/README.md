# 02 — Safety Tiers

Demonstrates octos safety tier authorization — tools are gated by permission level.

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
