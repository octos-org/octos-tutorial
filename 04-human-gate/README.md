# 04 — Human Gate

Demonstrates operator approval gates in octos pipelines — the robot pauses before each navigation and waits for approval.

## Background

### The Problem

Your chemical plant has 3 inspection zones: Zone A (routine), Zone B (high-voltage equipment), and Zone C (clean room). The robot patrols all three zones autonomously. Last week, maintenance was working in Zone B with the power panels open. The robot drove in anyway — nobody told it to skip Zone B. The near-miss triggered a safety review: **"How do we let the robot run autonomously while keeping humans in control of where it goes?"**

### Why This Matters

Full autonomy is fast but blind to context. Full manual control is safe but slow. Most real operations need **mixed autonomy**: the robot handles the routine steps, but a human approves high-stakes decisions. This is not a rare pattern — it's how aviation (autopilot + pilot), surgery (da Vinci + surgeon), and nuclear power (automation + operator) all work.

Gate nodes give you this pattern in 3 lines of DOT:
```
gate_b [label="Approve Zone B?" type="gate"]
```

The pipeline pauses. The policy checks. If approved, the robot proceeds. If rejected, it skips that zone and continues. No code changes, no redeployment — just a policy update.

### Before vs After

| | Before (full autonomy) | After (gated autonomy) |
|---|---|---|
| Maintenance in Zone B | Robot enters anyway. Near-miss. Safety report. | Gate rejects Zone B. Robot skips it. Zero risk. |
| Shift change | New shift has no way to restrict the robot | Update `GATE_POLICY` — takes 5 seconds |
| Audit trail | "The robot went everywhere. We don't know why." | Logs show: "Gate B REJECTED by policy: zone under maintenance" |
| Emergency | Robot is somewhere in the plant. Stop it manually? | Reject all gates. Robot finishes current step and holds. |
| Scaling | Works until someone forgets to tell the robot about maintenance | Policy is centralized, version-controlled, auditable |

## What You'll Learn

- Gate nodes in DOT pipelines (`type="gate"`)
- Gate approval/rejection policies
- Skipping pipeline steps when gates are rejected
- Industrial pattern: human-in-the-loop safety

## Scenario

The pipeline has 3 gates (before stations A, B, and home). The `GATE_POLICY` controls which gates are approved:

- **Default**: `"Station A,home"` — approves A and home, **rejects B** (zone under maintenance)
- Robot navigates to A, skips B, returns home

## Run

```bash
cd 04-human-gate
dora up && dora start dataflow.yaml --attach
```

## Expected Output

```
[gate 2/10] APPROVED: Approve navigation to Station A?
  Reason: policy: 'station a' matches gate
[step 3/10] navigate_to({'waypoint': 'A'})
  Checkpoint: at_A
[step 4/10] read_lidar({})
[gate 5/10] REJECTED: Approve navigation to Station B?
  Reason: policy: gate not in approved list [Station A,home]
[step 6/10] SKIPPED: navigate_to_B (gate rejected)
[step 7/10] read_lidar({})
[gate 8/10] APPROVED: Approve return to home?
  Reason: policy: 'home' matches gate
[step 9/10] navigate_to({'waypoint': 'home'})
  Checkpoint: at_home

Mission Summary:
  Approved gates: ['Approve navigation to Station A?', 'Approve return to home?']
  Rejected gates: ['Approve navigation to Station B?']
  Tools executed: [get_robot_state, navigate_to, read_lidar, read_lidar, navigate_to, get_robot_state]
```

## Customize Gate Policy

Edit `GATE_POLICY` in `dataflow.yaml`:

| Policy | Effect |
|--------|--------|
| `"all"` | Approve everything (no gates block) |
| `"Station A,home"` | Skip station B |
| `"home"` | Skip A and B, only return home |
| `""` | Reject all gates (nothing navigates) |

## Files

| File | Purpose |
|------|---------|
| `dataflow.yaml` | 2-node dataflow with GATE_POLICY |
| `inspection_gated.dot` | 10-node pipeline with 3 gate nodes |
| `nodes/agent_node.py` | Pipeline executor with gate checking |
| `nodes/mock_robot_node.py` | Simulated robot |

## Next

Try [05-slam-nav-sim](../05-slam-nav-sim/) for visual MuJoCo simulation.
