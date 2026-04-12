# DOT Pipeline Guide

## Overview

Octos pipelines are defined as DOT (Graphviz) directed graphs. Each node represents a step — a tool call, a gate, or an LLM reasoning task. Edges define execution order.

## Basic Pipeline

```dot
digraph my_patrol {
    check [label="check_state" tool="get_robot_state"];
    nav_a [label="go_to_A" tool="navigate_to" args='{"waypoint": "A"}'];
    nav_b [label="go_to_B" tool="navigate_to" args='{"waypoint": "B"}'];

    check -> nav_a;
    nav_a -> nav_b;
}
```

Run: `OCTOS_PIPELINE=my_patrol.dot dora start dataflow.yaml`

## Node Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `label` | string | Human-readable step name |
| `tool` | string | Tool to call directly (no LLM needed) |
| `args` | JSON string | Arguments for the tool |
| `type` | string | Node type: `codergen` (default), `gate`, `safety_gate` |
| `deadline` | number | Max seconds for this step |
| `deadline_action` | string | `abort` (default), `skip`, `emergency_stop` |
| `invariant` | string | Post-condition expression |
| `checkpoint` | string | Checkpoint name for recovery |

## Direct Tool Execution

When a node has `tool=` and `args=`, the pipeline executor calls the tool directly without involving the LLM:

```dot
nav_a [
    label="navigate_to_A"
    tool="navigate_to"
    args='{"waypoint": "A"}'
    checkpoint="at_A"
];
```

This produces:
```
[pipeline] Step 2/4: navigate_to_A
[pipeline] Direct: navigate_to({'waypoint': 'A'})
[pipeline] Checkpoint: at_A
```

## Gate Nodes

Gates pause the pipeline and require approval before proceeding:

```dot
gate_b [
    label="Approve Zone B access?"
    type="gate"
];
```

In the Python agent, gates are checked against `GATE_POLICY` (substring match). In the Rust binary, gates are auto-approved.

### Gate policy (Python agent):
```yaml
env:
  GATE_POLICY: "Zone A,home"   # approve A and home, reject B
```

## Deadlines

Per-step timeouts with configurable actions:

```dot
nav_a [
    label="navigate_to_A"
    tool="navigate_to"
    args='{"waypoint": "A"}'
    deadline="120"
    deadline_action="abort"
];
```

Actions:
- `abort` — stop the pipeline on timeout (default)
- `skip` — skip this step and continue
- `emergency_stop` — trigger emergency stop tool

## Checkpoints

Mark progress for crash recovery:

```dot
nav_a [
    label="navigate_to_A"
    tool="navigate_to"
    args='{"waypoint": "A"}'
    checkpoint="reached_A"
];
```

The pipeline executor logs checkpoints. In production, these are persisted to disk for resumption.

## Cyclic Pipelines

For repeating workflows (patrols, monitoring loops):

```dot
digraph cyclic_patrol {
    graph [cycle=true max_cycles=3];

    check [label="check" tool="get_robot_state"];
    nav_a [label="go_A" tool="navigate_to" args='{"waypoint": "A"}'];
    nav_home [label="go_home" tool="navigate_to" args='{"waypoint": "home"}'];

    check -> nav_a;
    nav_a -> nav_home;
    nav_home -> check;  // back-edge creates the loop
}
```

- `cycle=true` — enables cyclic execution
- `max_cycles=3` — stop after 3 complete loops (0 = infinite)

## Invariants

Post-conditions checked after step execution:

```dot
nav_a [
    label="navigate_to_A"
    tool="navigate_to"
    args='{"waypoint": "A"}'
    invariant="position == 'A'"
];
```

## Examples

### Simple patrol (4 steps)
```dot
digraph patrol {
    check [label="check_state" tool="get_robot_state"];
    nav_a [label="go_A" tool="navigate_to" args='{"waypoint": "A"}' checkpoint="at_A"];
    nav_b [label="go_B" tool="navigate_to" args='{"waypoint": "B"}' checkpoint="at_B"];
    go_home [label="go_home" tool="navigate_to" args='{"waypoint": "home"}' checkpoint="at_home"];

    check -> nav_a -> nav_b -> go_home;
}
```

### Gated inspection (with operator approval)
```dot
digraph gated {
    check [label="check" tool="get_robot_state"];
    gate_a [label="Approve Station A?" type="gate"];
    nav_a [label="go_A" tool="navigate_to" args='{"waypoint": "A"}'];
    gate_b [label="Approve Station B?" type="gate"];
    nav_b [label="go_B" tool="navigate_to" args='{"waypoint": "B"}'];

    check -> gate_a -> nav_a -> gate_b -> nav_b;
}
```

### Safety-tiered tools
```dot
digraph safety_test {
    t1 [label="read_state" tool="get_robot_state"];       // observe tier
    t2 [label="read_lidar" tool="read_lidar"];            // observe tier
    t3 [label="navigate" tool="navigate_to" args='{"waypoint": "A"}'];  // safe_motion tier
    t4 [label="scan" tool="scan_station"];                 // full_actuation tier

    t1 -> t2 -> t3 -> t4;
}
```

## Runtime Selection

| Runner | Pipeline support | LLM needed | Install |
|--------|-----------------|------------|---------|
| Python agent (`agent_node.py`) | Full (gates, policies) | No for pipeline mode | `pip install dora-rs` |
| Rust binary (`octos-dora-agent`) | Full (gates auto-approve) | No | `cargo install` |
| `octos serve` + MCP bridge | Full (all features) | Yes (for LLM mode) | `cargo install octos-cli` |
