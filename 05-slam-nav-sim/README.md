# 05 — SLAM Navigation Simulation

Visual MuJoCo simulation where a Hunter SE robot navigates a warehouse. 5 dataflows demonstrate progressive octos integration.

## Background

### The Problem

You're building a warehouse patrol robot. Your pipeline works in the mock examples (01-04), but you need to answer harder questions: Does the path planner actually avoid obstacles? Does the Pure Pursuit controller track the path within 0.5m? Does the robot arrive at station A before the 120s deadline? You can't answer these with a mock that just returns "Arrived at A" after 0.5 seconds.

### Why This Matters

Mock robots test your **agent logic**. Simulation tests your **whole stack**: physics, control, planning, perception, and agent logic together. The gap matters:

- A mock `navigate_to` always succeeds. In simulation, the robot can overshoot, oscillate, or stall — exactly like real hardware.
- A mock returns instantly. Simulation reveals timing: does the planner keep up at 100ms? Does the LLM respond before the robot reaches the obstacle?
- A mock has no spatial awareness. Simulation shows: is the robot actually at y=10.0 when it claims to be at station A?

This example runs the same 5 octos patterns (pipeline, replan, safety, cyclic) but with a real physics loop — MuJoCo at 100Hz, Frenet planning at 10Hz, Pure Pursuit control at 100Hz, Rerun visualization at 10Hz.

### Before vs After

| | Before (mock robot) | After (MuJoCo simulation) |
|---|---|---|
| "Does the path planner work?" | "It returned a result" | Watch the blue path in Rerun — see it curve around waypoints |
| "Does the robot reach station A?" | "The mock said yes" | Check: robot position is (-0.45, 10.02) — within 0.5m threshold |
| "How long does a full patrol take?" | ~2 seconds (mock sleeps) | ~90 seconds with real physics, real control, real timing |
| "What happens when the planner is slow?" | Nothing — mock is instant | Robot overshoots the turn because the path update was late |
| "Can I show this to stakeholders?" | Terminal text output | Rerun 3D: robot moving through warehouse, LiDAR pointcloud, path overlay |

## What You'll Learn

- Full dora-nav pipeline ported to Python (5 C++ nodes → Python)
- MuJoCo physics simulation with Rerun 3D visualization
- Octos pipeline patrol (deterministic, no LLM)
- Octos LLM replanning (obstacle handling)
- Octos safety tiers (observe-only mode)
- Octos cyclic patrol (continuous loops)

## Prerequisites

```bash
pip install dora-rs pyarrow numpy mujoco rerun-sdk
```

You also need:
- **dora-mujoco**: symlink to `dora-moveit2/dora-mujoco` (MuJoCo sim node)
- **Model meshes**: symlink `models/hunter_se` and `models/GEN72` to mesh directories
- **Ollama** (for LLM examples): `ollama pull qwen3:30b`

## Dataflows

| Dataflow | What | LLM? |
|----------|------|------|
| `dataflow_nav_sim_py.yaml` | Pure Python nav — robot follows 1065-waypoint path | No |
| `dataflow_octos_nav.yaml` | Pipeline patrol A→B→home (10 steps, mock provider) | No |
| `dataflow_octos_replan.yaml` | Obstacle at y=8, LLM agent replans | Yes |
| `dataflow_octos_safety.yaml` | Observe-only tier blocks navigation | Yes |
| `dataflow_octos_cyclic.yaml` | 2-loop continuous patrol | No |
| `dataflow_obstacle_avoidance.yaml` | Nav + obstacle detection + costmap + recovery | No |

## Setup (macOS)

```bash
cd 05-slam-nav-sim

# Symlink dora-mujoco
ln -sf /path/to/dora-moveit2/dora-mujoco dora-mujoco

# Symlink model meshes
ln -sf /path/to/dora-moveit2/examples/hunter_with_arm/models/hunter_se models/hunter_se
ln -sf /path/to/dora-moveit2/examples/hunter_with_arm/models/GEN72 models/GEN72
```

## Run

### Pure navigation (no octos)
```bash
export MUJOCO_GL=cgl  # macOS
dora up && dora start dataflow_nav_sim_py.yaml --attach
```

### Octos pipeline patrol
```bash
dora up && dora start dataflow_octos_nav.yaml --attach
```

### Octos LLM replan (with obstacle)
```bash
dora up && dora start dataflow_octos_replan.yaml --attach
```

## Architecture

```
mujoco-sim → pose-extractor → road-lane-pub ← pub-road
                                    ↓
                               planning ← task-pub-stub
                              ↙        ↘
                        lat-control  lon-control
                              ↘        ↙
                             nav-bridge ←→ robot-edge-a (optional)
                                 ↓
                            wheel_commands → mujoco-sim

                            rerun (3D viz)
```

## Files

| File | Purpose |
|------|---------|
| `nodes/pub_road_node.py` | Waypoint publisher (Python port of C++) |
| `nodes/road_lane_publisher_node.py` | Frenet coordinate converter |
| `nodes/planning_node.py` | Frenet path planner (30-point local path) |
| `nodes/lat_controller_node.py` | Pure Pursuit lateral control |
| `nodes/lon_controller_node.py` | Longitudinal speed control |
| `nodes/nav_bridge_node.py` | Nav bridge + safety tiers + obstacle sim |
| `nodes/pose_extractor_node.py` | Extract Pose2D from MuJoCo qpos |
| `nodes/rerun_viz_node.py` | Rerun 3D visualization |
| `nodes/octos_robot_edge_node.py` | Octos LLM agent (nav-only) |
| `Waypoints.txt` | 1065 waypoints (52.8m warehouse path) |
| `models/hunter_se_warehouse.xml` | MuJoCo Hunter SE + warehouse model |
