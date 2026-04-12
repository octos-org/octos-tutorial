# 05 — SLAM Navigation Simulation

Visual MuJoCo simulation where a Hunter SE robot navigates a warehouse. 5 dataflows demonstrate progressive octos integration.

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
