---
name: navigation
description: Hunter SE mobile base navigation using dora-nav pipeline
version: 1.0.0
author: octos_inspection
always: true
robot_type: hunter_se
required_safety_tier: safe_motion
timeout_secs: 120
---

# Navigation Skill

Control the Hunter SE differential-drive mobile base through warehouse stations.

## Robot: Hunter SE

- Drive: 4-wheel differential drive
- Wheelbase: 0.55m, wheel radius: 0.1m
- Path: 1065-waypoint global path along Y-axis (heading North)
- Speed control: torque-to-velocity mapping via dora-nav C++ pipeline

## Stations

| ID   | Y Position | Description |
|------|-----------|-------------|
| A    | 10.0m     | Station A — 3 colored cubes on table at x=2.5 |
| B    | 18.0m     | Station B — 3 colored cubes on table at x=2.5 |
| home | 0.28m     | Path start (home position) |

## Tool Usage

Navigate to a station:
```
navigate_to(waypoint="A")
```

Check current position:
```
get_robot_state()
```

Get map with all stations:
```
get_map()
```

Emergency stop:
```
stop_base()
```

Read LiDAR status:
```
read_lidar()
```

## Important Rules

- Path is one-way (North); navigate_to "home" uses stall detection (10s timeout)
- Station arrival threshold: 1.0m from target Y coordinate
- Navigation timeout: 120 seconds per station
- Robot follows dora-nav global path — cannot go to arbitrary coordinates
