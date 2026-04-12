# Safety Tiers Reference

## Tier Ordering

```
Observe < SafeMotion < FullActuation < EmergencyOverride
```

A session with `max_tier = SafeMotion` can use tools up to `SafeMotion` but NOT `FullActuation` or `EmergencyOverride`.

## Tier Definitions

| Tier | Value | Use Case | Example Tools |
|------|-------|----------|---------------|
| **Observe** | `observe` | Read-only monitoring | `get_robot_state`, `get_map`, `read_lidar` |
| **SafeMotion** | `safe_motion` | Controlled base movement | `navigate_to`, `stop_base` |
| **FullActuation** | `full_actuation` | Arm control, physical interaction | `scan_station`, `pick_object` |
| **EmergencyOverride** | `emergency_override` | Bypass all restrictions | `emergency_stop`, `force_shutdown` |

## Configuration

### Per-session (env var)
```yaml
env:
  SAFETY_TIER: "observe"    # this session can only read sensors
```

### Per-tool (nav_tool_map.json)
```json
{
  "name": "navigate_to",
  "safety_tier": "safe_motion",
  "description": "Navigate to a station"
}
```

### Authorization check
```
tool.required_tier <= session.max_tier → ALLOWED
tool.required_tier >  session.max_tier → DENIED
```

## Rust API

```rust
use octos_agent::permissions::{SafetyTier, RobotPermissionPolicy};

let policy = RobotPermissionPolicy {
    max_tier: SafetyTier::SafeMotion,
    workspace: None,
};

let (allowed, err) = policy.authorize("navigate_to", SafetyTier::SafeMotion);
assert!(allowed);

let (allowed, err) = policy.authorize("scan_station", SafetyTier::FullActuation);
assert!(!allowed);
// err: "permission denied: tool 'scan_station' requires tier 'full_actuation'
//       but session allows up to 'safe_motion'"
```

## Python API

```python
from octos_py.safety import SafetyTier, RobotPermissionPolicy

policy = RobotPermissionPolicy(max_tier=SafetyTier.SAFE_MOTION)

allowed, err = policy.authorize("navigate_to", SafetyTier.SAFE_MOTION)
# allowed=True

allowed, err = policy.authorize("scan_station", SafetyTier.FULL_ACTUATION)
# allowed=False, err="permission denied: ..."
```

## Deployment Patterns

### Progressive rollout
```
Day 1: SAFETY_TIER=observe       → monitor-only, validate sensors
Day 3: SAFETY_TIER=safe_motion   → enable navigation after validation
Day 7: SAFETY_TIER=full_actuation → enable arm after navigation is stable
```

### Role-based access
```
Monitoring staff:    SAFETY_TIER=observe
Navigation operator: SAFETY_TIER=safe_motion
Maintenance engineer: SAFETY_TIER=full_actuation
Site supervisor:     SAFETY_TIER=emergency_override
```

### Incident response
```
Normal operation:    SAFETY_TIER=safe_motion
Incident detected:   SAFETY_TIER=observe (freeze all motion)
Emergency:          SAFETY_TIER=emergency_override (supervisor takes control)
```
