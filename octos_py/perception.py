"""
Perception aggregator — fuses data from dora-hub nodes (cameras, YOLO, sensors).

Provides a text summary for LLM context injection via the dora_perceive tool.
Dora-hub nodes (camera, YOLO, lidar, TTS) are just dora nodes with typed
inputs/outputs — integration is purely at the dataflow YAML level.
"""

import time
from dataclasses import dataclass, field


@dataclass
class Detection:
    """A single object detection from YOLO or similar."""
    label: str
    confidence: float
    bbox: list[float] = field(default_factory=list)  # [x1, y1, x2, y2]
    source: str = ""


class PerceptionAggregator:
    """
    Aggregates perception data from multiple dora-hub sensor nodes.

    Sensor types:
    - Camera frames (stored as metadata, not raw bytes for LLM)
    - YOLO/object detections
    - Force/torque sensor readings
    - Depth/pointcloud summaries
    - Generic sensor data
    """

    def __init__(self):
        self.camera_frames: dict[str, dict] = {}   # camera_id -> metadata
        self.detections: list[Detection] = []
        self.force_torque: dict[str, float] = {}
        self.depth_summary: dict[str, object] = {}
        self.generic_sensors: dict[str, object] = {}
        self._last_update: dict[str, float] = {}

    def update_camera(self, camera_id: str, metadata: dict) -> None:
        """Update camera frame metadata (resolution, timestamp, etc.)."""
        self.camera_frames[camera_id] = metadata
        self._last_update[f"camera:{camera_id}"] = time.time()

    def update_detections(self, detections_data: list[dict], source: str = "yolo") -> None:
        """Update object detections from YOLO or similar detector."""
        self.detections = [
            Detection(
                label=d.get("label", "unknown"),
                confidence=d.get("confidence", 0.0),
                bbox=d.get("bbox", []),
                source=source,
            )
            for d in detections_data
        ]
        self._last_update["detections"] = time.time()

    def update_force_torque(self, readings: dict[str, float]) -> None:
        """Update force/torque sensor readings."""
        self.force_torque = {**readings}
        self._last_update["force_torque"] = time.time()

    def update_depth(self, depth_id: str, summary: dict) -> None:
        """Update depth/pointcloud summary."""
        self.depth_summary[depth_id] = summary
        self._last_update[f"depth:{depth_id}"] = time.time()

    def update_generic(self, sensor_id: str, data: object) -> None:
        """Update any generic sensor data."""
        self.generic_sensors[sensor_id] = data
        self._last_update[f"sensor:{sensor_id}"] = time.time()

    def summarize(self) -> str:
        """
        Generate a text summary of all perception data for LLM context.

        Returns a human-readable summary suitable for injection into
        the agent's conversation as a tool result.
        """
        sections = []

        # Cameras
        if self.camera_frames:
            cam_lines = ["Cameras:"]
            for cam_id, meta in self.camera_frames.items():
                age = self._age_str(f"camera:{cam_id}")
                res = meta.get("resolution", "unknown")
                cam_lines.append(f"  {cam_id}: resolution={res}, {age}")
            sections.append("\n".join(cam_lines))

        # Detections
        if self.detections:
            det_lines = [f"Detections ({len(self.detections)} objects):"]
            for det in self.detections[:20]:  # cap at 20 for LLM context
                conf = f"{det.confidence:.0%}"
                bbox_str = f" bbox={det.bbox}" if det.bbox else ""
                det_lines.append(f"  {det.label} ({conf}){bbox_str}")
            if len(self.detections) > 20:
                det_lines.append(f"  ... and {len(self.detections) - 20} more")
            age = self._age_str("detections")
            det_lines.append(f"  Last updated: {age}")
            sections.append("\n".join(det_lines))

        # Force/torque
        if self.force_torque:
            ft_lines = ["Force/Torque:"]
            for name, val in self.force_torque.items():
                ft_lines.append(f"  {name}: {val:.3f}")
            age = self._age_str("force_torque")
            ft_lines.append(f"  Last updated: {age}")
            sections.append("\n".join(ft_lines))

        # Depth
        if self.depth_summary:
            depth_lines = ["Depth/Pointcloud:"]
            for depth_id, summary in self.depth_summary.items():
                age = self._age_str(f"depth:{depth_id}")
                depth_lines.append(f"  {depth_id}: {summary}, {age}")
            sections.append("\n".join(depth_lines))

        # Generic sensors
        if self.generic_sensors:
            sensor_lines = ["Other sensors:"]
            for sensor_id, data in self.generic_sensors.items():
                age = self._age_str(f"sensor:{sensor_id}")
                preview = str(data)[:100]
                sensor_lines.append(f"  {sensor_id}: {preview}, {age}")
            sections.append("\n".join(sensor_lines))

        if not sections:
            return "No perception data available."

        return "\n\n".join(sections)

    def has_anomaly(self, thresholds: dict | None = None) -> tuple[bool, str]:
        """
        Check for anomalies in perception data.

        Args:
            thresholds: Optional dict of threshold configs, e.g.
                {"force_max": 50.0, "min_detections": 1}

        Returns:
            (is_anomaly, description)
        """
        thresholds = thresholds or {}

        # Force/torque threshold check
        force_max = thresholds.get("force_max", 100.0)
        for name, val in self.force_torque.items():
            if abs(val) > force_max:
                return True, f"Force/torque {name} = {val:.2f} exceeds threshold {force_max}"

        # Detection count check
        min_detections = thresholds.get("min_detections")
        if min_detections is not None and len(self.detections) < min_detections:
            return True, f"Only {len(self.detections)} detections, expected >= {min_detections}"

        # Stale data check (no updates in last N seconds)
        stale_threshold = thresholds.get("stale_secs", 30.0)
        now = time.time()
        for key, ts in self._last_update.items():
            if now - ts > stale_threshold:
                return True, f"Sensor {key} is stale ({now - ts:.0f}s since last update)"

        return False, ""

    def _age_str(self, key: str) -> str:
        ts = self._last_update.get(key)
        if ts is None:
            return "age: unknown"
        age_ms = int((time.time() - ts) * 1000)
        if age_ms < 1000:
            return f"age: {age_ms}ms"
        return f"age: {age_ms / 1000:.1f}s"
