"""
SKILL.md loader — mirrors octos_agent::skills::SkillsLoader.

Reads SKILL.md files with YAML frontmatter and concatenates their content
into the system prompt. Skills with `always: true` are always loaded.
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LifecycleStep:
    """A single step in a hardware lifecycle phase."""
    label: str
    command: str
    timeout_secs: int = 30
    retries: int = 0
    critical: bool = True


@dataclass
class HardwareLifecycle:
    """Hardware lifecycle declaration for a plugin.

    Mirrors octos_plugin::lifecycle::HardwareLifecycle.
    """
    preflight: list[LifecycleStep] = field(default_factory=list)
    init: list[LifecycleStep] = field(default_factory=list)
    ready_check: list[LifecycleStep] = field(default_factory=list)
    shutdown: list[LifecycleStep] = field(default_factory=list)
    emergency_shutdown: list[LifecycleStep] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareLifecycle":
        """Parse from a frontmatter dict."""
        def parse_steps(raw: list) -> list[LifecycleStep]:
            steps = []
            for item in raw:
                if isinstance(item, dict):
                    steps.append(LifecycleStep(
                        label=item.get("label", ""),
                        command=item.get("command", ""),
                        timeout_secs=int(item.get("timeout_secs", 30)),
                        retries=int(item.get("retries", 0)),
                        critical=bool(item.get("critical", True)),
                    ))
            return steps

        return cls(
            preflight=parse_steps(data.get("preflight", [])),
            init=parse_steps(data.get("init", [])),
            ready_check=parse_steps(data.get("ready_check", [])),
            shutdown=parse_steps(data.get("shutdown", [])),
            emergency_shutdown=parse_steps(data.get("emergency_shutdown", [])),
        )

    def run_phase(self, phase_name: str, steps: list[LifecycleStep]) -> tuple[bool, str]:
        """Run a lifecycle phase. Returns (success, error_message)."""
        for i, step in enumerate(steps):
            last_error = None
            for attempt in range(step.retries + 1):
                try:
                    result = subprocess.run(
                        ["sh", "-c", step.command],
                        capture_output=True,
                        text=True,
                        timeout=step.timeout_secs,
                    )
                    if result.returncode == 0:
                        last_error = None
                        break
                    last_error = f"exit {result.returncode}: {result.stderr.strip()}"
                except subprocess.TimeoutExpired:
                    last_error = f"timed out after {step.timeout_secs}s"
                except Exception as e:
                    last_error = str(e)

            if last_error and step.critical:
                return False, f"{phase_name}/{step.label}: {last_error}"
        return True, ""


@dataclass
class SkillInfo:
    name: str
    description: str
    version: str
    author: str
    always: bool
    content: str
    path: str
    # New robot-specific fields (Area 7)
    robot_type: str = ""  # e.g. "gen72", "hunter_se", "ur5e"
    required_safety_tier: str = "observe"  # observe, safe_motion, full_actuation, emergency_override
    workspace: Optional[dict] = field(default=None)  # {x_min, x_max, y_min, y_max, z_min, z_max}
    hardware_requirements: Optional[list] = field(default=None)  # e.g. ["gripper", "camera"]
    timeout_secs: int = 0  # 0 means no timeout
    depends_on: Optional[list] = field(default=None)  # other skill names this depends on


def _parse_workspace(value: str) -> dict:
    """Parse a workspace string into a dict of float values.

    Expects format: ``x_min=-1.0 x_max=1.0 y_min=-1.0 y_max=1.0 z_min=0.0 z_max=2.0``
    """
    result: dict = {}
    for token in value.split():
        if "=" in token:
            k, _, v = token.partition("=")
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                result[k.strip()] = v.strip()
    return result


_LIST_FIELDS = {"hardware_requirements", "depends_on"}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from a SKILL.md file.

    List fields (``hardware_requirements``, ``depends_on``) are returned as
    ``list[str]`` by splitting on commas and stripping whitespace.
    The ``workspace`` field is returned as a ``dict[str, float]``.
    Boolean values ``true``/``false`` are converted to ``bool``.
    Integer values are converted to ``int`` when possible.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text

    frontmatter_text = match.group(1)
    content = match.group(2)

    meta: dict = {}
    for line in frontmatter_text.strip().split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if value.lower() in ("true", "false"):
            meta[key] = value.lower() == "true"
        elif key in _LIST_FIELDS:
            meta[key] = [item.strip() for item in value.split(",") if item.strip()]
        elif key == "workspace":
            meta[key] = _parse_workspace(value)
        else:
            # Attempt int coercion for numeric-only values.
            try:
                meta[key] = int(value)
            except ValueError:
                meta[key] = value

    return meta, content


def load_skill(path: str) -> SkillInfo:
    """Load a single SKILL.md file."""
    with open(path) as f:
        text = f.read()

    meta, content = _parse_frontmatter(text)
    return SkillInfo(
        name=meta.get("name", os.path.basename(os.path.dirname(path))),
        description=meta.get("description", ""),
        version=meta.get("version", "0.0.0"),
        author=meta.get("author", ""),
        always=meta.get("always", False),
        content=content.strip(),
        path=path,
        robot_type=meta.get("robot_type", ""),
        required_safety_tier=meta.get("required_safety_tier", "observe"),
        workspace=meta.get("workspace"),
        hardware_requirements=meta.get("hardware_requirements"),
        timeout_secs=meta.get("timeout_secs", 0),
        depends_on=meta.get("depends_on"),
    )


def load_skills(skills_dir: str) -> list[SkillInfo]:
    """
    Load all SKILL.md files from a directory tree.

    Searches for files named SKILL.md in immediate subdirectories:
      skills_dir/
        dora-moveit2/SKILL.md
        dora-mujoco/SKILL.md
    """
    skills = []
    if not os.path.isdir(skills_dir):
        return skills

    for entry in sorted(os.listdir(skills_dir)):
        skill_path = os.path.join(skills_dir, entry, "SKILL.md")
        if os.path.isfile(skill_path):
            skill = load_skill(skill_path)
            skills.append(skill)
            print(f"  [skill] Loaded '{skill.name}' v{skill.version} (always={skill.always})")

    return skills


def skills_to_prompt(skills: list[SkillInfo], always_only: bool = False) -> str:
    """Concatenate skill content into a system prompt section."""
    sections = []
    for skill in skills:
        if always_only and not skill.always:
            continue
        sections.append(f"## Skill: {skill.name}\n\n{skill.content}")
    return "\n\n---\n\n".join(sections)
