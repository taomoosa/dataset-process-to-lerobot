"""ROS-independent validation and signal generation shared by the mock nodes."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

AXIS_COUNT = 7
DEFAULT_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)
_ENTITY_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def require_positive(value: float, parameter_name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"parameter '{parameter_name}' must be a positive finite number")
    return value


def validate_entity_ids(values: Sequence[str], parameter_name: str) -> tuple[str, ...]:
    entity_ids = tuple(str(value) for value in values)
    if not entity_ids:
        raise ValueError(f"parameter '{parameter_name}' must contain at least one ID")
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError(f"parameter '{parameter_name}' must not contain duplicate IDs")
    invalid = [value for value in entity_ids if not _ENTITY_ID_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(
            f"parameter '{parameter_name}' contains invalid ROS topic tokens: {invalid}; "
            "use letters, digits, and underscores, and do not start with a digit"
        )
    return entity_ids


def validate_axis_vector(values: Sequence[float], parameter_name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if len(vector) != AXIS_COUNT:
        raise ValueError(f"parameter '{parameter_name}' must contain exactly {AXIS_COUNT} values")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"parameter '{parameter_name}' values must all be finite")
    return vector


def validate_joint_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if len(names) != AXIS_COUNT:
        raise ValueError(f"parameter 'joint_names' must contain exactly {AXIS_COUNT} names")
    if any(not value for value in names) or len(set(names)) != len(names):
        raise ValueError("parameter 'joint_names' must contain unique, non-empty names")
    return names


def join_topic(root: str, entity_id: str, suffix: str) -> str:
    root = str(root).strip()
    if not root:
        raise ValueError("topic root must not be empty")
    absolute = root.startswith("/")
    parts = [part for part in root.strip("/").split("/") if part]
    parts.extend((entity_id, suffix.strip("/")))
    topic = "/".join(parts)
    return f"/{topic}" if absolute else topic


def sampled_sine(
    elapsed_seconds: float,
    entity_index: int,
    amplitudes: Sequence[float],
    offsets: Sequence[float],
    frequency_hz: float,
) -> list[float]:
    """Generate deterministic, distinguishable seven-axis values for one entity."""

    base_phase = math.tau * frequency_hz * max(0.0, elapsed_seconds)
    entity_phase = entity_index * 0.61
    return [
        offset + amplitude * math.sin(base_phase + entity_phase + axis * 0.37)
        for axis, (amplitude, offset) in enumerate(zip(amplitudes, offsets, strict=True))
    ]
