"""Path and parameter helpers for rosbag2 recording sessions."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

_SESSION_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_topics(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    topics = tuple(str(value) for value in values)
    if not topics:
        raise ValueError("parameter 'topics' must contain at least one topic")
    if len(set(topics)) != len(topics):
        raise ValueError("parameter 'topics' must not contain duplicates")
    invalid = [topic for topic in topics if not topic.startswith("/") or topic == "/"]
    if invalid:
        raise ValueError(f"all recorded topics must be absolute ROS topic names: {invalid}")
    return topics


def validate_session_prefix(prefix: str) -> str:
    prefix = str(prefix)
    if not _SESSION_PREFIX_PATTERN.fullmatch(prefix) or prefix in {".", ".."}:
        raise ValueError(
            "parameter 'session_prefix' must use only letters, digits, '.', '_', and '-', "
            "must start with a letter or digit, and must not be '.' or '..'"
        )
    return prefix


def next_session_path(
    output_directory: str | Path,
    session_prefix: str,
    session_number: int,
    now: datetime | None = None,
) -> Path:
    base = Path(output_directory).expanduser().resolve()
    prefix = validate_session_prefix(session_prefix)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{prefix}_{timestamp}_{session_number:03d}"
    candidate = base / stem
    collision = 1
    while candidate.exists():
        candidate = base / f"{stem}_{collision:02d}"
        collision += 1
    return candidate


def discard_session_directory(output_directory: str | Path, session_path: str | Path) -> None:
    """Delete exactly one session directory, refusing traversal and symlink escapes."""

    base = Path(output_directory).expanduser().resolve()
    candidate = Path(session_path)
    if not candidate.exists():
        raise FileNotFoundError(f"recording session does not exist: {candidate}")
    resolved = candidate.resolve()
    if resolved.parent != base or not resolved.is_dir():
        raise ValueError(f"refusing to discard unsafe recording path: {candidate}")
    shutil.rmtree(resolved)
