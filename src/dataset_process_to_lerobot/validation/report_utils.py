"""Shared helpers for machine-readable validation reports."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def walk_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_objects(child)


def episode_indices_from_report(report: dict[str, Any], fail_on: str) -> set[int]:
    """Extract episode indices from structured findings at or above a severity threshold."""
    threshold = SEVERITY_ORDER[fail_on.upper()]
    indices: set[int] = set()
    for item in walk_objects(report):
        episode_index = item.get("episode_index")
        severity = item.get("severity")
        if (
            isinstance(episode_index, int)
            and isinstance(severity, str)
            and severity.upper() in SEVERITY_ORDER
            and SEVERITY_ORDER[severity.upper()] >= threshold
        ):
            indices.add(episode_index)
    declared = report.get("deletable_episode_indices", [])
    if isinstance(declared, list):
        indices.update(index for index in declared if isinstance(index, int))
    return indices


def report_reaches_threshold(report: dict[str, Any], fail_on: str) -> bool:
    overall = report.get("overall_severity")
    return (
        isinstance(overall, str)
        and overall.upper() in SEVERITY_ORDER
        and SEVERITY_ORDER[overall.upper()] >= SEVERITY_ORDER[fail_on.upper()]
    )


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON report {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Validation report must contain a JSON object: {path}")
    return value


def write_json_atomic(value: dict[str, Any], path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def blockers_from_report(report: dict[str, Any]) -> Sequence[str]:
    blockers = report.get("non_episode_blockers", [])
    if not isinstance(blockers, list):
        return ()
    return tuple(str(blocker) for blocker in blockers)
