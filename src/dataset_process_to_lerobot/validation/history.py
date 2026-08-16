"""Append dataset-linked validation and episode-selection provenance."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .report_utils import read_json_object, write_json_atomic

HISTORY_VERSION = "1.0"
HISTORY_TOOL = "dataset-validation-history"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(dataset: Path) -> dict[str, Any]:
    """Identify a dataset without modifying its LeRobot metadata tree."""
    root = dataset.expanduser().resolve()
    identity: dict[str, Any] = {"path": str(root)}
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        identity["metadata_status"] = "missing"
        return identity
    try:
        info = read_json_object(info_path)
    except ValueError as error:
        identity["metadata_status"] = "invalid"
        identity["metadata_error"] = str(error)
        return identity
    identity.update(
        {
            "metadata_status": "available",
            "info_sha256": _sha256(info_path),
            "codebase_version": info.get("codebase_version"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "fps": info.get("fps"),
        }
    )
    return identity


def file_reference(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    reference: dict[str, Any] = {"path": str(resolved)}
    if resolved.is_file():
        reference.update({"sha256": _sha256(resolved), "size": resolved.stat().st_size})
    else:
        reference["status"] = "missing"
    return reference


def history_entry(
    operation: str,
    dataset: Path,
    *,
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
    config_source: Path | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "dataset": dataset_identity(dataset),
        "result": result,
    }
    if config is not None:
        entry["config"] = config
    if config_source is not None:
        entry["config_source"] = file_reference(config_source)
    if details:
        entry["details"] = details
    return entry


def append_history(path: Path, entry: dict[str, Any]) -> None:
    """Atomically append one immutable entry to a JSON history document."""
    destination = path.expanduser().resolve()
    if destination.exists():
        document = read_json_object(destination)
        if document.get("version") != HISTORY_VERSION or document.get("tool") != HISTORY_TOOL:
            raise ValueError(f"Unsupported validation history document: {destination}")
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Validation history entries must be an array: {destination}")
    else:
        document = {"version": HISTORY_VERSION, "tool": HISTORY_TOOL, "entries": []}
        entries = document["entries"]
    entries.append(entry)
    write_json_atomic(document, destination)
