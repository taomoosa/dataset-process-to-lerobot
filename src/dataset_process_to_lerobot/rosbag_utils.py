"""Discover rosbag2 directories for conversion and archival workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RosbagLocation:
    path: Path
    collection_root: Path | None


def is_rosbag_directory(path: Path) -> bool:
    return path.is_dir() and (path / "metadata.yaml").is_file()


def discover_rosbags(
    explicit_bags: Sequence[Path] = (),
    bag_directories: Sequence[Path] = (),
    *,
    recursive: bool = False,
) -> tuple[RosbagLocation, ...]:
    """Resolve explicit bags and discover bags below repeatable collection directories."""
    discovered: list[RosbagLocation] = []
    seen: set[Path] = set()

    for supplied in explicit_bags:
        path = supplied.expanduser().resolve()
        if not is_rosbag_directory(path):
            raise ValueError(f"rosbag2 directory or metadata.yaml is missing: {path}")
        if path not in seen:
            discovered.append(RosbagLocation(path=path, collection_root=None))
            seen.add(path)

    for supplied_root in bag_directories:
        root = supplied_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"bag collection directory does not exist: {root}")
        if is_rosbag_directory(root):
            candidates = [root]
            collection_root = None
        elif recursive:
            candidates = sorted(
                metadata.parent for metadata in root.rglob("metadata.yaml") if metadata.is_file()
            )
            collection_root = root
        else:
            candidates = sorted(
                child for child in root.iterdir() if child.is_dir() and is_rosbag_directory(child)
            )
            collection_root = root
        for candidate in candidates:
            path = candidate.resolve()
            if path not in seen:
                discovered.append(RosbagLocation(path=path, collection_root=collection_root))
                seen.add(path)

    if not discovered:
        raise ValueError("no rosbag2 directories were supplied or discovered")
    return tuple(discovered)
