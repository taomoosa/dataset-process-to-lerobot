"""Copy or move rosbag2 directories to slower archival storage with verification."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dataset_process_to_lerobot.rosbag_utils import RosbagLocation, discover_rosbags
from dataset_process_to_lerobot.validation.report_utils import write_json_atomic


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def file_inventory(root: Path, verification: str) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symbolic links are not supported in rosbag archives: {path}")
        if not path.is_file():
            continue
        details: dict[str, Any] = {"size": path.stat().st_size}
        if verification == "sha256":
            details["sha256"] = _digest(path)
        inventory[path.relative_to(root).as_posix()] = details
    return inventory


def _destination_for(location: RosbagLocation, archive_root: Path) -> Path:
    if location.collection_root is None:
        return archive_root / location.path.name
    relative = location.path.relative_to(location.collection_root)
    return archive_root / location.collection_root.name / relative


def _validate_archive_boundaries(locations: Sequence[RosbagLocation], archive_root: Path) -> None:
    destinations: set[Path] = set()
    for location in locations:
        source = location.path
        destination = _destination_for(location, archive_root)
        if archive_root == source or archive_root.is_relative_to(source):
            raise ValueError(f"archive directory must not be inside source bag: {source}")
        if source.is_relative_to(archive_root):
            raise ValueError(f"source bag is already inside the archive directory: {source}")
        if destination in destinations:
            raise ValueError(f"multiple source bags map to the same archive path: {destination}")
        destinations.add(destination)


def archive_rosbags(
    locations: Sequence[RosbagLocation],
    archive_directory: Path,
    *,
    mode: str = "copy",
    verification: str = "sha256",
    existing: str = "error",
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"copy", "move"}:
        raise ValueError("archive mode must be copy or move")
    if verification not in {"size", "sha256"}:
        raise ValueError("verification must be size or sha256")
    if existing not in {"error", "verify"}:
        raise ValueError("existing policy must be error or verify")
    if not locations:
        raise ValueError("at least one rosbag2 directory is required")

    archive_root = archive_directory.expanduser().resolve()
    _validate_archive_boundaries(locations, archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    manifest = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else archive_root / "archive-manifest.json"
    )
    source_sizes = {
        location.path: sum(
            path.stat().st_size for path in location.path.rglob("*") if path.is_file()
        )
        for location in locations
    }
    required_bytes = sum(
        size
        for location, size in ((item, source_sizes[item.path]) for item in locations)
        if not _destination_for(location, archive_root).exists()
    )
    free_bytes = shutil.disk_usage(archive_root).free
    if free_bytes < required_bytes:
        raise OSError(
            f"archive filesystem has {free_bytes} free bytes, but {required_bytes} are required"
        )

    result: dict[str, Any] = {
        "version": "1.0",
        "tool": "archive-rosbags",
        "archive_directory": str(archive_root),
        "mode": mode,
        "verification": verification,
        "bags": [],
    }
    for location in locations:
        source = location.path
        destination = _destination_for(location, archive_root)
        record: dict[str, Any] = {
            "source": str(source),
            "destination": str(destination),
            "bytes": source_sizes[source],
            "status": "pending",
        }
        result["bags"].append(record)
        write_json_atomic(result, manifest)
        source_inventory = file_inventory(source, verification)
        if destination.exists():
            if existing != "verify":
                raise FileExistsError(f"archive destination already exists: {destination}")
            if file_inventory(destination, verification) != source_inventory:
                raise ValueError(f"existing archive does not match source: {destination}")
            record["status"] = "verified_existing"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
            try:
                shutil.copytree(source, staging, copy_function=shutil.copy2)
                if file_inventory(staging, verification) != source_inventory:
                    raise ValueError(f"archive verification failed for {source}")
                staging.rename(destination)
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                raise
            record["status"] = "copied"

        if mode == "move":
            shutil.rmtree(source)
            record["status"] = "moved"
        write_json_atomic(result, manifest)
    result["status"] = "complete"
    write_json_atomic(result, manifest)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy or move rosbag2 directories to archival storage after verification."
    )
    parser.add_argument("bags", nargs="*", type=Path, help="explicit rosbag2 directories")
    parser.add_argument("--bag-dir", action="append", type=Path, default=[])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("copy", "move"), default="copy")
    parser.add_argument("--verify", choices=("size", "sha256"), default="sha256")
    parser.add_argument(
        "--existing",
        choices=("error", "verify"),
        default="error",
        help="verify matching existing archives when resuming instead of failing",
    )
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        locations = discover_rosbags(args.bags, args.bag_dir, recursive=args.recursive)
        result = archive_rosbags(
            locations,
            args.archive_dir,
            mode=args.mode,
            verification=args.verify,
            existing=args.existing,
            manifest_path=args.manifest,
        )
        print(f"Archived {len(result['bags'])} rosbag2 directories to {args.archive_dir}")
        return 0
    except Exception as error:
        print(f"Could not archive rosbag2 data: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
