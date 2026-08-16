"""Create a new LeRobotDataset without episodes flagged by validation reports."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .history import append_history, dataset_identity, file_reference, history_entry
from .report_utils import (
    blockers_from_report,
    episode_indices_from_report,
    read_json_object,
    report_reaches_threshold,
    write_json_atomic,
)

REPO_ID_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")


def load_report_episode_indices(paths: Sequence[Path], fail_on: str) -> set[int]:
    indices: set[int] = set()
    for path in paths:
        report = read_json_object(path)
        blockers = blockers_from_report(report)
        if blockers:
            raise ValueError(f"Report {path} contains non-episode blockers: {'; '.join(blockers)}")
        report_indices = episode_indices_from_report(report, fail_on)
        if not report_indices and report_reaches_threshold(report, fail_on):
            raise ValueError(
                f"Report {path} reaches the {fail_on.upper()} threshold but has no structured "
                "episode indices; use --episode explicitly"
            )
        indices.update(report_indices)
    return indices


def default_repo_id(path: Path) -> str:
    token = REPO_ID_TOKEN.sub("-", path.name).strip("-.") or "dataset"
    return f"local/{token}"


def _total_episodes(dataset: Path) -> int:
    info_path = dataset / "meta" / "info.json"
    try:
        info = read_json_object(info_path)
        total = int(info["total_episodes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Could not read total_episodes from {info_path}: {error}") from error
    if total <= 0:
        raise ValueError("Dataset must contain at least one episode")
    return total


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a LeRobotDataset while deleting episodes flagged by JSON reports."
    )
    parser.add_argument("dataset", type=Path, help="source LeRobotDataset V3 directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", action="append", type=Path, default=[])
    parser.add_argument("--episode", action="append", type=int, default=[])
    parser.add_argument("--fail-on", choices=("warn", "fail"), default="fail")
    parser.add_argument("--source-repo-id")
    parser.add_argument("--output-repo-id")
    parser.add_argument(
        "--result-file",
        type=Path,
        help="always write the selected effective dataset and removed indices as JSON",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        help=("append selection history; defaults beside --result-file when that option is used"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _record_selection(
    args: argparse.Namespace,
    dataset: Path,
    result: dict[str, Any],
) -> None:
    if args.result_file:
        write_json_atomic(result, args.result_file)
    history_file = (
        args.history_file.expanduser().resolve()
        if args.history_file
        else args.result_file.expanduser().resolve().parent / "validation-history.json"
        if args.result_file
        else None
    )
    if history_file is None:
        return
    if args.result_file and history_file == args.result_file.expanduser().resolve():
        raise ValueError("History file and result file must be different")
    effective = Path(str(result["effective_dataset"])).resolve()
    append_history(
        history_file,
        history_entry(
            "episode_selection",
            dataset,
            result={
                "status": result["status"],
                "removed_episode_indices": result["removed_episode_indices"],
                "would_remove_episode_indices": result.get("would_remove_episode_indices", []),
            },
            config={
                "fail_on": args.fail_on,
                "explicit_episode_indices": sorted(set(args.episode)),
            },
            details={
                "reports": [file_reference(path) for path in args.report],
                "effective_dataset": dataset_identity(effective),
                "result_file": (
                    str(args.result_file.expanduser().resolve()) if args.result_file else None
                ),
            },
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        if not dataset_path.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset_path}")
        indices = load_report_episode_indices(args.report, args.fail_on)
        indices.update(args.episode)
        total_episodes = _total_episodes(dataset_path)
        invalid = {index for index in indices if index < 0 or index >= total_episodes}
        if invalid:
            raise ValueError(f"Episode indices are outside the dataset: {sorted(invalid)}")
        ordered = sorted(indices)
        if not ordered:
            print("No episodes matched the requested severity; no dataset was created.")
            _record_selection(
                args,
                dataset_path,
                {
                    "status": "no_change",
                    "source_dataset": str(dataset_path),
                    "effective_dataset": str(dataset_path),
                    "output_dataset": None,
                    "removed_episode_indices": [],
                },
            )
            return 0
        if len(ordered) == total_episodes:
            raise ValueError("Refusing to delete every episode")
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        print(f"Episodes selected for deletion: {ordered}")
        if args.dry_run:
            _record_selection(
                args,
                dataset_path,
                {
                    "status": "dry_run",
                    "source_dataset": str(dataset_path),
                    "effective_dataset": str(dataset_path),
                    "output_dataset": None,
                    "removed_episode_indices": [],
                    "would_remove_episode_indices": ordered,
                    "dry_run": True,
                },
            )
            return 0

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from lerobot.datasets.dataset_tools import delete_episodes
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        source_repo_id = args.source_repo_id or default_repo_id(dataset_path)
        output_repo_id = args.output_repo_id or default_repo_id(output_dir)
        dataset = LeRobotDataset(source_repo_id, root=dataset_path)
        result = delete_episodes(
            dataset,
            episode_indices=ordered,
            output_dir=output_dir,
            repo_id=output_repo_id,
        )
        print(
            f"Created {output_dir} with {result.meta.total_episodes} episodes and "
            f"{result.meta.total_frames} frames."
        )
        _record_selection(
            args,
            dataset_path,
            {
                "status": "filtered",
                "source_dataset": str(dataset_path),
                "effective_dataset": str(output_dir),
                "output_dataset": str(output_dir),
                "removed_episode_indices": ordered,
            },
        )
        return 0
    except Exception as error:
        print(f"Could not remove failed episodes: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
