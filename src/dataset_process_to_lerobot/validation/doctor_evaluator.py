"""Adapt lerobot-doctor to the dataset evaluator contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .evaluator_contract import (
    EXIT_BLOCKED,
    add_common_evaluator_arguments,
    exit_code_for_result,
    make_evaluation_result,
    result_path,
)
from .report_utils import write_json_atomic

DOCTOR_EPISODE_REFERENCE = re.compile(r"\bepisode\s+(\d+)\b", re.IGNORECASE)
DOCTOR_EPISODE_LIST = re.compile(r"\bepisode\(s\).*:\s*\[([0-9,\s]+)\]\s*$", re.IGNORECASE)


def build_doctor_command(
    executable: str,
    dataset: Path,
    markdown_report: Path,
    fail_on: str,
) -> list[str]:
    return [
        executable,
        "check",
        str(dataset),
        "--ci",
        "--fail-on",
        fail_on,
        "--markdown",
        str(markdown_report),
    ]


def doctor_selection_report(raw_report: dict[str, Any], fail_on: str) -> dict[str, Any]:
    """Add structured selections without changing the saved doctor report."""
    threshold = {"PASS": 0, "WARN": 1, "FAIL": 2}[fail_on.upper()]
    indices: set[int] = set()
    checks = raw_report.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            messages = check.get("messages", [])
            if not isinstance(messages, list):
                continue
            for item in messages:
                if not isinstance(item, dict):
                    continue
                severity = str(item.get("severity", "")).upper()
                if severity not in {"PASS", "WARN", "FAIL"}:
                    continue
                if {"PASS": 0, "WARN": 1, "FAIL": 2}[severity] < threshold:
                    continue
                message = item.get("message")
                if not isinstance(message, str):
                    continue
                indices.update(
                    int(match.group(1)) for match in DOCTOR_EPISODE_REFERENCE.finditer(message)
                )
                episode_list = DOCTOR_EPISODE_LIST.search(message)
                if episode_list is not None:
                    indices.update(
                        int(value.strip())
                        for value in episode_list.group(1).split(",")
                        if value.strip()
                    )
    selection_report = dict(raw_report)
    selection_report["deletable_episode_indices"] = sorted(indices)
    return selection_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-lerobot-doctor",
        description="Run lerobot-doctor and emit a normalized episode-selection result.",
    )
    add_common_evaluator_arguments(parser)
    parser.add_argument("--doctor-command", default="lerobot-doctor")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    output_path = result_path(report_dir, args.result_file)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_json = report_dir / "lerobot-doctor.json"
    markdown = report_dir / "lerobot-doctor.md"
    stdout_log = report_dir / "lerobot-doctor.stdout.log"
    stderr_log = report_dir / "lerobot-doctor.stderr.log"
    temporary_markdown = report_dir / f".lerobot-doctor-{uuid.uuid4().hex}.md"
    raw_report: dict[str, Any] | None = None
    return_code = EXIT_BLOCKED
    error: str | None = None

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        if not dataset.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset}")
        command = build_doctor_command(
            args.doctor_command,
            dataset,
            temporary_markdown,
            args.fail_on,
        )
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        return_code = completed.returncode
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")
        if completed.stdout:
            print(completed.stdout, end="")
            try:
                loaded = json.loads(completed.stdout)
                if isinstance(loaded, dict):
                    raw_report = loaded
            except json.JSONDecodeError:
                pass
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if raw_report is not None:
            write_json_atomic(raw_report, raw_json)
        if temporary_markdown.exists():
            temporary_markdown.replace(markdown)
    except (FileNotFoundError, OSError, ValueError) as caught:
        error = str(caught)
        print(f"Could not run lerobot-doctor: {caught}", file=sys.stderr)
    finally:
        if temporary_markdown.exists():
            temporary_markdown.unlink()

    artifacts = {
        name: str(path)
        for name, path in {
            "json": raw_json,
            "markdown": markdown,
            "stdout": stdout_log,
            "stderr": stderr_log,
        }.items()
        if path.exists()
    }
    result = make_evaluation_result(
        evaluator="lerobot-doctor",
        dataset=dataset,
        fail_on=args.fail_on,
        raw_report=(
            doctor_selection_report(raw_report, args.fail_on) if raw_report is not None else None
        ),
        return_code=return_code,
        artifacts=artifacts,
        error=error,
    )
    write_json_atomic(result, output_path)
    print(f"Wrote evaluation result to {output_path}", file=sys.stderr)
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
