"""Run structural and temporal validation for a local LeRobotDataset V3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .configuration import (
    PROFILE_VERSION,
    add_validation_config_argument,
    dataset_argument_defaults,
    effective_dataset_config,
    parse_with_validation_profile,
)
from .doctor_evaluator import doctor_selection_report
from .evaluator_contract import CONTRACT_NAME, EXIT_BLOCKED, EXIT_CLEAN, EXIT_FINDINGS
from .history import append_history, history_entry
from .lerobot_video_check import main as video_check_main
from .report_utils import (
    SEVERITY_ORDER,
    episode_indices_from_report,
    read_json_object,
    report_reaches_threshold,
    write_json_atomic,
)


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


def _component_summary(
    name: str,
    return_code: int,
    report: dict[str, Any] | None,
    fail_on: str,
) -> tuple[dict[str, Any], set[int], list[str]]:
    if report is None:
        blocker = f"{name} did not produce a machine-readable report"
        return (
            {"name": name, "status": "error", "return_code": return_code},
            set(),
            [blocker],
        )
    indices = episode_indices_from_report(report, fail_on)
    reaches_threshold = report_reaches_threshold(report, fail_on)
    blockers: list[str] = []
    if reaches_threshold and not indices:
        blockers.append(f"{name} reached the {fail_on.upper()} threshold without episode indices")
    elif return_code != 0 and not reaches_threshold:
        blockers.append(f"{name} exited with status {return_code} without report findings")
    status = "blocked" if blockers else "findings" if indices else "pass"
    return (
        {
            "name": name,
            "status": status,
            "return_code": return_code,
            "overall_severity": report.get("overall_severity", "UNKNOWN"),
            "episode_indices": sorted(indices),
        },
        indices,
        blockers,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lerobot-doctor and temporal video validation."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument(
        "--history-file",
        type=Path,
        help=(
            "append dataset-linked validation history (default: REPORT_DIR/validation-history.json)"
        ),
    )
    add_validation_config_argument(parser)
    parser.add_argument("--doctor-command", default="lerobot-doctor")
    parser.set_defaults(skip_doctor=False)
    doctor_mode = parser.add_mutually_exclusive_group()
    doctor_mode.add_argument("--skip-doctor", action="store_true", default=argparse.SUPPRESS)
    doctor_mode.add_argument(
        "--run-doctor", action="store_false", dest="skip_doctor", default=argparse.SUPPRESS
    )
    parser.add_argument("--fail-on", choices=("warn", "fail"), default="fail")
    parser.add_argument("--features", help="comma-separated video feature keys")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--thumbnail-size", type=int, default=32)
    parser.add_argument("--duplicate-threshold", type=float, default=0.1)
    parser.add_argument("--freeze-min-seconds", type=float, default=1.0)
    parser.add_argument("--repeat-threshold", type=float, default=0.1)
    parser.add_argument("--repeat-min-cycles", type=int, default=3)
    parser.add_argument("--repeat-max-period-seconds", type=float, default=5.0)
    parser.add_argument("--jump-percentile", type=float, default=10.0)
    parser.add_argument("--jump-min-score", type=float, default=0.1)
    parser.add_argument("--artifact-block-threshold", type=float, default=20.0)
    parser.add_argument("--artifact-min-block-fraction", type=float, default=0.05)
    parser.add_argument("--artifact-max-duration-frames", type=int, default=3)
    parser.add_argument("--flat-frame-std-threshold", type=float, default=2.0)
    parser.add_argument("--temporal-discontinuity-threshold", type=float, default=5.0)
    parser.add_argument("--state-motion-support-threshold", type=float, default=0.02)
    parser.add_argument("--max-findings", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, _, config_source = parse_with_validation_profile(parser, argv, dataset_argument_defaults)
    dataset = args.dataset.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    summary_file = (
        args.summary_file.expanduser().resolve()
        if args.summary_file
        else report_dir / "validation-summary.json"
    )
    history_file = (
        args.history_file.expanduser().resolve()
        if args.history_file
        else report_dir / "validation-history.json"
    )
    if history_file == summary_file:
        print("History file and summary file must be different", file=sys.stderr)
        return EXIT_BLOCKED
    if not dataset.is_dir():
        print(f"Dataset directory does not exist: {dataset}", file=sys.stderr)
        return EXIT_BLOCKED

    report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    run_token = uuid.uuid4().hex
    components: list[dict[str, Any]] = []
    deletable_indices: set[int] = set()
    blockers: list[str] = []
    severities: list[str] = []

    if args.skip_doctor:
        components.append({"name": "lerobot-doctor", "status": "skipped", "return_code": 0})
    else:
        doctor_markdown = report_dir / "lerobot-doctor.md"
        temporary_markdown = report_dir / f".lerobot-doctor-{run_token}.md"
        command = build_doctor_command(
            args.doctor_command,
            dataset,
            temporary_markdown,
            args.fail_on,
        )
        doctor_report: dict[str, Any] | None = None
        doctor_status = EXIT_BLOCKED
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            doctor_status = completed.returncode
            if completed.stdout:
                print(completed.stdout, end="")
                try:
                    loaded = json.loads(completed.stdout)
                    if isinstance(loaded, dict):
                        doctor_report = loaded
                except json.JSONDecodeError:
                    pass
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            (report_dir / "lerobot-doctor.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (report_dir / "lerobot-doctor.stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
            if doctor_report is not None:
                write_json_atomic(doctor_report, report_dir / "lerobot-doctor.json")
            if temporary_markdown.exists():
                temporary_markdown.replace(doctor_markdown)
        except FileNotFoundError:
            print(
                f"Could not find {args.doctor_command!r}; install lerobot-doctor or use "
                "--doctor-command.",
                file=sys.stderr,
            )
        finally:
            if temporary_markdown.exists():
                temporary_markdown.unlink()
        selection_report = (
            doctor_selection_report(doctor_report, args.fail_on)
            if doctor_report is not None
            else None
        )
        component, indices, component_blockers = _component_summary(
            "lerobot-doctor", doctor_status, selection_report, args.fail_on
        )
        components.append(component)
        deletable_indices.update(indices)
        blockers.extend(component_blockers)
        if doctor_report is not None:
            severity = doctor_report.get("overall_severity")
            if isinstance(severity, str) and severity in SEVERITY_ORDER:
                severities.append(severity)

    video_markdown = report_dir / "lerobot-video-check.md"
    video_json = report_dir / "lerobot-video-check.json"
    temporary_video_markdown = report_dir / f".lerobot-video-check-{run_token}.md"
    temporary_video_json = report_dir / f".lerobot-video-check-{run_token}.json"
    video_args = [
        str(dataset),
        "--ci",
        "--fail-on",
        args.fail_on,
        "--markdown",
        str(temporary_video_markdown),
        "--json-file",
        str(temporary_video_json),
        "--thumbnail-size",
        str(args.thumbnail_size),
        "--duplicate-threshold",
        str(args.duplicate_threshold),
        "--freeze-min-seconds",
        str(args.freeze_min_seconds),
        "--repeat-threshold",
        str(args.repeat_threshold),
        "--repeat-min-cycles",
        str(args.repeat_min_cycles),
        "--repeat-max-period-seconds",
        str(args.repeat_max_period_seconds),
        "--jump-percentile",
        str(args.jump_percentile),
        "--jump-min-score",
        str(args.jump_min_score),
        "--artifact-block-threshold",
        str(args.artifact_block_threshold),
        "--artifact-min-block-fraction",
        str(args.artifact_min_block_fraction),
        "--artifact-max-duration-frames",
        str(args.artifact_max_duration_frames),
        "--flat-frame-std-threshold",
        str(args.flat_frame_std_threshold),
        "--temporal-discontinuity-threshold",
        str(args.temporal_discontinuity_threshold),
        "--state-motion-support-threshold",
        str(args.state_motion_support_threshold),
        "--max-findings",
        str(args.max_findings),
    ]
    if args.features:
        video_args.extend(("--features", args.features))
    if args.max_episodes is not None:
        video_args.extend(("--max-episodes", str(args.max_episodes)))
    video_status = video_check_main(video_args)
    video_report: dict[str, Any] | None = None
    try:
        if temporary_video_json.exists():
            video_report = read_json_object(temporary_video_json)
            temporary_video_json.replace(video_json)
        if temporary_video_markdown.exists():
            temporary_video_markdown.replace(video_markdown)
    finally:
        if temporary_video_json.exists():
            temporary_video_json.unlink()
        if temporary_video_markdown.exists():
            temporary_video_markdown.unlink()
    component, indices, component_blockers = _component_summary(
        "lerobot-video-check", video_status, video_report, args.fail_on
    )
    components.append(component)
    deletable_indices.update(indices)
    blockers.extend(component_blockers)
    if video_report is not None:
        severity = video_report.get("overall_severity")
        if isinstance(severity, str) and severity in SEVERITY_ORDER:
            severities.append(severity)

    overall_severity = max(
        severities,
        key=lambda severity: SEVERITY_ORDER[severity],
        default="FAIL" if blockers else "PASS",
    )
    status = "blocked" if blockers else "findings" if deletable_indices else "pass"
    effective_config = {
        "version": PROFILE_VERSION,
        "dataset": effective_dataset_config(args),
    }
    summary = {
        "version": "1.0",
        "contract": CONTRACT_NAME,
        "evaluator": "combined-validation",
        "tool": "validate-lerobot-dataset",
        "dataset_path": str(dataset),
        "fail_on": args.fail_on.upper(),
        "status": status,
        "overall_severity": overall_severity,
        "deletable_episode_indices": sorted(deletable_indices),
        "findings": [
            {
                "episode_index": index,
                "severity": args.fail_on.upper(),
                "kind": "validation_failure",
            }
            for index in sorted(deletable_indices)
        ],
        "non_episode_blockers": blockers,
        "validation_config": effective_config,
        "validation_config_source": (str(config_source) if config_source is not None else None),
        "components": components,
        "artifacts": {
            "report_directory": str(report_dir),
            "history": str(history_file),
        },
        "evaluator_return_code": max(
            (int(component["return_code"]) for component in components), default=0
        ),
    }
    write_json_atomic(summary, summary_file)
    try:
        append_history(
            history_file,
            history_entry(
                "validation",
                dataset,
                result={
                    "status": summary["status"],
                    "overall_severity": summary["overall_severity"],
                    "fail_on": summary["fail_on"],
                    "deletable_episode_indices": summary["deletable_episode_indices"],
                    "non_episode_blockers": summary["non_episode_blockers"],
                    "components": summary["components"],
                },
                config=effective_config,
                config_source=config_source,
                details={"summary_file": str(summary_file)},
            ),
        )
    except (OSError, ValueError) as error:
        blockers.append(f"Could not append validation history: {error}")
        summary["status"] = "blocked"
        summary["overall_severity"] = "FAIL"
        summary["non_episode_blockers"] = blockers
        write_json_atomic(summary, summary_file)
        print(blockers[-1], file=sys.stderr)
        return EXIT_BLOCKED
    print(f"Wrote validation summary to {summary_file}", file=sys.stderr)
    if blockers:
        return EXIT_BLOCKED
    if deletable_indices:
        return EXIT_FINDINGS
    return EXIT_CLEAN


if __name__ == "__main__":
    raise SystemExit(main())
