"""Run conversion, configurable evaluation/filter stages, and rosbag archival."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_process_to_lerobot.conversion.rosbag_to_lerobot import main as conversion_main
from dataset_process_to_lerobot.recording.archive_rosbags import archive_rosbags
from dataset_process_to_lerobot.rosbag_utils import discover_rosbags
from dataset_process_to_lerobot.validation.configuration import (
    PROFILE_VERSION,
    add_validation_config_argument,
    effective_dataset_config,
    effective_input_config,
    parse_with_validation_profile,
    workflow_argument_defaults,
)
from dataset_process_to_lerobot.validation.evaluator_contract import (
    EXIT_BLOCKED,
    EXIT_CLEAN,
    EXIT_FINDINGS,
    exit_code_for_result,
    validate_evaluation_result,
)
from dataset_process_to_lerobot.validation.history import append_history, history_entry
from dataset_process_to_lerobot.validation.remove_failed_episodes import main as removal_main
from dataset_process_to_lerobot.validation.report_utils import read_json_object, write_json_atomic

STAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RESERVED_EVALUATOR_OPTIONS = ("--report-dir", "--result-file", "--fail-on")


@dataclass(frozen=True)
class EvaluationStage:
    """A command implementing the common evaluator CLI and result contract."""

    name: str
    command: tuple[str, ...]


def _repeat(arguments: list[str], option: str, values: Sequence[str]) -> None:
    for value in values:
        arguments.extend((option, value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert rosbag2 batches, evaluate and filter the dataset through configurable "
            "stages, then archive raw bags."
        )
    )
    parser.add_argument("--bag-dir", action="append", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--clean-dataset-dir", type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument(
        "--history-file",
        type=Path,
        help=(
            "append dataset validation and selection history "
            "(default: REPORT_DIR/validation-history.json)"
        ),
    )
    add_validation_config_argument(parser)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--archive-mode", choices=("copy", "move"), default="copy")
    parser.add_argument("--archive-verify", choices=("size", "sha256"), default="sha256")
    parser.add_argument("--archive-existing", choices=("error", "verify"), default="error")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--clean-repo-id")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-map", action="append", default=[])
    parser.add_argument("--camera-topic", action="append", default=[])
    parser.add_argument("--action-topic", action="append", default=[])
    parser.add_argument("--state-topic", action="append", default=[])
    parser.add_argument("--robot-type", default="mock_7axis")
    parser.add_argument("--video-codec", default="libsvtav1")
    parser.add_argument(
        "--input-validation", choices=("off", "warn", "fail", "drop"), default="drop"
    )
    parser.add_argument(
        "--input-validation-drop-on",
        choices=("warn", "fail"),
        default="fail",
        help="minimum input finding severity rejected by drop policy",
    )
    parser.add_argument("--source-gap-min-seconds", type=float, default=1.0)
    parser.add_argument("--source-gap-factor", type=float, default=5.0)
    parser.add_argument("--source-drop-factor", type=float, default=2.5)
    parser.add_argument("--minimum-camera-source-frames", type=int, default=2)
    parser.add_argument("--duplicate-min-source-frames", type=int, default=2)
    parser.add_argument("--state-motion-threshold", type=float, default=1e-4)
    parser.add_argument("--max-input-findings", type=int, default=100)
    parser.add_argument(
        "--evaluation-stage",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help=(
            "replace the default doctor/video stages with a repeatable evaluator command; "
            "the workflow appends the common dataset, report, result, and fail-on arguments"
        ),
    )
    parser.add_argument("--doctor-command", default="lerobot-doctor")
    parser.set_defaults(skip_doctor=False)
    doctor_mode = parser.add_mutually_exclusive_group()
    doctor_mode.add_argument("--skip-doctor", action="store_true", default=argparse.SUPPRESS)
    doctor_mode.add_argument(
        "--run-doctor", action="store_false", dest="skip_doctor", default=argparse.SUPPRESS
    )
    parser.add_argument("--fail-on", choices=("warn", "fail"), default="fail")
    parser.add_argument("--features")
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
    parser.add_argument("--max-video-findings", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_evaluation_stage(value: str) -> EvaluationStage:
    """Parse NAME=COMMAND without invoking a shell."""
    if "=" not in value:
        raise ValueError("evaluation stage must use NAME=COMMAND")
    name, command_text = value.split("=", 1)
    name = name.strip()
    if not STAGE_NAME.fullmatch(name):
        raise ValueError(f"invalid evaluation stage name: {name!r}")
    try:
        command = tuple(shlex.split(command_text))
    except ValueError as error:
        message = f"could not parse command for evaluation stage {name!r}: {error}"
        raise ValueError(message) from error
    if not command:
        raise ValueError(f"evaluation stage {name!r} has an empty command")
    reserved = [
        token
        for token in command
        if any(
            token == option or token.startswith(f"{option}=")
            for option in RESERVED_EVALUATOR_OPTIONS
        )
    ]
    if reserved:
        raise ValueError(
            f"evaluation stage {name!r} sets workflow-owned options: {', '.join(reserved)}"
        )
    return EvaluationStage(name, command)


def _default_evaluation_stages(args: argparse.Namespace) -> list[EvaluationStage]:
    stages: list[EvaluationStage] = []
    if not args.skip_doctor:
        stages.append(
            EvaluationStage(
                "doctor",
                (
                    sys.executable,
                    "-m",
                    "dataset_process_to_lerobot.validation.doctor_evaluator",
                    "--doctor-command",
                    args.doctor_command,
                ),
            )
        )
    video_command = [
        sys.executable,
        "-m",
        "dataset_process_to_lerobot.validation.video_evaluator",
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
        str(args.max_video_findings),
    ]
    if args.features:
        video_command.extend(("--features", args.features))
    if args.max_episodes is not None:
        video_command.extend(("--max-episodes", str(args.max_episodes)))
    stages.append(EvaluationStage("video", tuple(video_command)))
    return stages


def _evaluation_stages(args: argparse.Namespace) -> list[EvaluationStage]:
    if args.evaluation_stage:
        stages = [parse_evaluation_stage(value) for value in args.evaluation_stage]
    else:
        stages = _default_evaluation_stages(args)
    duplicate_names = {
        stage.name for stage in stages if sum(s.name == stage.name for s in stages) > 1
    }
    if duplicate_names:
        raise ValueError(f"evaluation stage names must be unique: {sorted(duplicate_names)}")
    if not stages:
        raise ValueError("at least one evaluation stage is required")
    return stages


def run_evaluation_stage(
    stage: EvaluationStage,
    dataset: Path,
    report_dir: Path,
    fail_on: str,
) -> tuple[int, dict[str, Any]]:
    """Execute one evaluator and verify its normalized result before trusting it."""
    result_file = report_dir / "evaluation-result.json"
    temporary_result = report_dir / f".evaluation-result-{uuid.uuid4().hex}.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *stage.command,
        str(dataset),
        "--report-dir",
        str(report_dir),
        "--result-file",
        str(temporary_result),
        "--fail-on",
        fail_on,
    ]
    try:
        completed = subprocess.run(command, check=False)
    except OSError as error:
        raise ValueError(f"could not execute evaluator {stage.name!r}: {error}") from error
    try:
        if not temporary_result.is_file():
            raise ValueError(
                f"evaluator {stage.name!r} did not write its required result: {temporary_result}"
            )
        result = read_json_object(temporary_result)
        validate_evaluation_result(result, dataset=dataset, fail_on=fail_on)
        expected = exit_code_for_result(result)
        if completed.returncode != expected:
            raise ValueError(
                f"evaluator {stage.name!r} returned {completed.returncode}, but its result "
                f"requires {expected}"
            )
        temporary_result.replace(result_file)
        return completed.returncode, result
    finally:
        if temporary_result.exists():
            temporary_result.unlink()


def _conversion_arguments(
    args: argparse.Namespace,
    dataset: Path,
    report_dir: Path,
) -> list[str]:
    arguments = [
        "--output-dir",
        str(dataset),
        "--repo-id",
        args.repo_id,
        "--fps",
        str(args.fps),
        "--task",
        args.task,
        "--robot-type",
        args.robot_type,
        "--video-codec",
        args.video_codec,
        "--input-validation",
        args.input_validation,
        "--input-validation-drop-on",
        args.input_validation_drop_on,
        "--input-validation-report",
        str(report_dir / "input-validation.json"),
        "--conversion-manifest",
        str(report_dir / "conversion-manifest.json"),
        "--source-gap-min-seconds",
        str(args.source_gap_min_seconds),
        "--source-gap-factor",
        str(args.source_gap_factor),
        "--source-drop-factor",
        str(args.source_drop_factor),
        "--minimum-camera-source-frames",
        str(args.minimum_camera_source_frames),
        "--duplicate-min-source-frames",
        str(args.duplicate_min_source_frames),
        "--state-motion-threshold",
        str(args.state_motion_threshold),
        "--max-input-findings",
        str(args.max_input_findings),
    ]
    if args.validation_config:
        arguments.extend(("--validation-config", str(args.validation_config)))
    for directory in args.bag_dir:
        arguments.extend(("--bag-dir", str(directory)))
    if args.recursive:
        arguments.append("--recursive")
    _repeat(arguments, "--task-map", args.task_map)
    _repeat(arguments, "--camera-topic", args.camera_topic)
    _repeat(arguments, "--action-topic", args.action_topic)
    _repeat(arguments, "--state-topic", args.state_topic)
    return arguments


def _write_stage(
    summary: dict[str, Any],
    summary_path: Path,
    name: str,
    status: str,
    **details: Any,
) -> None:
    summary["stages"].append({"name": name, "status": status, **details})
    write_json_atomic(summary, summary_path)


def _filtered_destination(base: Path, filter_number: int, stage_name: str) -> Path:
    if filter_number == 1:
        return base
    return base.with_name(f"{base.name}_{filter_number:02d}_{stage_name}")


def _filtered_repo_id(base: str, filter_number: int, stage_name: str) -> str:
    if filter_number == 1:
        return base
    return f"{base}_{filter_number:02d}_{stage_name}"


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _validate_workflow_paths(
    bag_paths: Sequence[Path],
    *,
    dataset: Path,
    clean_dataset: Path,
    report_dir: Path,
    archive_dir: Path,
) -> None:
    """Reject layouts where archival or output creation can consume another artifact."""
    outputs = {
        "dataset directory": dataset,
        "clean dataset directory": clean_dataset,
        "report directory": report_dir,
        "archive directory": archive_dir,
    }
    output_items = tuple(outputs.items())
    for index, (first_name, first_path) in enumerate(output_items):
        for second_name, second_path in output_items[index + 1 :]:
            if _paths_overlap(first_path, second_path):
                raise ValueError(
                    f"{first_name} and {second_name} must not overlap: {first_path} ; {second_path}"
                )
    for bag_path in bag_paths:
        for output_name, output_path in output_items:
            if _paths_overlap(bag_path, output_path):
                raise ValueError(
                    f"input rosbag2 and {output_name} must not overlap: {bag_path} ; {output_path}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, _, config_source = parse_with_validation_profile(parser, argv, workflow_argument_defaults)
    dataset = args.dataset_dir.expanduser().resolve()
    clean_dataset = (
        args.clean_dataset_dir.expanduser().resolve()
        if args.clean_dataset_dir
        else dataset.with_name(f"{dataset.name}_clean")
    )
    report_dir = args.report_dir.expanduser().resolve()
    archive_dir = args.archive_dir.expanduser().resolve()
    workflow_summary = report_dir / "workflow-summary.json"
    history_file = (
        args.history_file.expanduser().resolve()
        if args.history_file
        else report_dir / "validation-history.json"
    )
    try:
        locations = discover_rosbags((), args.bag_dir, recursive=args.recursive)
        evaluator_stages = _evaluation_stages(args)
        _validate_workflow_paths(
            [location.path for location in locations],
            dataset=dataset,
            clean_dataset=clean_dataset,
            report_dir=report_dir,
            archive_dir=archive_dir,
        )
        if history_file == workflow_summary:
            raise ValueError("history file and workflow summary must be different")
    except Exception as error:
        print(f"Could not prepare workflow: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Discovered {len(locations)} rosbag2 episodes:")
        for location in locations:
            print(f"  {location.path}")
        print(f"Dataset: {dataset}")
        print(f"First filtered dataset: {clean_dataset}")
        print("Evaluation stages:")
        for stage in evaluator_stages:
            print(f"  {stage.name}: {shlex.join(stage.command)}")
        print(f"Reports: {report_dir}")
        print(f"Validation history: {history_file}")
        print(f"Archive: {archive_dir} ({args.archive_mode}, {args.archive_verify})")
        return 0

    report_dir.mkdir(parents=True, exist_ok=True)
    effective_config = {
        "version": PROFILE_VERSION,
        "input": effective_input_config(args),
        "dataset": effective_dataset_config(args, max_findings_name="max_video_findings"),
    }
    summary: dict[str, Any] = {
        "version": "2.0",
        "tool": "process-teleop-dataset",
        "status": "running",
        "input_bags": [str(location.path) for location in locations],
        "dataset": str(dataset),
        "validation_config": effective_config,
        "validation_config_source": (str(config_source) if config_source is not None else None),
        "validation_history": str(history_file),
        "evaluation_stages": [
            {"name": stage.name, "command": list(stage.command)} for stage in evaluator_stages
        ],
        "stages": [],
    }
    write_json_atomic(summary, workflow_summary)

    conversion_status = conversion_main(_conversion_arguments(args, dataset, report_dir))
    if conversion_status != 0:
        summary["status"] = "failed"
        _write_stage(
            summary, workflow_summary, "conversion", "failed", return_code=conversion_status
        )
        return 1
    _write_stage(summary, workflow_summary, "conversion", "passed")

    effective_dataset = dataset
    current_repo_id = args.repo_id
    clean_repo_id = args.clean_repo_id or f"{args.repo_id}_clean"
    filter_number = 0
    for ordinal, evaluator in enumerate(evaluator_stages, start=1):
        stage_reports = report_dir / "evaluations" / f"{ordinal:02d}-{evaluator.name}"
        try:
            status, result = run_evaluation_stage(
                evaluator, effective_dataset, stage_reports, args.fail_on
            )
        except Exception as error:
            print(f"Evaluation stage {evaluator.name!r} failed: {error}", file=sys.stderr)
            summary["status"] = "blocked"
            _write_stage(
                summary,
                workflow_summary,
                f"evaluation:{evaluator.name}",
                "blocked",
                message=str(error),
            )
            return EXIT_BLOCKED
        evaluation_status = (
            "findings"
            if status == EXIT_FINDINGS
            else "passed"
            if status == EXIT_CLEAN
            else "blocked"
        )
        _write_stage(
            summary,
            workflow_summary,
            f"evaluation:{evaluator.name}",
            evaluation_status,
            dataset=str(effective_dataset),
            result_file=str(stage_reports / "evaluation-result.json"),
            deletable_episode_indices=result["deletable_episode_indices"],
        )
        try:
            append_history(
                history_file,
                history_entry(
                    "validation",
                    effective_dataset,
                    result=result,
                    config=effective_config,
                    config_source=config_source,
                    details={
                        "stage": evaluator.name,
                        "result_file": str(stage_reports / "evaluation-result.json"),
                    },
                ),
            )
        except (OSError, ValueError) as error:
            summary["status"] = "blocked"
            _write_stage(
                summary,
                workflow_summary,
                f"history:{evaluator.name}",
                "blocked",
                message=str(error),
            )
            return EXIT_BLOCKED
        if status == EXIT_BLOCKED:
            summary["status"] = "blocked"
            write_json_atomic(summary, workflow_summary)
            return EXIT_BLOCKED
        if status == EXIT_CLEAN:
            continue

        filter_number += 1
        filtered_dataset = _filtered_destination(clean_dataset, filter_number, evaluator.name)
        output_repo_id = _filtered_repo_id(clean_repo_id, filter_number, evaluator.name)
        filter_result = stage_reports / "filter-result.json"
        removal_status = removal_main(
            [
                str(effective_dataset),
                "--output-dir",
                str(filtered_dataset),
                "--report",
                str(stage_reports / "evaluation-result.json"),
                "--fail-on",
                args.fail_on,
                "--source-repo-id",
                current_repo_id,
                "--output-repo-id",
                output_repo_id,
                "--result-file",
                str(filter_result),
                "--history-file",
                str(history_file),
            ]
        )
        if removal_status != 0:
            summary["status"] = "failed"
            _write_stage(
                summary,
                workflow_summary,
                f"filter:{evaluator.name}",
                "failed",
                return_code=removal_status,
            )
            return 1
        filter_details = read_json_object(filter_result)
        effective_dataset = Path(str(filter_details["effective_dataset"])).resolve()
        current_repo_id = output_repo_id
        removed = filter_details.get("removed_episode_indices", [])
        _write_stage(
            summary,
            workflow_summary,
            f"filter:{evaluator.name}",
            "filtered",
            removed_episode_indices=removed,
            effective_dataset=str(effective_dataset),
        )

        post_filter_reports = stage_reports / "post-filter"
        post_error: str | None
        try:
            post_status, post_result = run_evaluation_stage(
                evaluator, effective_dataset, post_filter_reports, args.fail_on
            )
        except Exception as error:
            print(f"Post-filter evaluation {evaluator.name!r} failed: {error}", file=sys.stderr)
            post_status = EXIT_BLOCKED
            post_error = str(error)
        else:
            post_error = None
            try:
                append_history(
                    history_file,
                    history_entry(
                        "validation",
                        effective_dataset,
                        result=post_result,
                        config=effective_config,
                        config_source=config_source,
                        details={
                            "stage": evaluator.name,
                            "phase": "post_filter",
                            "result_file": str(post_filter_reports / "evaluation-result.json"),
                        },
                    ),
                )
            except (OSError, ValueError) as error:
                post_status = EXIT_BLOCKED
                post_error = str(error)
        if post_status != EXIT_CLEAN:
            summary["status"] = "blocked"
            _write_stage(
                summary,
                workflow_summary,
                f"post-filter:{evaluator.name}",
                "blocked",
                return_code=post_status,
                message=post_error,
            )
            return EXIT_BLOCKED
        _write_stage(
            summary,
            workflow_summary,
            f"post-filter:{evaluator.name}",
            "passed",
            dataset=str(effective_dataset),
        )

    try:
        archive_result = archive_rosbags(
            locations,
            archive_dir,
            mode=args.archive_mode,
            verification=args.archive_verify,
            existing=args.archive_existing,
            manifest_path=report_dir / "archive-manifest.json",
        )
    except Exception as error:
        print(f"Could not archive rosbag2 data: {error}", file=sys.stderr)
        summary["status"] = "failed"
        _write_stage(summary, workflow_summary, "archive", "failed", message=str(error))
        return 1
    _write_stage(
        summary,
        workflow_summary,
        "archive",
        "passed",
        bags=len(archive_result["bags"]),
        mode=args.archive_mode,
    )
    summary["status"] = "complete"
    summary["effective_dataset"] = str(effective_dataset)
    write_json_atomic(summary, workflow_summary)
    print(f"Workflow complete. Effective dataset: {effective_dataset}")
    print(f"Summary: {workflow_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
