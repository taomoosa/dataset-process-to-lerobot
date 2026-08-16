"""Adapt encoded-video checks to the dataset evaluator contract."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence

from .configuration import (
    effective_video_config,
    parse_with_validation_profile,
    video_evaluator_argument_defaults,
)
from .evaluator_contract import (
    EXIT_BLOCKED,
    add_common_evaluator_arguments,
    exit_code_for_result,
    make_evaluation_result,
    result_path,
)
from .lerobot_video_check import main as video_check_main
from .report_utils import read_json_object, write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-lerobot-video",
        description="Check dataset videos and emit a normalized episode-selection result.",
    )
    add_common_evaluator_arguments(parser)
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
    args, _, config_source = parse_with_validation_profile(
        parser, argv, video_evaluator_argument_defaults
    )
    dataset = args.dataset.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    output_path = result_path(report_dir, args.result_file)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_json = report_dir / "lerobot-video-check.json"
    markdown = report_dir / "lerobot-video-check.md"
    token = uuid.uuid4().hex
    temporary_json = report_dir / f".lerobot-video-check-{token}.json"
    temporary_markdown = report_dir / f".lerobot-video-check-{token}.md"
    arguments = [
        str(dataset),
        "--ci",
        "--fail-on",
        args.fail_on,
        "--markdown",
        str(temporary_markdown),
        "--json-file",
        str(temporary_json),
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
        arguments.extend(("--features", args.features))
    if args.max_episodes is not None:
        arguments.extend(("--max-episodes", str(args.max_episodes)))

    raw_report = None
    return_code = EXIT_BLOCKED
    error: str | None = None
    try:
        if not dataset.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset}")
        return_code = video_check_main(arguments)
        if temporary_json.exists():
            raw_report = read_json_object(temporary_json)
            temporary_json.replace(raw_json)
        if temporary_markdown.exists():
            temporary_markdown.replace(markdown)
    except (OSError, ValueError) as caught:
        error = str(caught)
        print(f"Could not run video evaluation: {caught}", file=sys.stderr)
    finally:
        if temporary_json.exists():
            temporary_json.unlink()
        if temporary_markdown.exists():
            temporary_markdown.unlink()

    result = make_evaluation_result(
        evaluator="lerobot-video-check",
        dataset=dataset,
        fail_on=args.fail_on,
        raw_report=raw_report,
        return_code=return_code,
        artifacts={
            name: str(path)
            for name, path in {"json": raw_json, "markdown": markdown}.items()
            if path.exists()
        },
        error=error,
    )
    result["validation_config"] = {
        "fail_on": args.fail_on,
        "video": effective_video_config(args),
    }
    result["validation_config_source"] = str(config_source) if config_source is not None else None
    write_json_atomic(result, output_path)
    print(f"Wrote evaluation result to {output_path}", file=sys.stderr)
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
