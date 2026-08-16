"""Load reusable validation settings without accepting run-specific paths."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .report_utils import read_json_object

PROFILE_VERSION = "1.0"
DEFAULT_VALIDATION_PROFILE: dict[str, Any] = {
    "version": PROFILE_VERSION,
    "input": {
        "policy": "drop",
        "drop_on": "fail",
        "source_gap_min_seconds": 1.0,
        "source_gap_factor": 5.0,
        "source_drop_factor": 2.5,
        "minimum_camera_source_frames": 2,
        "duplicate_min_source_frames": 2,
        "state_motion_threshold": 1e-4,
        "max_findings": 100,
    },
    "dataset": {
        "fail_on": "fail",
        "doctor": {"enabled": True},
        "video": {
            "features": None,
            "max_episodes": None,
            "thumbnail_size": 32,
            "duplicate_threshold": 0.1,
            "freeze_min_seconds": 1.0,
            "repeat_threshold": 0.1,
            "repeat_min_cycles": 3,
            "repeat_max_period_seconds": 5.0,
            "jump_percentile": 10.0,
            "jump_min_score": 0.1,
            "artifact_block_threshold": 20.0,
            "artifact_min_block_fraction": 0.05,
            "artifact_max_duration_frames": 3,
            "flat_frame_std_threshold": 2.0,
            "temporal_discontinuity_threshold": 5.0,
            "state_motion_support_threshold": 0.02,
            "max_findings": 100,
        },
    },
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"{label} contains unknown keys: {', '.join(unknown)}")


def _number(value: Any, label: str, *, minimum: float, inclusive: bool = True) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    valid = result >= minimum if inclusive else result > minimum
    if not valid:
        operator = "at least" if inclusive else "greater than"
        raise ValueError(f"{label} must be {operator} {minimum}")
    return result


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _choice(value: Any, label: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return value


def _merge_profile(raw: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(raw, {"version", "input", "dataset"}, "validation profile")
    if raw.get("version") != PROFILE_VERSION:
        raise ValueError(f"validation profile version must be {PROFILE_VERSION!r}")
    profile = copy.deepcopy(DEFAULT_VALIDATION_PROFILE)

    if "input" in raw:
        supplied_input = _object(raw["input"], "input")
        _reject_unknown(supplied_input, set(profile["input"]), "input")
        profile["input"].update(supplied_input)

    if "dataset" in raw:
        supplied_dataset = _object(raw["dataset"], "dataset")
        _reject_unknown(supplied_dataset, {"fail_on", "doctor", "video"}, "dataset")
        if "fail_on" in supplied_dataset:
            profile["dataset"]["fail_on"] = supplied_dataset["fail_on"]
        if "doctor" in supplied_dataset:
            supplied_doctor = _object(supplied_dataset["doctor"], "dataset.doctor")
            _reject_unknown(supplied_doctor, {"enabled"}, "dataset.doctor")
            profile["dataset"]["doctor"].update(supplied_doctor)
        if "video" in supplied_dataset:
            supplied_video = _object(supplied_dataset["video"], "dataset.video")
            _reject_unknown(supplied_video, set(profile["dataset"]["video"]), "dataset.video")
            profile["dataset"]["video"].update(supplied_video)
    return profile


def _validate_profile(profile: dict[str, Any]) -> None:
    input_config = profile["input"]
    input_config["policy"] = _choice(
        input_config["policy"], "input.policy", {"off", "warn", "fail", "drop"}
    )
    input_config["drop_on"] = _choice(input_config["drop_on"], "input.drop_on", {"warn", "fail"})
    input_config["source_gap_min_seconds"] = _number(
        input_config["source_gap_min_seconds"],
        "input.source_gap_min_seconds",
        minimum=0,
        inclusive=False,
    )
    for name in ("source_gap_factor", "source_drop_factor"):
        input_config[name] = _number(
            input_config[name], f"input.{name}", minimum=1, inclusive=False
        )
    for name in ("minimum_camera_source_frames", "duplicate_min_source_frames"):
        input_config[name] = _integer(input_config[name], f"input.{name}", minimum=2)
    input_config["state_motion_threshold"] = _number(
        input_config["state_motion_threshold"],
        "input.state_motion_threshold",
        minimum=0,
    )
    input_config["max_findings"] = _integer(
        input_config["max_findings"], "input.max_findings", minimum=1
    )

    dataset = profile["dataset"]
    dataset["fail_on"] = _choice(dataset["fail_on"], "dataset.fail_on", {"warn", "fail"})
    doctor_enabled = dataset["doctor"]["enabled"]
    if not isinstance(doctor_enabled, bool):
        raise ValueError("dataset.doctor.enabled must be a boolean")

    video = dataset["video"]
    features = video["features"]
    if features is not None:
        if (
            not isinstance(features, list)
            or not features
            or any(not isinstance(item, str) or not item.strip() for item in features)
        ):
            raise ValueError("dataset.video.features must be null or a non-empty string array")
        normalized_features = [item.strip() for item in features]
        if len(set(normalized_features)) != len(normalized_features):
            raise ValueError("dataset.video.features must not contain duplicates")
        video["features"] = normalized_features
    if video["max_episodes"] is not None:
        video["max_episodes"] = _integer(
            video["max_episodes"], "dataset.video.max_episodes", minimum=1
        )
    for name, minimum in (
        ("thumbnail_size", 8),
        ("repeat_min_cycles", 2),
        ("artifact_max_duration_frames", 1),
        ("max_findings", 1),
    ):
        video[name] = _integer(video[name], f"dataset.video.{name}", minimum=minimum)
    for name in (
        "duplicate_threshold",
        "repeat_threshold",
        "jump_min_score",
        "flat_frame_std_threshold",
        "state_motion_support_threshold",
    ):
        video[name] = _number(video[name], f"dataset.video.{name}", minimum=0)
    for name in (
        "freeze_min_seconds",
        "repeat_max_period_seconds",
        "artifact_block_threshold",
        "temporal_discontinuity_threshold",
    ):
        video[name] = _number(video[name], f"dataset.video.{name}", minimum=0, inclusive=False)
    video["artifact_min_block_fraction"] = _number(
        video["artifact_min_block_fraction"],
        "dataset.video.artifact_min_block_fraction",
        minimum=0,
        inclusive=False,
    )
    if video["artifact_min_block_fraction"] > 1:
        raise ValueError("dataset.video.artifact_min_block_fraction must not exceed 1")
    video["jump_percentile"] = _number(
        video["jump_percentile"],
        "dataset.video.jump_percentile",
        minimum=0,
        inclusive=False,
    )
    if video["jump_percentile"] > 100:
        raise ValueError("dataset.video.jump_percentile must not exceed 100")


def load_validation_profile(path: Path | None) -> dict[str, Any]:
    """Return a complete validated profile, using built-in defaults when omitted."""
    if path is None:
        profile = copy.deepcopy(DEFAULT_VALIDATION_PROFILE)
    else:
        resolved = path.expanduser().resolve()
        profile = _merge_profile(read_json_object(resolved))
    _validate_profile(profile)
    return profile


def add_validation_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validation-config",
        type=Path,
        help="JSON validation profile; explicit CLI validation options override its values",
    )


def parse_with_validation_profile(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    defaults: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[argparse.Namespace, dict[str, Any], Path | None]:
    """Load a profile before the full parse so explicit CLI options retain precedence."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--validation-config", type=Path)
    known, _ = bootstrap.parse_known_args(arguments)
    source = known.validation_config.expanduser().resolve() if known.validation_config else None
    try:
        profile = load_validation_profile(source)
    except ValueError as error:
        parser.error(str(error))
    if source is not None:
        parser.set_defaults(**defaults(profile))
    return parser.parse_args(arguments), profile, source


def input_argument_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    value = profile["input"]
    return {
        "input_validation": value["policy"],
        "input_validation_drop_on": value["drop_on"],
        "source_gap_min_seconds": value["source_gap_min_seconds"],
        "source_gap_factor": value["source_gap_factor"],
        "source_drop_factor": value["source_drop_factor"],
        "minimum_camera_source_frames": value["minimum_camera_source_frames"],
        "duplicate_min_source_frames": value["duplicate_min_source_frames"],
        "state_motion_threshold": value["state_motion_threshold"],
        "max_input_findings": value["max_findings"],
    }


def video_argument_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    video = profile["dataset"]["video"]
    features = video["features"]
    return {
        **{name: value for name, value in video.items() if name != "features"},
        "features": ",".join(features) if features else None,
    }


def dataset_argument_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "fail_on": profile["dataset"]["fail_on"],
        "skip_doctor": not profile["dataset"]["doctor"]["enabled"],
        **video_argument_defaults(profile),
    }


def doctor_argument_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    return {"fail_on": profile["dataset"]["fail_on"]}


def video_evaluator_argument_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "fail_on": profile["dataset"]["fail_on"],
        **video_argument_defaults(profile),
    }


def workflow_argument_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    dataset = dataset_argument_defaults(profile)
    dataset["max_video_findings"] = dataset.pop("max_findings")
    return {**input_argument_defaults(profile), **dataset}


def effective_input_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "policy": args.input_validation,
        "drop_on": args.input_validation_drop_on,
        "source_gap_min_seconds": args.source_gap_min_seconds,
        "source_gap_factor": args.source_gap_factor,
        "source_drop_factor": args.source_drop_factor,
        "minimum_camera_source_frames": args.minimum_camera_source_frames,
        "duplicate_min_source_frames": args.duplicate_min_source_frames,
        "state_motion_threshold": args.state_motion_threshold,
        "max_findings": args.max_input_findings,
    }


def effective_video_config(
    args: argparse.Namespace, *, max_findings_name: str = "max_findings"
) -> dict[str, Any]:
    features = tuple(part.strip() for part in args.features.split(",")) if args.features else None
    return {
        "features": list(features) if features else None,
        "max_episodes": getattr(args, "max_episodes", None),
        "thumbnail_size": args.thumbnail_size,
        "duplicate_threshold": args.duplicate_threshold,
        "freeze_min_seconds": args.freeze_min_seconds,
        "repeat_threshold": args.repeat_threshold,
        "repeat_min_cycles": args.repeat_min_cycles,
        "repeat_max_period_seconds": args.repeat_max_period_seconds,
        "jump_percentile": args.jump_percentile,
        "jump_min_score": args.jump_min_score,
        "artifact_block_threshold": args.artifact_block_threshold,
        "artifact_min_block_fraction": args.artifact_min_block_fraction,
        "artifact_max_duration_frames": args.artifact_max_duration_frames,
        "flat_frame_std_threshold": args.flat_frame_std_threshold,
        "temporal_discontinuity_threshold": args.temporal_discontinuity_threshold,
        "state_motion_support_threshold": args.state_motion_support_threshold,
        "max_findings": getattr(args, max_findings_name),
    }


def effective_dataset_config(
    args: argparse.Namespace, *, max_findings_name: str = "max_findings"
) -> dict[str, Any]:
    return {
        "fail_on": args.fail_on,
        "doctor": {"enabled": not getattr(args, "skip_doctor", False)},
        "video": effective_video_config(args, max_findings_name=max_findings_name),
    }
