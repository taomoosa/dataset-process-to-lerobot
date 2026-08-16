"""Shared contract for independently executable dataset evaluators."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .configuration import add_validation_config_argument
from .report_utils import (
    SEVERITY_ORDER,
    episode_indices_from_report,
    report_reaches_threshold,
)

CONTRACT_NAME = "lerobot-dataset-evaluation/v1"
EXIT_CLEAN = 0
EXIT_FINDINGS = 10
EXIT_BLOCKED = 20


def add_common_evaluator_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the stable CLI arguments implemented by every evaluator wrapper."""
    parser.add_argument("dataset", type=Path, help="local LeRobotDataset V3 directory")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--result-file",
        type=Path,
        help="normalized episode-selection result (default: REPORT_DIR/evaluation-result.json)",
    )
    parser.add_argument("--fail-on", choices=("warn", "fail"), default="fail")
    add_validation_config_argument(parser)


def result_path(report_dir: Path, requested: Path | None) -> Path:
    return requested.expanduser().resolve() if requested else report_dir / "evaluation-result.json"


def make_evaluation_result(
    *,
    evaluator: str,
    dataset: Path,
    fail_on: str,
    raw_report: dict[str, Any] | None,
    return_code: int,
    artifacts: dict[str, str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Normalize evaluator-specific output into the workflow selection contract."""
    threshold = fail_on.upper()
    indices: set[int] = set()
    blockers: list[str] = []
    declared_severity = (
        str(raw_report.get("overall_severity", "UNKNOWN")).upper()
        if raw_report is not None
        else "FAIL"
    )
    overall_severity = declared_severity if declared_severity in SEVERITY_ORDER else "FAIL"

    if error:
        blockers.append(error)
    elif raw_report is None:
        blockers.append(f"{evaluator} did not produce a machine-readable report")
    elif declared_severity not in SEVERITY_ORDER:
        blockers.append(f"{evaluator} reported an invalid overall severity: {declared_severity}")
    else:
        indices = episode_indices_from_report(raw_report, fail_on)
        reaches_threshold = report_reaches_threshold(raw_report, fail_on)
        if indices and not reaches_threshold:
            blockers.append(
                f"{evaluator} selected episodes without reaching the {threshold} threshold"
            )
        elif reaches_threshold and not indices:
            blockers.append(
                f"{evaluator} reached the {threshold} threshold without episode indices"
            )
        elif return_code != 0 and not reaches_threshold:
            blockers.append(f"{evaluator} exited with status {return_code} without report findings")

    status = "blocked" if blockers else "findings" if indices else "pass"
    return {
        "version": "1.0",
        "contract": CONTRACT_NAME,
        "evaluator": evaluator,
        "dataset_path": str(dataset),
        "fail_on": threshold,
        "status": status,
        "overall_severity": overall_severity,
        "deletable_episode_indices": sorted(indices),
        "findings": [
            {
                "episode_index": index,
                "severity": threshold,
                "kind": "evaluation_failure",
                "evaluator": evaluator,
            }
            for index in sorted(indices)
        ],
        "non_episode_blockers": blockers,
        "artifacts": artifacts or {},
        "evaluator_return_code": return_code,
    }


def validate_evaluation_result(
    result: dict[str, Any],
    *,
    dataset: Path | None = None,
    fail_on: str | None = None,
) -> None:
    """Reject malformed or stale evaluator results before filtering a dataset."""
    if result.get("contract") != CONTRACT_NAME:
        raise ValueError(f"evaluation result does not declare contract {CONTRACT_NAME!r}")
    if result.get("status") not in {"pass", "findings", "blocked"}:
        raise ValueError("evaluation result has an invalid status")
    if result.get("fail_on") not in {"WARN", "FAIL"}:
        raise ValueError("evaluation result has an invalid fail_on value")
    if fail_on is not None and result["fail_on"] != fail_on.upper():
        raise ValueError("evaluation result used a different fail_on threshold")
    if not isinstance(result.get("evaluator"), str) or not result["evaluator"]:
        raise ValueError("evaluation result has an invalid evaluator")
    if result.get("overall_severity") not in SEVERITY_ORDER:
        raise ValueError("evaluation result has an invalid overall_severity")
    indices = result.get("deletable_episode_indices")
    if not isinstance(indices, list) or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices
    ):
        raise ValueError("evaluation result has invalid deletable_episode_indices")
    if indices != sorted(set(indices)):
        raise ValueError("deletable_episode_indices must be sorted and unique")
    blockers = result.get("non_episode_blockers")
    if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
        raise ValueError("evaluation result has invalid non_episode_blockers")
    if not isinstance(result.get("findings"), list):
        raise ValueError("evaluation result has invalid findings")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or any(
        not isinstance(name, str) or not isinstance(path, str) for name, path in artifacts.items()
    ):
        raise ValueError("evaluation result has invalid artifacts")
    return_code = result.get("evaluator_return_code")
    if not isinstance(return_code, int) or isinstance(return_code, bool):
        raise ValueError("evaluation result has an invalid evaluator_return_code")
    status = result["status"]
    threshold = SEVERITY_ORDER[result["fail_on"]]
    severity = SEVERITY_ORDER[result["overall_severity"]]
    if status == "pass" and (indices or blockers):
        raise ValueError("a passing evaluation result cannot contain selections or blockers")
    if status == "pass" and severity >= threshold:
        raise ValueError("a passing evaluation result reaches its failure threshold")
    if status == "findings" and (not indices or blockers):
        raise ValueError("a findings result must contain selections and no blockers")
    if status == "findings" and severity < threshold:
        raise ValueError("a findings result does not reach its failure threshold")
    if status == "blocked" and not blockers:
        raise ValueError("a blocked evaluation result must contain a blocker")
    if dataset is not None:
        declared = result.get("dataset_path")
        declared_path = Path(declared).expanduser().resolve() if isinstance(declared, str) else None
        if declared_path != dataset.resolve():
            raise ValueError("evaluation result belongs to a different dataset")


def exit_code_for_result(result: dict[str, Any]) -> int:
    return {
        "pass": EXIT_CLEAN,
        "findings": EXIT_FINDINGS,
        "blocked": EXIT_BLOCKED,
    }[str(result["status"])]
