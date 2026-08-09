"""Validate rosbag camera streams before LeRobot video encoding."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

NANOSECONDS_PER_SECOND = 1_000_000_000
SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class TimedSample:
    """A decoded topic value and the provenance needed for input validation."""

    value: Any
    bag_stamp_ns: int
    source_stamp_ns: int | None
    source_index: int
    content_fingerprint: bytes | None = None


@dataclass(frozen=True)
class InputValidationConfig:
    source_gap_min_seconds: float = 1.0
    source_gap_factor: float = 5.0
    source_drop_factor: float = 2.5
    minimum_camera_source_frames: int = 2
    duplicate_min_source_frames: int = 2
    state_motion_threshold: float = 1e-4
    max_findings: int = 100


@dataclass(frozen=True)
class InputFinding:
    kind: str
    severity: str
    camera_id: str
    episode_index: int
    start_time_s: float
    end_time_s: float
    evidence: dict[str, int | float | bool | str]


@dataclass(frozen=True)
class CameraValidationStats:
    camera_id: str
    source_frames: int
    output_frames: int
    resampled_reuses: int
    duplicate_source_transitions: int
    median_source_period_ms: float | None
    max_source_age_ms: float


@dataclass(frozen=True)
class EpisodeValidationReport:
    episode_index: int
    bag_path: str
    severity: str
    camera_stats: tuple[CameraValidationStats, ...]
    findings: tuple[InputFinding, ...]


@dataclass
class ConversionValidationReport:
    episodes: list[EpisodeValidationReport] = field(default_factory=list)

    @property
    def overall_severity(self) -> str:
        if not self.episodes:
            return "PASS"
        return max(
            (episode.severity for episode in self.episodes),
            key=lambda severity: SEVERITY_ORDER[severity],
        )

    @property
    def findings(self) -> list[InputFinding]:
        return [finding for episode in self.episodes for finding in episode.findings]

    @property
    def summary(self) -> dict[str, int]:
        return {
            severity: sum(finding.severity == severity for finding in self.findings)
            for severity in ("WARN", "FAIL")
        }


@dataclass
class _DuplicateRun:
    start_ns: int
    end_ns: int
    source_frames: int
    source_stamps_advanced: bool


@dataclass(frozen=True)
class _SourceSample:
    bag_stamp_ns: int
    source_stamp_ns: int | None
    source_index: int
    content_fingerprint: bytes | None


@dataclass
class _CameraState:
    camera_id: str
    source_samples: list[_SourceSample] = field(default_factory=list)
    intervals_ns: list[int] = field(default_factory=list)
    duplicate_runs: list[_DuplicateRun] = field(default_factory=list)
    active_duplicate: _DuplicateRun | None = None
    output_frames: int = 0
    resampled_reuses: int = 0
    max_source_age_ns: int = 0
    last_output_source_index: int | None = None


@dataclass(frozen=True)
class _OutputState:
    tick_ns: int
    state: np.ndarray


def tracked_events(
    events: Iterable[tuple[int, str, TimedSample]],
    validator: ConversionInputValidator,
) -> Iterator[tuple[int, str, TimedSample]]:
    """Observe decoded events without adding another rosbag2 pass."""
    for timestamp, topic, sample in events:
        validator.observe_event(timestamp, topic, sample)
        yield timestamp, topic, sample


class ConversionInputValidator:
    """Collect source-stream and resampling evidence for one episode."""

    def __init__(
        self,
        episode_index: int,
        bag_path: Path,
        camera_topics: Sequence[tuple[str, str]],
        state_topics: Sequence[tuple[str, str]],
        config: InputValidationConfig,
    ) -> None:
        self.episode_index = episode_index
        self.bag_path = bag_path
        self.camera_topics = {topic: camera_id for camera_id, topic in camera_topics}
        self.state_topics = tuple(topic for _, topic in state_topics)
        self.config = config
        self._cameras = {
            topic: _CameraState(camera_id) for topic, camera_id in self.camera_topics.items()
        }
        self._output_states: list[_OutputState] = []
        self._findings: list[InputFinding] = []
        self._first_event_ns: int | None = None
        self._last_event_ns: int | None = None

    def observe_event(self, timestamp: int, topic: str, sample: TimedSample) -> None:
        if self._first_event_ns is None:
            self._first_event_ns = timestamp
        self._last_event_ns = timestamp
        camera = self._cameras.get(topic)
        if camera is None:
            return

        current = _SourceSample(
            bag_stamp_ns=sample.bag_stamp_ns,
            source_stamp_ns=sample.source_stamp_ns,
            source_index=sample.source_index,
            content_fingerprint=sample.content_fingerprint,
        )
        previous = camera.source_samples[-1] if camera.source_samples else None
        if previous is not None:
            interval = current.bag_stamp_ns - previous.bag_stamp_ns
            camera.intervals_ns.append(interval)
            if (
                previous.source_stamp_ns is not None
                and current.source_stamp_ns is not None
                and current.source_stamp_ns < previous.source_stamp_ns
            ):
                self._add_finding(
                    InputFinding(
                        kind="camera_source_stamp_regression",
                        severity="FAIL",
                        camera_id=camera.camera_id,
                        episode_index=self.episode_index,
                        start_time_s=self._relative_seconds(previous.bag_stamp_ns),
                        end_time_s=self._relative_seconds(current.bag_stamp_ns),
                        evidence={
                            "previous_source_stamp_ns": previous.source_stamp_ns,
                            "current_source_stamp_ns": current.source_stamp_ns,
                        },
                    )
                )

            if (
                previous.content_fingerprint is not None
                and current.content_fingerprint == previous.content_fingerprint
            ):
                stamps_advanced = (
                    previous.source_stamp_ns is None
                    or current.source_stamp_ns is None
                    or current.source_stamp_ns > previous.source_stamp_ns
                )
                if camera.active_duplicate is None:
                    camera.active_duplicate = _DuplicateRun(
                        start_ns=previous.bag_stamp_ns,
                        end_ns=current.bag_stamp_ns,
                        source_frames=2,
                        source_stamps_advanced=stamps_advanced,
                    )
                else:
                    camera.active_duplicate.end_ns = current.bag_stamp_ns
                    camera.active_duplicate.source_frames += 1
                    camera.active_duplicate.source_stamps_advanced &= stamps_advanced
            else:
                self._close_duplicate_run(camera)

        camera.source_samples.append(current)

    def observe_output(self, tick_ns: int, snapshot: dict[str, TimedSample]) -> None:
        states = [
            np.asarray(snapshot[topic].value, dtype=np.float32) for topic in self.state_topics
        ]
        self._output_states.append(_OutputState(tick_ns, np.concatenate(states)))
        for topic, camera in self._cameras.items():
            sample = snapshot[topic]
            camera.output_frames += 1
            if camera.last_output_source_index == sample.source_index:
                camera.resampled_reuses += 1
            camera.last_output_source_index = sample.source_index
            camera.max_source_age_ns = max(
                camera.max_source_age_ns, max(0, tick_ns - sample.bag_stamp_ns)
            )

    def finish(self) -> EpisodeValidationReport:
        for camera in self._cameras.values():
            self._close_duplicate_run(camera)
            self._add_sample_count_finding(camera)
            self._add_gap_findings(camera)
            self._add_duplicate_findings(camera)

        findings = tuple(self._findings[: self.config.max_findings])
        if len(self._findings) > self.config.max_findings:
            findings += (
                InputFinding(
                    kind="findings_truncated",
                    severity="WARN",
                    camera_id="*",
                    episode_index=self.episode_index,
                    start_time_s=0.0,
                    end_time_s=0.0,
                    evidence={"maximum": self.config.max_findings},
                ),
            )
        severity = max(
            (finding.severity for finding in findings),
            key=lambda item: SEVERITY_ORDER[item],
            default="PASS",
        )
        stats = tuple(self._camera_stats(camera) for camera in self._cameras.values())
        return EpisodeValidationReport(
            episode_index=self.episode_index,
            bag_path=str(self.bag_path),
            severity=severity,
            camera_stats=stats,
            findings=findings,
        )

    def _close_duplicate_run(self, camera: _CameraState) -> None:
        if camera.active_duplicate is not None:
            camera.duplicate_runs.append(camera.active_duplicate)
            camera.active_duplicate = None

    def _add_sample_count_finding(self, camera: _CameraState) -> None:
        source_frames = len(camera.source_samples)
        if source_frames >= self.config.minimum_camera_source_frames:
            return
        start_ns = (
            camera.source_samples[0].bag_stamp_ns
            if camera.source_samples
            else self._first_event_ns or 0
        )
        end_ns = self._last_event_ns if self._last_event_ns is not None else start_ns
        self._add_finding(
            InputFinding(
                kind="insufficient_camera_source_frames",
                severity="FAIL",
                camera_id=camera.camera_id,
                episode_index=self.episode_index,
                start_time_s=self._relative_seconds(start_ns),
                end_time_s=self._relative_seconds(end_ns),
                evidence={
                    "source_frames": source_frames,
                    "minimum_source_frames": self.config.minimum_camera_source_frames,
                },
            )
        )

    def _add_gap_findings(self, camera: _CameraState) -> None:
        positive_intervals = [interval for interval in camera.intervals_ns if interval > 0]
        if not positive_intervals:
            return
        median_ns = float(np.median(np.asarray(positive_intervals, dtype=np.float64)))
        gap_threshold_ns = max(
            self.config.source_gap_min_seconds * NANOSECONDS_PER_SECOND,
            median_ns * self.config.source_gap_factor,
        )
        drop_threshold_ns = median_ns * self.config.source_drop_factor
        samples = camera.source_samples
        for previous, current in pairwise(samples):
            interval_ns = current.bag_stamp_ns - previous.bag_stamp_ns
            if interval_ns > gap_threshold_ns:
                self._add_finding(
                    InputFinding(
                        kind="camera_source_gap",
                        severity="FAIL",
                        camera_id=camera.camera_id,
                        episode_index=self.episode_index,
                        start_time_s=self._relative_seconds(previous.bag_stamp_ns),
                        end_time_s=self._relative_seconds(current.bag_stamp_ns),
                        evidence={
                            "gap_ms": interval_ns / 1_000_000,
                            "threshold_ms": gap_threshold_ns / 1_000_000,
                            "median_period_ms": median_ns / 1_000_000,
                        },
                    )
                )
            elif interval_ns >= drop_threshold_ns:
                self._add_finding(
                    InputFinding(
                        kind="camera_source_frame_drop",
                        severity="WARN",
                        camera_id=camera.camera_id,
                        episode_index=self.episode_index,
                        start_time_s=self._relative_seconds(previous.bag_stamp_ns),
                        end_time_s=self._relative_seconds(current.bag_stamp_ns),
                        evidence={
                            "gap_ms": interval_ns / 1_000_000,
                            "drop_threshold_ms": drop_threshold_ns / 1_000_000,
                            "median_period_ms": median_ns / 1_000_000,
                            "estimated_missing_frames": max(1, round(interval_ns / median_ns) - 1),
                        },
                    )
                )

        if samples and self._last_event_ns is not None:
            tail_gap_ns = self._last_event_ns - samples[-1].bag_stamp_ns
            if tail_gap_ns > gap_threshold_ns:
                self._add_finding(
                    InputFinding(
                        kind="camera_source_tail_gap",
                        severity="FAIL",
                        camera_id=camera.camera_id,
                        episode_index=self.episode_index,
                        start_time_s=self._relative_seconds(samples[-1].bag_stamp_ns),
                        end_time_s=self._relative_seconds(self._last_event_ns),
                        evidence={
                            "gap_ms": tail_gap_ns / 1_000_000,
                            "threshold_ms": gap_threshold_ns / 1_000_000,
                            "median_period_ms": median_ns / 1_000_000,
                        },
                    )
                )
            elif tail_gap_ns >= drop_threshold_ns:
                self._add_finding(
                    InputFinding(
                        kind="camera_source_tail_drop",
                        severity="WARN",
                        camera_id=camera.camera_id,
                        episode_index=self.episode_index,
                        start_time_s=self._relative_seconds(samples[-1].bag_stamp_ns),
                        end_time_s=self._relative_seconds(self._last_event_ns),
                        evidence={
                            "gap_ms": tail_gap_ns / 1_000_000,
                            "drop_threshold_ms": drop_threshold_ns / 1_000_000,
                            "median_period_ms": median_ns / 1_000_000,
                            "estimated_missing_frames": max(1, round(tail_gap_ns / median_ns)),
                        },
                    )
                )

    def _add_duplicate_findings(self, camera: _CameraState) -> None:
        for run in camera.duplicate_runs:
            if run.source_frames < self.config.duplicate_min_source_frames:
                continue
            max_state_motion = self._max_state_motion(run.start_ns, run.end_ns)
            other_camera_changes = self._other_camera_changes(
                camera.camera_id, run.start_ns, run.end_ns
            )
            self._add_finding(
                InputFinding(
                    kind="duplicate_source_frames",
                    severity="WARN" if run.source_stamps_advanced else "FAIL",
                    camera_id=camera.camera_id,
                    episode_index=self.episode_index,
                    start_time_s=self._relative_seconds(run.start_ns),
                    end_time_s=self._relative_seconds(run.end_ns),
                    evidence={
                        "source_frames": run.source_frames,
                        "duplicate_transitions": run.source_frames - 1,
                        "source_stamps_advanced": run.source_stamps_advanced,
                        "max_state_motion": max_state_motion,
                        "state_motion_detected": (
                            max_state_motion >= self.config.state_motion_threshold
                        ),
                        "other_camera_content_changes": other_camera_changes,
                    },
                )
            )

    def _max_state_motion(self, start_ns: int, end_ns: int) -> float:
        values = [
            output.state for output in self._output_states if start_ns <= output.tick_ns <= end_ns
        ]
        if len(values) < 2:
            return 0.0
        differences = np.abs(np.diff(np.stack(values), axis=0))
        return float(np.max(differences)) if differences.size else 0.0

    def _other_camera_changes(self, camera_id: str, start_ns: int, end_ns: int) -> int:
        changes = 0
        for camera in self._cameras.values():
            if camera.camera_id == camera_id:
                continue
            samples = [
                sample
                for sample in camera.source_samples
                if start_ns <= sample.bag_stamp_ns <= end_ns
            ]
            changes += sum(
                current.content_fingerprint != previous.content_fingerprint
                for previous, current in pairwise(samples)
            )
        return changes

    def _camera_stats(self, camera: _CameraState) -> CameraValidationStats:
        positive_intervals = [interval for interval in camera.intervals_ns if interval > 0]
        median_ms = (
            float(np.median(np.asarray(positive_intervals, dtype=np.float64))) / 1_000_000
            if positive_intervals
            else None
        )
        return CameraValidationStats(
            camera_id=camera.camera_id,
            source_frames=len(camera.source_samples),
            output_frames=camera.output_frames,
            resampled_reuses=camera.resampled_reuses,
            duplicate_source_transitions=sum(
                run.source_frames - 1 for run in camera.duplicate_runs
            ),
            median_source_period_ms=median_ms,
            max_source_age_ms=camera.max_source_age_ns / 1_000_000,
        )

    def _relative_seconds(self, stamp_ns: int) -> float:
        origin = self._first_event_ns if self._first_event_ns is not None else stamp_ns
        return (stamp_ns - origin) / NANOSECONDS_PER_SECOND

    def _add_finding(self, finding: InputFinding) -> None:
        self._findings.append(finding)


class InputValidationError(RuntimeError):
    def __init__(self, report: ConversionValidationReport) -> None:
        super().__init__(
            f"input validation reported {report.overall_severity}: "
            f"{len(report.findings)} finding(s)"
        )
        self.report = report


def report_to_dict(report: ConversionValidationReport) -> dict[str, Any]:
    return {
        "tool": "rosbag-to-lerobot-input-validation",
        "overall_severity": report.overall_severity,
        "summary": report.summary,
        "episodes": [asdict(episode) for episode in report.episodes],
    }


def report_to_json(report: ConversionValidationReport) -> str:
    return json.dumps(report_to_dict(report), indent=2)


def report_to_markdown(report: ConversionValidationReport) -> str:
    lines = [
        "# rosbag-to-lerobot input validation report",
        "",
        f"- **Overall:** **{report.overall_severity}**",
        f"- **Episodes:** {len(report.episodes)}",
        f"- **WARN:** {report.summary['WARN']}",
        f"- **FAIL:** {report.summary['FAIL']}",
        "",
        "## Camera statistics",
        "",
        (
            "| Episode | Camera | Source | Output | Resampled reuse | Source duplicates | "
            "Median period (ms) | Max age (ms) |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode in report.episodes:
        for stats in episode.camera_stats:
            median = (
                f"{stats.median_source_period_ms:.3f}"
                if stats.median_source_period_ms is not None
                else "-"
            )
            lines.append(
                f"| {episode.episode_index} | `{stats.camera_id}` | {stats.source_frames} | "
                f"{stats.output_frames} | {stats.resampled_reuses} | "
                f"{stats.duplicate_source_transitions} | {median} | "
                f"{stats.max_source_age_ms:.3f} |"
            )
    lines.extend(["", "## Findings", ""])
    if not report.findings:
        lines.append("No rosbag input anomalies were detected.")
    else:
        lines.extend(
            [
                "| Severity | Kind | Camera | Episode | Time (s) | Evidence |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for finding in report.findings:
            evidence = ", ".join(f"{key}={value}" for key, value in finding.evidence.items())
            lines.append(
                f"| **{finding.severity}** | {finding.kind} | `{finding.camera_id}` | "
                f"{finding.episode_index} | {finding.start_time_s:.3f}-"
                f"{finding.end_time_s:.3f} | {evidence} |"
            )
    return "\n".join(lines)


def write_report(report: ConversionValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = report_to_markdown(report) if path.suffix.lower() == ".md" else report_to_json(report)
    path.write_text(content, encoding="utf-8")
