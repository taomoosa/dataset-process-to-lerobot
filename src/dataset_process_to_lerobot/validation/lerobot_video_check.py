"""Check LeRobotDataset v3 videos for temporal freeze and repeat anomalies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import (
    add_validation_config_argument,
    parse_with_validation_profile,
    video_evaluator_argument_defaults,
)

VERSION = "0.1.0"
SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class VideoSegment:
    feature: str
    episode_index: int
    video_path: Path
    start_frame: int
    length: int
    fps: int

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.length


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: str
    feature: str
    episode_index: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    jump_score: float
    jump_threshold: float
    period_frames: int | None = None
    cycles: int | None = None


@dataclass
class CheckResult:
    name: str
    severity: str = "PASS"
    messages: list[dict[str, str]] = field(default_factory=list)

    def add(self, severity: str, message: str) -> None:
        self.messages.append({"severity": severity, "message": message})
        if SEVERITY_ORDER[severity] > SEVERITY_ORDER[self.severity]:
            self.severity = severity


@dataclass(frozen=True)
class AnalysisConfig:
    thumbnail_size: int = 32
    duplicate_threshold: float = 0.1
    freeze_min_seconds: float = 1.0
    repeat_threshold: float = 0.1
    repeat_min_cycles: int = 3
    repeat_max_period_seconds: float = 5.0
    jump_percentile: float = 10.0
    jump_min_score: float = 0.1
    artifact_block_threshold: float = 20.0
    artifact_min_block_fraction: float = 0.05
    artifact_max_duration_frames: int = 3
    flat_frame_std_threshold: float = 2.0
    temporal_discontinuity_threshold: float = 5.0
    state_motion_support_threshold: float = 0.02
    max_findings: int = 100


@dataclass
class VideoReport:
    dataset_path: str
    dataset_name: str
    codebase_version: str | None
    format_version: str
    total_episodes: int
    total_frames: int
    fps: int
    checks: list[CheckResult]
    findings: list[Finding]
    config: AnalysisConfig

    @property
    def overall_severity(self) -> str:
        return max(
            (check.severity for check in self.checks),
            key=lambda severity: SEVERITY_ORDER[severity],
        )

    @property
    def summary(self) -> dict[str, int]:
        return {
            severity: sum(check.severity == severity for check in self.checks)
            for severity in ("PASS", "WARN", "FAIL")
        }


@dataclass(frozen=True)
class SignatureAnalysis:
    jump_threshold: float
    frozen: tuple[tuple[int, int, float], ...]
    repeated: tuple[tuple[int, int, int, int, float], ...]
    terminal_frozen: tuple[tuple[int, int, float], ...]
    terminal_repeated: tuple[tuple[int, int, int, int, float], ...]
    visual_artifacts: tuple[tuple[str, int, int, float, float], ...]
    temporal_discontinuities: tuple[tuple[int, int, float, float], ...]


def _runs(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    start: int | None = None
    for index, enabled in enumerate(mask):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def frame_scores(signatures: np.ndarray) -> np.ndarray:
    """Return mean absolute gray-level differences between adjacent signatures."""
    if len(signatures) < 2:
        return np.empty(0, dtype=np.float32)
    axes = tuple(range(1, signatures.ndim))
    return np.mean(np.abs(signatures[1:] - signatures[:-1]), axis=axes)


def _block_difference(first: np.ndarray, second: np.ndarray, blocks: int = 4) -> np.ndarray:
    """Return block MAE values without requiring dimensions divisible by the grid size."""
    height, width = first.shape[:2]
    row_edges = np.linspace(0, height, min(blocks, height) + 1, dtype=int)
    column_edges = np.linspace(0, width, min(blocks, width) + 1, dtype=int)
    values = []
    difference = np.abs(first - second)
    for row_start, row_end in pairwise(row_edges):
        for column_start, column_end in pairwise(column_edges):
            values.append(float(np.mean(difference[row_start:row_end, column_start:column_end])))
    return np.asarray(values, dtype=np.float32)


def _robust_high_threshold(values: np.ndarray, minimum: float) -> float:
    if not values.size:
        return minimum
    median = float(np.median(values))
    median_absolute_deviation = float(np.median(np.abs(values - median)))
    return max(minimum, median + 6.0 * 1.4826 * median_absolute_deviation)


def _state_motion_support(
    states: np.ndarray | None,
    frame_count: int,
    threshold: float,
    radius: int,
) -> np.ndarray:
    """Return frames whose visual change is supported by robot-state acceleration."""
    support = np.zeros(frame_count, dtype=bool)
    if states is None or len(states) != frame_count or frame_count < 3:
        return support
    values = np.asarray(states, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    else:
        values = values.reshape(frame_count, -1)
    finite_columns = np.all(np.isfinite(values), axis=0)
    values = values[:, finite_columns]
    if not values.size:
        return support
    scale = np.percentile(values, 95, axis=0) - np.percentile(values, 5, axis=0)
    varying_columns = scale > np.finfo(np.float32).eps
    if not np.any(varying_columns):
        return support
    normalized = values[:, varying_columns] / scale[varying_columns]
    acceleration = np.zeros(frame_count, dtype=np.float64)
    acceleration[1:-1] = np.mean(
        np.abs(normalized[:-2] - 2.0 * normalized[1:-1] + normalized[2:]),
        axis=1,
    )
    motion_support = acceleration >= threshold
    expanded = motion_support.copy()
    for offset in range(1, radius + 1):
        expanded[offset:] |= motion_support[:-offset]
        expanded[:-offset] |= motion_support[offset:]
    return expanded


def analyze_signatures(
    signatures: np.ndarray,
    fps: int,
    config: AnalysisConfig,
    states: np.ndarray | None = None,
) -> SignatureAnalysis:
    """Find freeze-then-jump and repeated-motion-then-jump patterns."""
    scores = frame_scores(signatures)
    moving_scores = scores[scores > config.duplicate_threshold]
    if moving_scores.size:
        jump_threshold = max(
            config.jump_min_score,
            float(np.percentile(moving_scores, config.jump_percentile)),
        )
    else:
        jump_threshold = config.jump_min_score

    frozen: list[tuple[int, int, float]] = []
    terminal_frozen: list[tuple[int, int, float]] = []
    min_freeze_frames = max(2, math.ceil(config.freeze_min_seconds * fps))
    for transition_start, transition_end in _runs(scores <= config.duplicate_threshold):
        frozen_frames = transition_end - transition_start + 1
        if frozen_frames < min_freeze_frames:
            continue
        if transition_end >= len(scores):
            terminal_frozen.append((transition_start, transition_end, 0.0))
            continue
        jump_score = float(scores[transition_end])
        if jump_score >= jump_threshold:
            frozen.append((transition_start, transition_end, jump_score))

    repeated: list[tuple[int, int, int, int, float]] = []
    terminal_repeated: list[tuple[int, int, int, int, float]] = []
    max_period = min(
        max(2, math.floor(config.repeat_max_period_seconds * fps)),
        max(2, len(signatures) // config.repeat_min_cycles),
    )
    axes = tuple(range(1, signatures.ndim))
    for period in range(2, max_period + 1):
        offset_scores = np.mean(np.abs(signatures[period:] - signatures[:-period]), axis=axes)
        matches = offset_scores <= config.repeat_threshold
        required_matches = period * (config.repeat_min_cycles - 1)
        for match_start, match_end in _runs(matches):
            if match_end - match_start < required_matches:
                continue
            repeated_end = match_end + period
            if repeated_end > len(signatures):
                continue
            first_period_scores = scores[match_start : match_start + period - 1]
            if (
                not first_period_scores.size
                or np.max(first_period_scores) <= config.duplicate_threshold
            ):
                continue
            cycles = (repeated_end - match_start) // period
            is_terminal = repeated_end == len(signatures)
            jump_score = 0.0 if is_terminal else float(scores[repeated_end - 1])
            if not is_terminal and jump_score < jump_threshold:
                continue
            candidate = (match_start, repeated_end - 1, period, cycles, jump_score)
            candidates = terminal_repeated if is_terminal else repeated
            if any(
                existing[0] <= candidate[0] <= existing[1]
                or candidate[0] <= existing[0] <= candidate[1]
                for existing in candidates
            ):
                continue
            candidates.append(candidate)

    motion_support = _state_motion_support(
        states,
        len(signatures),
        config.state_motion_support_threshold,
        config.artifact_max_duration_frames,
    )
    visual_artifacts: list[tuple[str, int, int, float, float]] = []
    spatial_std = np.std(signatures, axis=tuple(range(1, signatures.ndim)))
    flat_mask = spatial_std <= config.flat_frame_std_threshold
    for start, end in _runs(flat_mask):
        visual_artifacts.append(
            (
                "low_information_frames",
                start,
                end - 1,
                float(config.flat_frame_std_threshold - np.min(spatial_std[start:end])),
                config.flat_frame_std_threshold,
            )
        )

    artifact_scores = np.zeros(len(signatures), dtype=np.float32)
    for frame_index in range(1, len(signatures) - 1):
        if flat_mask[frame_index]:
            continue
        current = signatures[frame_index]
        maximum_radius = min(
            config.artifact_max_duration_frames,
            frame_index,
            len(signatures) - frame_index - 1,
        )
        for radius in range(1, maximum_radius + 1):
            previous = signatures[frame_index - radius]
            following = signatures[frame_index + radius]
            previous_difference = _block_difference(previous, current)
            following_difference = _block_difference(current, following)
            neighbor_difference = _block_difference(previous, following)
            minimum_side_difference = np.minimum(previous_difference, following_difference)
            corrupted_blocks = (
                (previous_difference >= config.artifact_block_threshold)
                & (following_difference >= config.artifact_block_threshold)
                & (neighbor_difference <= minimum_side_difference * 0.5)
            )
            artifact_scores[frame_index] = max(
                artifact_scores[frame_index], float(np.mean(corrupted_blocks))
            )
    artifact_mask = (artifact_scores >= config.artifact_min_block_fraction) & ~motion_support
    for start, end in _runs(artifact_mask):
        visual_artifacts.append(
            (
                "transient_visual_corruption",
                start,
                end - 1,
                float(np.max(artifact_scores[start:end])),
                config.artifact_min_block_fraction,
            )
        )

    temporal_scores = np.zeros(len(signatures), dtype=np.float32)
    if len(signatures) >= 3:
        axes = tuple(range(1, signatures.ndim))
        temporal_scores[1:-1] = np.mean(
            np.abs(signatures[:-2] - 2.0 * signatures[1:-1] + signatures[2:]),
            axis=axes,
        )
    temporal_threshold = _robust_high_threshold(
        temporal_scores[1:-1], config.temporal_discontinuity_threshold
    )
    temporal_mask = temporal_scores >= temporal_threshold
    temporal_mask[motion_support] = False
    visual_anomaly_mask = artifact_mask | flat_mask
    temporal_exclusion_mask = visual_anomaly_mask.copy()
    temporal_exclusion_mask[1:] |= visual_anomaly_mask[:-1]
    temporal_exclusion_mask[:-1] |= visual_anomaly_mask[1:]
    temporal_mask[temporal_exclusion_mask] = False
    for start, end, *_ in (*repeated, *terminal_repeated):
        temporal_mask[start : end + 1] = False
    temporal_discontinuities = tuple(
        (start, end - 1, float(np.max(temporal_scores[start:end])), temporal_threshold)
        for start, end in _runs(temporal_mask)
    )

    return SignatureAnalysis(
        jump_threshold,
        tuple(frozen),
        tuple(repeated),
        tuple(terminal_frozen),
        tuple(terminal_repeated),
        tuple(visual_artifacts),
        temporal_discontinuities,
    )


def _load_episode_states(
    dataset_path: Path,
    info: dict[str, Any],
    episode_indices: set[int],
) -> dict[int, np.ndarray]:
    """Load state rows used to distinguish robot motion from camera-only anomalies."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    state_feature = "observation.state"
    feature = info.get("features", {}).get(state_feature)
    if not isinstance(feature, dict) or feature.get("dtype") == "video":
        return {}
    data_files = sorted((dataset_path / "data").glob("**/*.parquet"))
    if not data_files:
        return {}
    tables = [
        pq.read_table(path, columns=["episode_index", "frame_index", state_feature])
        for path in data_files
    ]
    rows = pa.concat_tables(tables).to_pylist()
    grouped: dict[int, list[tuple[int, Any]]] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_index in episode_indices:
            grouped.setdefault(episode_index, []).append(
                (int(row["frame_index"]), row[state_feature])
            )
    return {
        episode_index: np.asarray([state for _, state in sorted(samples)], dtype=np.float32)
        for episode_index, samples in grouped.items()
    }


def _load_layout(
    dataset_path: Path,
    selected_features: Sequence[str] | None,
    max_episodes: int | None,
) -> tuple[dict[str, Any], list[VideoSegment]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = int(info["fps"])
    if fps <= 0:
        raise ValueError("dataset FPS must be greater than zero")
    video_features = tuple(
        key for key, value in info.get("features", {}).items() if value.get("dtype") == "video"
    )
    if not video_features:
        raise ValueError("dataset has no video features")
    if selected_features:
        unknown = set(selected_features).difference(video_features)
        if unknown:
            raise ValueError(f"unknown video features: {', '.join(sorted(unknown))}")
        video_features = tuple(selected_features)

    episode_files = sorted((dataset_path / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise ValueError("episode metadata parquet files were not found")
    tables = [pq.read_table(path) for path in episode_files]
    rows = pa.concat_tables(tables).to_pylist()
    rows.sort(key=lambda row: int(row["episode_index"]))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be greater than zero")
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError("dataset has no episodes to check")

    video_template = info["video_path"]
    segments: list[VideoSegment] = []
    for row in rows:
        length = int(row["length"])
        episode_index = int(row["episode_index"])
        for feature in video_features:
            prefix = f"videos/{feature}"
            chunk_index = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            from_timestamp = float(row[f"{prefix}/from_timestamp"])
            relative_path = video_template.format(
                video_key=feature,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            segments.append(
                VideoSegment(
                    feature=feature,
                    episode_index=episode_index,
                    video_path=dataset_path / relative_path,
                    start_frame=round(from_timestamp * fps),
                    length=length,
                    fps=fps,
                )
            )
    return info, segments


def _segment_signatures(
    video_path: Path,
    segments: Sequence[VideoSegment],
    thumbnail_size: int,
) -> Iterator[tuple[VideoSegment, np.ndarray]]:
    import av
    import cv2

    ordered = sorted(segments, key=lambda segment: segment.start_frame)
    if not video_path.is_file():
        raise ValueError(f"video file not found: {video_path}")
    for previous, current in pairwise(ordered):
        if current.start_frame < previous.end_frame:
            raise ValueError(f"overlapping episode ranges in {video_path}")

    segment_index = 0
    current_signatures: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            while segment_index < len(ordered) and frame_index >= ordered[segment_index].end_frame:
                segment = ordered[segment_index]
                if len(current_signatures) != segment.length:
                    raise ValueError(
                        f"episode {segment.episode_index} in {video_path} decoded "
                        f"{len(current_signatures)} frames, expected {segment.length}"
                    )
                yield segment, np.stack(current_signatures)
                segment_index += 1
                current_signatures = []
            if segment_index >= len(ordered):
                break
            segment = ordered[segment_index]
            if frame_index < segment.start_frame:
                continue
            gray = frame.to_ndarray(format="gray")
            thumbnail = cv2.resize(
                gray,
                (thumbnail_size, thumbnail_size),
                interpolation=cv2.INTER_AREA,
            )
            current_signatures.append(thumbnail.astype(np.float32))

    while segment_index < len(ordered):
        segment = ordered[segment_index]
        if len(current_signatures) != segment.length:
            raise ValueError(
                f"episode {segment.episode_index} in {video_path} decoded "
                f"{len(current_signatures)} frames, expected {segment.length}"
            )
        yield segment, np.stack(current_signatures)
        segment_index += 1
        current_signatures = []


def check_dataset_videos(
    dataset_path: Path,
    config: AnalysisConfig,
    selected_features: Sequence[str] | None = None,
    max_episodes: int | None = None,
) -> VideoReport:
    """Load a local V3 dataset and return a doctor-style video report."""
    root = dataset_path.expanduser().resolve()
    info, segments = _load_layout(root, selected_features, max_episodes)
    episode_states = _load_episode_states(
        root, info, {segment.episode_index for segment in segments}
    )
    episode_count = len({segment.episode_index for segment in segments})
    feature_count = len({segment.feature for segment in segments})
    checks = [
        CheckResult("Dataset Metadata & Episode Boundaries"),
        CheckResult("Video Decode & Frame Counts"),
        CheckResult("Frozen Frames Followed by Jump"),
        CheckResult("Repeated Motion Followed by Jump"),
        CheckResult("Visual Frame Integrity"),
        CheckResult("Temporal Continuity"),
    ]
    (
        metadata_check,
        decode_check,
        freeze_check,
        repeat_check,
        visual_check,
        temporal_check,
    ) = checks
    boundary_count = max(0, episode_count - 1) * feature_count
    metadata_check.add(
        "PASS",
        f"Loaded {episode_count} episode(s) and {feature_count} video feature(s); "
        f"skipping {boundary_count} cross-episode transition(s)",
    )

    grouped: dict[Path, list[VideoSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.video_path, []).append(segment)

    findings: list[Finding] = []
    decoded_segments = 0
    try:
        for video_path, file_segments in grouped.items():
            for segment, signatures in _segment_signatures(
                video_path, file_segments, config.thumbnail_size
            ):
                decoded_segments += 1
                analysis = analyze_signatures(
                    signatures,
                    segment.fps,
                    config,
                    episode_states.get(segment.episode_index),
                )
                for start, end, jump_score in analysis.frozen:
                    findings.append(
                        Finding(
                            kind="freeze_then_jump",
                            severity="FAIL",
                            feature=segment.feature,
                            episode_index=segment.episode_index,
                            start_frame=start,
                            end_frame=end,
                            start_time_s=start / segment.fps,
                            end_time_s=end / segment.fps,
                            jump_score=jump_score,
                            jump_threshold=analysis.jump_threshold,
                        )
                    )
                for start, end, jump_score in analysis.terminal_frozen:
                    findings.append(
                        Finding(
                            kind="freeze_at_episode_end",
                            severity="FAIL",
                            feature=segment.feature,
                            episode_index=segment.episode_index,
                            start_frame=start,
                            end_frame=end,
                            start_time_s=start / segment.fps,
                            end_time_s=end / segment.fps,
                            jump_score=jump_score,
                            jump_threshold=analysis.jump_threshold,
                        )
                    )
                for start, end, period, cycles, jump_score in analysis.repeated:
                    findings.append(
                        Finding(
                            kind="repeated_motion_then_jump",
                            severity="FAIL",
                            feature=segment.feature,
                            episode_index=segment.episode_index,
                            start_frame=start,
                            end_frame=end,
                            start_time_s=start / segment.fps,
                            end_time_s=end / segment.fps,
                            jump_score=jump_score,
                            jump_threshold=analysis.jump_threshold,
                            period_frames=period,
                            cycles=cycles,
                        )
                    )
                for start, end, period, cycles, jump_score in analysis.terminal_repeated:
                    findings.append(
                        Finding(
                            kind="repeated_motion_at_episode_end",
                            severity="FAIL",
                            feature=segment.feature,
                            episode_index=segment.episode_index,
                            start_frame=start,
                            end_frame=end,
                            start_time_s=start / segment.fps,
                            end_time_s=end / segment.fps,
                            jump_score=jump_score,
                            jump_threshold=analysis.jump_threshold,
                            period_frames=period,
                            cycles=cycles,
                        )
                    )
                for kind, start, end, score, threshold in analysis.visual_artifacts:
                    findings.append(
                        Finding(
                            kind=kind,
                            severity="WARN",
                            feature=segment.feature,
                            episode_index=segment.episode_index,
                            start_frame=start,
                            end_frame=end,
                            start_time_s=start / segment.fps,
                            end_time_s=end / segment.fps,
                            jump_score=score,
                            jump_threshold=threshold,
                        )
                    )
                for start, end, score, threshold in analysis.temporal_discontinuities:
                    findings.append(
                        Finding(
                            kind="temporal_discontinuity",
                            severity="WARN",
                            feature=segment.feature,
                            episode_index=segment.episode_index,
                            start_frame=start,
                            end_frame=end,
                            start_time_s=start / segment.fps,
                            end_time_s=end / segment.fps,
                            jump_score=score,
                            jump_threshold=threshold,
                        )
                    )
    except Exception as error:
        decode_check.add("FAIL", str(error))

    if decode_check.severity == "PASS":
        decode_check.add(
            "PASS",
            f"Decoded all {decoded_segments} episode/video segment(s) with expected frame counts",
        )

    frozen_findings = [
        finding
        for finding in findings
        if finding.kind in {"freeze_then_jump", "freeze_at_episode_end"}
    ]
    repeated_findings = [
        finding
        for finding in findings
        if finding.kind in {"repeated_motion_then_jump", "repeated_motion_at_episode_end"}
    ]
    visual_findings = [
        finding
        for finding in findings
        if finding.kind in {"low_information_frames", "transient_visual_corruption"}
    ]
    temporal_findings = [
        finding for finding in findings if finding.kind == "temporal_discontinuity"
    ]
    if frozen_findings:
        freeze_check.add(
            "FAIL",
            f"Detected {len(frozen_findings)} long frozen interval(s)",
        )
    else:
        freeze_check.add("PASS", "No long frozen intervals detected within episodes")
    if repeated_findings:
        repeat_check.add(
            "FAIL",
            f"Detected {len(repeated_findings)} repeated-motion interval(s)",
        )
    else:
        repeat_check.add("PASS", "No repeated-motion intervals detected within episodes")
    if visual_findings:
        visual_check.add(
            "WARN", f"Detected {len(visual_findings)} suspicious visual artifact interval(s)"
        )
    else:
        visual_check.add("PASS", "No suspicious visual artifacts detected within episodes")
    if temporal_findings:
        temporal_check.add(
            "WARN", f"Detected {len(temporal_findings)} temporal discontinuity interval(s)"
        )
    else:
        temporal_check.add("PASS", "No temporal discontinuities detected within episodes")

    if len(findings) > config.max_findings:
        findings = findings[: config.max_findings]
        temporal_check.add("WARN", f"Findings were truncated to {config.max_findings}")

    return VideoReport(
        dataset_path=str(root),
        dataset_name=root.name,
        codebase_version=info.get("codebase_version"),
        format_version="v3",
        total_episodes=episode_count,
        total_frames=sum(
            segment.length for segment in segments if segment.feature == segments[0].feature
        ),
        fps=int(info["fps"]),
        checks=checks,
        findings=findings,
        config=config,
    )


def report_to_dict(report: VideoReport) -> dict[str, Any]:
    return {
        "version": VERSION,
        "tool": "lerobot-video-check",
        "dataset_path": report.dataset_path,
        "dataset_name": report.dataset_name,
        "codebase_version": report.codebase_version,
        "format_version": report.format_version,
        "total_episodes": report.total_episodes,
        "total_frames": report.total_frames,
        "fps": report.fps,
        "overall_severity": report.overall_severity,
        "checks": [asdict(check) for check in report.checks],
        "summary": report.summary,
        "findings": [asdict(finding) for finding in report.findings],
        "config": asdict(report.config),
    }


def report_to_json(report: VideoReport) -> str:
    return json.dumps(report_to_dict(report), indent=2)


def report_to_markdown(report: VideoReport) -> str:
    lines = [
        "# lerobot-video-check report",
        "",
        f"- **Version:** {VERSION}",
        f"- **Dataset:** `{report.dataset_path}`",
        f"- **Codebase:** {report.codebase_version}",
        f"- **Format:** {report.format_version}",
        f"- **Episodes:** {report.total_episodes}",
        f"- **Frames:** {report.total_frames:,}",
        f"- **FPS:** {report.fps}",
        f"- **Overall:** **{report.overall_severity}**",
        "",
        "**Summary:** "
        + " | ".join(f"{count} {severity}" for severity, count in report.summary.items() if count),
        "",
        "## Checks",
        "",
        "| Check | Severity | Messages |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        messages = [
            message["message"] for message in check.messages if message["severity"] != "PASS"
        ]
        cell = "<br>".join(f"- {message}" for message in messages) if messages else "_clean_"
        lines.append(f"| {check.name} | **{check.severity}** | {cell} |")
    lines.extend(["", "## Findings", ""])
    if not report.findings:
        lines.append("No temporal video anomalies were detected.")
    else:
        lines.extend(
            [
                (
                    "| Kind | Feature | Episode | Frames | Time (s) | Jump / Threshold | "
                    "Period / Cycles |"
                ),
                "| --- | --- | ---: | --- | --- | --- | --- |",
            ]
        )
        for finding in report.findings:
            period = (
                f"{finding.period_frames} / {finding.cycles}"
                if finding.period_frames is not None
                else "-"
            )
            lines.append(
                f"| {finding.kind} | `{finding.feature}` | {finding.episode_index} | "
                f"{finding.start_frame}-{finding.end_frame} | "
                f"{finding.start_time_s:.3f}-{finding.end_time_s:.3f} | "
                f"{finding.jump_score:.4f} / {finding.jump_threshold:.4f} | {period} |"
            )
    lines.extend(
        [
            "",
            "Episode boundaries are excluded from all transition checks.",
            "",
            "_Generated by lerobot-video-check._",
        ]
    )
    return "\n".join(lines)


def print_report(report: VideoReport, verbose: bool) -> None:
    print(f"lerobot-video-check v{VERSION} -- Temporal Video Report")
    print(f"Dataset: {report.dataset_path} ({report.codebase_version})")
    print(
        f"Episodes: {report.total_episodes} | Frames: {report.total_frames:,} | FPS: {report.fps}"
    )
    print()
    for check in report.checks:
        print(f"[{check.severity}] {check.name}")
        for message in check.messages:
            if verbose or message["severity"] != "PASS":
                print(f"  - {message['message']}")
        print()
    for finding in report.findings:
        period = (
            f", period={finding.period_frames}, cycles={finding.cycles}"
            if finding.period_frames is not None
            else ""
        )
        print(
            f"  {finding.kind}: {finding.feature}, episode={finding.episode_index}, "
            f"frames={finding.start_frame}-{finding.end_frame}{period}, "
            f"jump={finding.jump_score:.4f}/{finding.jump_threshold:.4f}"
        )
    counts = report.summary
    print(f"Summary: {counts['PASS']} PASS | {counts['WARN']} WARN | {counts['FAIL']} FAIL")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-video-check",
        description="Check local LeRobotDataset v3 videos for temporal anomalies.",
    )
    parser.add_argument("dataset", type=Path)
    add_validation_config_argument(parser)
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
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--fail-on", choices=("warn", "fail"), default="fail")
    parser.add_argument("--markdown", type=Path, metavar="PATH")
    parser.add_argument("--json-file", type=Path, metavar="PATH")
    return parser


def _validate_config(config: AnalysisConfig) -> None:
    if config.thumbnail_size < 8:
        raise ValueError("thumbnail_size must be at least 8")
    if config.duplicate_threshold < 0 or config.repeat_threshold < 0:
        raise ValueError("frame difference thresholds must not be negative")
    if config.freeze_min_seconds <= 0 or config.repeat_max_period_seconds <= 0:
        raise ValueError("duration thresholds must be greater than zero")
    if config.repeat_min_cycles < 2:
        raise ValueError("repeat_min_cycles must be at least 2")
    if not 0 < config.jump_percentile <= 100:
        raise ValueError("jump_percentile must be in (0, 100]")
    if config.jump_min_score < 0 or config.max_findings <= 0:
        raise ValueError("jump_min_score must not be negative and max_findings must be positive")
    if config.artifact_block_threshold <= 0:
        raise ValueError("artifact_block_threshold must be greater than zero")
    if not 0 < config.artifact_min_block_fraction <= 1:
        raise ValueError("artifact_min_block_fraction must be in (0, 1]")
    if config.artifact_max_duration_frames <= 0:
        raise ValueError("artifact_max_duration_frames must be greater than zero")
    if config.flat_frame_std_threshold < 0:
        raise ValueError("flat_frame_std_threshold must not be negative")
    if config.temporal_discontinuity_threshold <= 0:
        raise ValueError("temporal_discontinuity_threshold must be greater than zero")
    if config.state_motion_support_threshold < 0:
        raise ValueError("state_motion_support_threshold must not be negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, _, _ = parse_with_validation_profile(parser, argv, video_evaluator_argument_defaults)
    config = AnalysisConfig(
        thumbnail_size=args.thumbnail_size,
        duplicate_threshold=args.duplicate_threshold,
        freeze_min_seconds=args.freeze_min_seconds,
        repeat_threshold=args.repeat_threshold,
        repeat_min_cycles=args.repeat_min_cycles,
        repeat_max_period_seconds=args.repeat_max_period_seconds,
        jump_percentile=args.jump_percentile,
        jump_min_score=args.jump_min_score,
        artifact_block_threshold=args.artifact_block_threshold,
        artifact_min_block_fraction=args.artifact_min_block_fraction,
        artifact_max_duration_frames=args.artifact_max_duration_frames,
        flat_frame_std_threshold=args.flat_frame_std_threshold,
        temporal_discontinuity_threshold=args.temporal_discontinuity_threshold,
        state_motion_support_threshold=args.state_motion_support_threshold,
        max_findings=args.max_findings,
    )
    try:
        _validate_config(config)
        features = (
            tuple(part.strip() for part in args.features.split(",")) if args.features else None
        )
        report = check_dataset_videos(args.dataset, config, features, args.max_episodes)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(report_to_markdown(report), encoding="utf-8")
            print(f"Wrote markdown report to {args.markdown}", file=sys.stderr)
        if args.json_file:
            args.json_file.parent.mkdir(parents=True, exist_ok=True)
            args.json_file.write_text(report_to_json(report), encoding="utf-8")
            print(f"Wrote JSON report to {args.json_file}", file=sys.stderr)
    except Exception as error:
        print(f"Error checking dataset videos: {error}", file=sys.stderr)
        return 1

    if args.ci:
        print(report_to_json(report))
        counts = report.summary
        print(
            f"lerobot-video-check: {counts['PASS']} pass, "
            f"{counts['WARN']} warn, {counts['FAIL']} fail",
            file=sys.stderr,
        )
    elif args.json_output:
        print(report_to_json(report))
    else:
        print_report(report, args.verbose)

    threshold = args.fail_on.upper()
    if threshold == "WARN" and report.overall_severity in ("WARN", "FAIL"):
        return 1
    if report.overall_severity == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
