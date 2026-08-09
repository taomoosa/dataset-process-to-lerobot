# dataset-process-to-lerobot

Tools for recording ROS 2 teleoperation episodes, converting rosbag2 recordings to
LeRobotDataset V3, and validating the resulting datasets.

The project is offline-first. Conversion and local validation set the Hugging Face offline
environment variables before importing LeRobot and never upload datasets.

## Components

The Python package is divided into four independent areas:

| Area | Package | Purpose |
| --- | --- | --- |
| Conversion | `dataset_process_to_lerobot.conversion` | Convert one or more rosbag2 directories to LeRobotDataset V3 and validate camera input before video encoding. |
| Validation | `dataset_process_to_lerobot.validation` | Run `lerobot-doctor`, inspect encoded videos, and create a filtered dataset without failed episodes. |
| Recording | `dataset_process_to_lerobot.recording` | Record teleoperation topics into saveable or discardable rosbag2 sessions. |
| Mock publishers | `dataset_process_to_lerobot.mock_publishers` | Publish configurable mock RGB images, seven-axis actions, and seven-axis joint positions. |

## Requirements

- ROS 2 Jazzy or another compatible ROS 2 distribution
- Python 3.10 or newer
- `colcon`
- LeRobot with Dataset V3 support
- `lerobot-doctor`
- PyAV and OpenCV for encoded-video validation

The ROS package manifest declares the ROS dependencies. LeRobot and `lerobot-doctor` are kept
outside the ROS package dependencies because projects commonly install them in a dedicated Python
environment. No network access is required while converting or checking local data.

## Build

Clone this repository into a ROS 2 workspace and build it:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd "$ROS_WORKSPACE"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select dataset_process_to_lerobot
source install/setup.bash
```

If LeRobot is installed in a virtual environment, create that environment with access to the ROS 2
system packages, then install this repository in editable mode:

```bash
python -m pip install -e "$ROS_WORKSPACE/src/dataset-process-to-lerobot"
```

The dataset commands shown below are installed by this editable installation. Run them with the
virtual environment activated so they use the same LeRobot installation.

## 1. Publish mock topics

The default configuration publishes two cameras and two robots:

```bash
ros2 launch dataset_process_to_lerobot mock_publishers.launch.py
```

Each camera publishes its camera ID and current ROS timestamp in an RGB image. Each robot publishes
six arm axes plus one gripper axis. Edit `config/mock_publishers.yaml` or pass another parameter file:

```bash
ros2 launch dataset_process_to_lerobot mock_publishers.launch.py \
  config_file:=/path/to/mock_publishers.yaml
```

The camera implementation uses a stream abstraction so depth or other image streams can be added
without changing the publishing loop.

## 2. Record teleoperation episodes

For interactive recording, start the keyboard controller after the teleoperation topics are
available:

```bash
record-teleop-episodes \
  --output-dir recordings/day_01 \
  --session-prefix teleop \
  --storage-id sqlite3
```

The controller launches the recorder and accepts single-key commands:

- `r`: start a new episode
- `s`: stop and save the active episode
- `d`: stop and discard the active failed episode
- `i`: show recorder status
- `q`: quit; an active episode is discarded before shutdown

Each saved episode is a separate rosbag2 directory below `--output-dir`. Use `--connect-only` to
control a recorder that is already running.

The recorder can also be started directly:

```bash
ros2 launch dataset_process_to_lerobot teleop_recorder.launch.py \
  output_directory:=bags
```

Control a manual recording with ROS services:

```bash
ros2 service call /teleop_bag_recorder/start std_srvs/srv/Trigger '{}'
ros2 service call /teleop_bag_recorder/stop std_srvs/srv/Trigger '{}'
```

Use `discard` instead of `stop` to delete the active failed attempt:

```bash
ros2 service call /teleop_bag_recorder/discard std_srvs/srv/Trigger '{}'
```

Automatic recording is available for repeatable tests:

```bash
ros2 launch dataset_process_to_lerobot mock_recording.launch.py \
  output_directory:=bags \
  auto_enabled:=true \
  session_count:=3 \
  duration_sec:=10.0 \
  exit_when_done:=true
```

## 3. Convert rosbag2 to LeRobotDataset V3

Each input bag becomes one episode. The converter uses zero-order hold to align all topics to the
requested output FPS and supports per-bag task descriptions. A recording directory can be supplied
instead of listing every episode:

```bash
rosbag-to-lerobot \
  --bag-dir recordings/day_01 \
  --bag-dir recordings/day_02 \
  --output-dir datasets/teleop_v3 \
  --repo-id local/teleop_v3 \
  --fps 10 \
  --task "operate both robots" \
  --input-validation drop \
  --input-validation-drop-on fail \
  --input-validation-report reports/input-validation.json \
  --conversion-manifest reports/conversion-manifest.json
```

Only direct child directories are discovered by default. Add `--recursive` when collection
directories contain another level of grouping. Explicit bag paths remain supported:

```bash
rosbag-to-lerobot \
  bags/episode_001 \
  bags/episode_002 \
  --output-dir datasets/teleop_v3 \
  --repo-id local/teleop_v3 \
  --fps 10 \
  --task "operate both robots" \
  --task-map "episode_002=return both robots" \
  --input-validation warn \
  --input-validation-report reports/input-validation.json
```

The converter refuses to overwrite an existing output directory. Camera input validation separates
intentional output-frame reuse from duplicate source messages:

```text
source_index:  10 10 11 11 12 12
resampled:      F  T  F  T  F  T
fingerprint:    A  A  A  A  B  B
```

The second copy of each source index is expected resampling. The matching fingerprints for source
indices 10 and 11 are reported as `duplicate_source_frames`. A content-only duplicate is a warning
because a stationary scene can produce the same pixels. A duplicate with a stalled source timestamp,
a source timestamp regression, an excessive camera input gap, or fewer than
`--minimum-camera-source-frames` source images is a failure. Gaps at least
`--source-drop-factor` times the median camera period are reported as probable short frame drops,
even when they are shorter than the long-gap threshold.

Use `--input-validation fail` to stop before encoding when any finding is present. The partial output
dataset is removed, while an explicitly requested validation report is retained. Use
`--input-validation drop` to discard failed episodes before MP4 encoding and continue with the
remaining input bags. Warning-only episodes are retained by default because an intentional static
scene can produce duplicate-frame warnings. Set `--input-validation-drop-on warn` to restore strict
removal of every episode with a finding. If every episode is rejected, conversion fails instead of
creating an empty dataset.

## 4. Validate a converted dataset

Each evaluator implements the same automation interface:

```text
EVALUATOR DATASET --report-dir DIRECTORY --result-file FILE --fail-on {warn,fail}
```

It writes an `lerobot-dataset-evaluation/v1` JSON result and returns `0` for a pass, `10` for
episode-local findings, or `20` for a blocker that episode deletion cannot fix. The normalized
result always contains `deletable_episode_indices` and `non_episode_blockers`, so filtering does not
need evaluator-specific parsing.

A minimal passing result has this shape:

```json
{
  "version": "1.0",
  "contract": "lerobot-dataset-evaluation/v1",
  "evaluator": "example-check",
  "dataset_path": "/absolute/path/to/dataset",
  "fail_on": "FAIL",
  "status": "pass",
  "overall_severity": "PASS",
  "deletable_episode_indices": [],
  "findings": [],
  "non_episode_blockers": [],
  "artifacts": {},
  "evaluator_return_code": 0
}
```

Run the external `lerobot-doctor` command through its adapter:

```bash
evaluate-lerobot-doctor \
  datasets/teleop_v3 \
  --report-dir reports/teleop_v3/doctor \
  --result-file reports/teleop_v3/doctor/evaluation-result.json \
  --fail-on fail
```

Run the encoded-video evaluator through the same interface:

```bash
evaluate-lerobot-video \
  datasets/teleop_v3 \
  --report-dir reports/teleop_v3/video \
  --result-file reports/teleop_v3/video/evaluation-result.json \
  --fail-on fail \
  --freeze-min-seconds 1.0
```

`evaluate-lerobot-doctor` stores the original doctor JSON, Markdown, stdout, and stderr beside its
normalized result. `evaluate-lerobot-video` stores the original video-check JSON and Markdown. The
video checker decodes each camera with PyAV and uses OpenCV for frame signatures. It detects long
frozen intervals and repeated motion both before a jump and at the end of an episode. It also reports
low-information frames, transient whole-frame or block corruption, and temporal discontinuities
such as short out-of-order runs. Episode boundaries are loaded from LeRobotDataset V3 metadata and
excluded from transition checks.

Terminal freezes and repeated-motion loops are `FAIL` findings. Visual corruption and temporal
discontinuity findings are `WARN` because sudden legitimate motion or an intentionally dark scene
can look similar. The default `--fail-on fail` therefore records those ambiguous findings without
deleting the episode; use `--fail-on warn` after calibrating the thresholds when automatic removal is
appropriate. The relevant tuning options are `--artifact-block-threshold`,
`--artifact-min-block-fraction`, `--artifact-max-duration-frames`,
`--flat-frame-std-threshold`, and
`--temporal-discontinuity-threshold`. When `observation.state` is present, visual changes supported
by simultaneous robot-state acceleration are excluded from camera-only artifact and ordering
findings. Tune this cross-check with `--state-motion-support-threshold`. The JSON and Markdown report
structures are unchanged; these
conditions appear as additional finding `kind` values.

The compatibility command below runs both checks and combines their selections:

```bash
validate-lerobot-dataset \
  datasets/teleop_v3 \
  --report-dir reports/teleop_v3 \
  --fail-on fail
```

The command writes:

- `lerobot-doctor.md`: structure, metadata, Parquet, and file checks
- `lerobot-doctor.json`: machine-readable doctor output
- `lerobot-video-check.md`: human-readable temporal video findings
- `lerobot-video-check.json`: structured episode findings for automation
- `validation-summary.json`: combined status, deletable episode indices, and non-episode blockers

Run only the project video checker when `lerobot-doctor` is unavailable:

```bash
validate-lerobot-dataset datasets/teleop_v3 --skip-doctor
```

Automation can distinguish the result by exit status:

| Exit status | Meaning |
| ---: | --- |
| `0` | All enabled checks passed at the requested threshold |
| `10` | Episode-local findings can be passed to `remove-failed-episodes` |
| `20` | A structural problem or tool failure cannot be fixed by deleting episodes |

## 5. Remove failed episodes

The filtering command reads structured JSON reports and calls LeRobot's official
`lerobot.datasets.dataset_tools.delete_episodes()` API. It always creates a new dataset and never
modifies the source dataset.

```bash
remove-failed-episodes \
  datasets/teleop_v3 \
  --report reports/teleop_v3/lerobot-video-check.json \
  --output-dir datasets/teleop_v3_clean \
  --result-file reports/teleop_v3/filter-result.json \
  --fail-on fail
```

Use `--fail-on warn` to remove warning and failure episodes. Additional indices can be supplied with
repeatable `--episode INDEX` options. Preview the selection without importing LeRobot or writing data:

```bash
remove-failed-episodes \
  datasets/teleop_v3 \
  --report reports/teleop_v3/lerobot-video-check.json \
  --episode 4 \
  --output-dir datasets/teleop_v3_clean \
  --dry-run
```

`remove-failed-episodes` does not heuristically parse arbitrary human-readable messages.
`evaluate-lerobot-doctor` normalizes explicit `Episode N` references and complete lists written as
`episode(s): [N, ...]` into the common result contract. Aggregate warnings that do not enumerate all
affected episodes remain non-deletable; provide those indices explicitly with `--episode` after
reviewing the raw report.

The result file is written even when no episodes need removal. Its `effective_dataset` field points
to the source dataset when clean and to the newly filtered dataset otherwise. Results from
`evaluate-lerobot-doctor`, `evaluate-lerobot-video`, and any conforming custom evaluator can be used
with the same `--report` option.

## 6. Archive rosbag2 recordings

Copy one or more recording collections to archival storage and verify every copied file:

```bash
archive-rosbags \
  --bag-dir recordings/day_01 \
  --bag-dir recordings/day_02 \
  --archive-dir /path/to/archive \
  --mode copy \
  --verify sha256
```

`copy` is the default and preserves the source. `--mode move` removes each source bag only after its
temporary destination has been copied, verified, and atomically renamed. Existing destinations are
rejected unless `--existing verify` confirms they match, which supports safe retries. The command
writes `archive-manifest.json` by default.

## 7. Run the unattended workflow

`process-teleop-dataset` connects directory discovery, conversion, an ordered list of evaluator and
episode-filter stages, and archival. The default list is `doctor` followed by `video`. Each stage
receives the current effective LeRobotDataset. If it selects episodes, the workflow creates a new
dataset without them, reruns that evaluator on the new dataset, and only then proceeds. Raw bags are
archived only after every stage passes.

```bash
process-teleop-dataset \
  --bag-dir recordings/day_01 \
  --bag-dir recordings/day_02 \
  --dataset-dir work/teleop_v3 \
  --clean-dataset-dir work/teleop_v3_clean \
  --report-dir work/reports \
  --archive-dir /path/to/archive \
  --archive-mode move \
  --archive-verify sha256 \
  --repo-id local/teleop_v3 \
  --fps 10 \
  --task "operate both robots"
```

Use `--dry-run` to inspect discovered bags and resolved destinations without writing data. The
workflow writes `workflow-summary.json` after every stage. An episode-local validation failure is
filtered automatically; a structural failure stops the workflow and leaves raw bags in place. The
first filtered dataset uses `--clean-dataset-dir`; additional filtering stages use sibling paths
with the filter number and evaluator name appended. Input bags, datasets, reports, and archives must
use non-overlapping directory trees; this is checked during preparation, including in dry-run mode.

Replace the defaults with any number of contract-compatible evaluators by repeating
`--evaluation-stage NAME=COMMAND`. Commands are tokenized without a shell. Do not include the four
common arguments in `COMMAND`; the workflow appends them:

```bash
process-teleop-dataset \
  --bag-dir recordings/day_01 \
  --dataset-dir work/teleop_v3 \
  --clean-dataset-dir work/teleop_v3_clean \
  --report-dir work/reports \
  --archive-dir /path/to/archive \
  --repo-id local/teleop_v3 \
  --fps 10 \
  --task "operate both robots" \
  --evaluation-stage 'doctor=evaluate-lerobot-doctor' \
  --evaluation-stage 'video=evaluate-lerobot-video --freeze-min-seconds 1.5' \
  --evaluation-stage 'sarm=evaluate-lerobot-sarm --minimum-terminal-progress 0.8'
```

The SARM command above illustrates the extension point; it is not supplied by this repository. A
future SARM adapter only needs to accept the common arguments, write the normalized contract, and
use the standard exit statuses. Its own thresholds remain ordinary command options.

## Installed commands

The editable installation described in the Build section provides these commands:

| Command | Purpose |
| --- | --- |
| `rosbag-to-lerobot` | Convert rosbag2 episodes and validate camera input |
| `evaluate-lerobot-doctor` | Adapt `lerobot-doctor` to the evaluator contract |
| `evaluate-lerobot-video` | Adapt video checks to the evaluator contract |
| `validate-lerobot-dataset` | Run `lerobot-doctor` and encoded-video validation |
| `lerobot-video-check` | Run only the encoded-video validation |
| `remove-failed-episodes` | Create a dataset copy without selected failed episodes |
| `record-teleop-episodes` | Interactively record, save, or discard multiple episodes |
| `archive-rosbags` | Copy or move raw bags after size or SHA-256 verification |
| `process-teleop-dataset` | Run conversion through archival as one guarded workflow |

ROS 2 nodes are available through `ros2 run dataset_process_to_lerobot EXECUTABLE`. The launch
commands in the preceding sections are preferred when starting multiple nodes or loading the
provided parameter file.

The repository intentionally does not duplicate these entry points with Python files under
`scripts/`. Installing the package provides the dataset commands, and ROS 2 launch files provide
the multi-node entry points shown above.

## License

Apache License 2.0. See `LICENSE`.
