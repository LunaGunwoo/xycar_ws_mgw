"""Competition mission dataset annotations, validation, and split tooling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ANNOTATION_FILENAME = "mission_annotations.yaml"
ANNOTATION_SCHEMA_VERSION = 1
DATASET_KIND = "competition_mission_sequence"
CAPTURE_KINDS = {"signal", "shortcut"}
LAMP_NAMES = ("red", "yellow", "left_arrow", "straight_green")
SHORTCUT_PHASES = (
    "APPROACH",
    "ENTRY_STRAIGHT",
    "FIRST_LEFT",
    "CONNECTOR",
    "SECOND_LEFT",
    "REACQUIRE",
)


class CompetitionDataError(ValueError):
    """Raised when a mission dataset violates its tracked contract."""


@dataclass(frozen=True)
class MissionSample:
    sample_index: int
    image_path: Path
    angle: float
    speed: float
    timestamp_sec: float


@dataclass(frozen=True)
class MissionSession:
    root: Path
    capture_kind: str
    samples: tuple[MissionSample, ...]
    annotation: Mapping[str, Any] | None
    annotation_sha256: str | None

    @property
    def session_id(self) -> str:
        return self.root.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mission_session(
    session_dir: str | Path,
    *,
    require_approved: bool = False,
) -> MissionSession:
    root = Path(session_dir).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CompetitionDataError(f"unsafe or missing session: {root}")
    metadata_path = root / "metadata.yaml"
    samples_path = root / "samples.csv"
    if not metadata_path.is_file() or not samples_path.is_file():
        raise CompetitionDataError(
            f"session requires metadata.yaml and samples.csv: {root}"
        )
    metadata = _load_mapping(metadata_path)
    if metadata.get("complete") is not True:
        raise CompetitionDataError(f"session is not complete: {root.name}")
    if metadata.get("dataset_kind") != DATASET_KIND:
        raise CompetitionDataError(
            f"unexpected dataset_kind in {root.name}: "
            f"{metadata.get('dataset_kind')!r}"
        )
    mission = _required_mapping(metadata, "mission", "metadata")
    capture_kind = _required_string(mission, "capture_kind", "mission")
    if capture_kind not in CAPTURE_KINDS:
        raise CompetitionDataError(
            f"unsupported capture_kind in {root.name}: {capture_kind!r}"
        )
    samples = _load_samples(root, samples_path)
    annotation_path = root / ANNOTATION_FILENAME
    annotation: Mapping[str, Any] | None = None
    annotation_sha256: str | None = None
    if annotation_path.is_file():
        annotation = _load_mapping(annotation_path)
        _validate_annotation(
            annotation,
            session_id=root.name,
            capture_kind=capture_kind,
            sample_count=len(samples),
            samples_sha256=sha256_file(samples_path),
            require_approved=require_approved,
        )
        annotation_sha256 = sha256_file(annotation_path)
    elif require_approved:
        raise CompetitionDataError(
            f"approved annotation is missing: {annotation_path}"
        )
    return MissionSession(
        root=root,
        capture_kind=capture_kind,
        samples=samples,
        annotation=annotation,
        annotation_sha256=annotation_sha256,
    )


def discover_sessions(
    dataset_root: str | Path,
    *,
    require_approved: bool,
) -> tuple[MissionSession, ...]:
    root = Path(dataset_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CompetitionDataError(f"unsafe or missing dataset root: {root}")
    sessions: list[MissionSession] = []
    for candidate in sorted(root.glob("*_session*")):
        if (
            candidate.name.startswith("_recording_")
            or "_incomplete" in candidate.name
            or not candidate.is_dir()
        ):
            continue
        sessions.append(
            load_mission_session(candidate, require_approved=require_approved)
        )
    if not sessions:
        raise CompetitionDataError(f"no mission sessions found in {root}")
    return tuple(sessions)


def write_annotation_template(
    session_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    session = load_mission_session(session_dir, require_approved=False)
    path = session.root / ANNOTATION_FILENAME
    if path.exists() and not overwrite:
        raise CompetitionDataError(
            f"annotation already exists; use --overwrite: {path}"
        )
    last_index = session.samples[-1].sample_index
    payload: dict[str, Any] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "session_id": session.session_id,
        "capture_kind": session.capture_kind,
        "source_samples_sha256": sha256_file(session.root / "samples.csv"),
        "status": "draft",
        "annotator": "",
        "reviewer": "",
        "notes": "Replace null values, then validate and approve.",
        "signal": {
            "event_present": None,
            "approach_start": None,
            "decision_deadline": None,
            "exit_end": None,
            "state_segments": [
                {
                    "start": 1,
                    "end": last_index,
                    "readable": False,
                    **{name: False for name in LAMP_NAMES},
                }
            ],
            "bbox_keyframes": [],
        },
    }
    if session.capture_kind == "shortcut":
        payload["shortcut"] = {
            "active_start": None,
            "active_end": None,
            "phase_starts": [
                {"phase": phase, "start": None}
                for phase in SHORTCUT_PHASES
            ],
            "handoff_ready_start": None,
            "handoff_ready_end": None,
        }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def approve_annotation(session_dir: str | Path, *, reviewer: str) -> Path:
    reviewer = reviewer.strip()
    if not reviewer:
        raise CompetitionDataError("reviewer must not be empty")
    root = Path(session_dir).expanduser().resolve()
    path = root / ANNOTATION_FILENAME
    payload = dict(_load_mapping(path))
    payload["status"] = "approved"
    payload["reviewer"] = reviewer
    _validate_annotation(
        payload,
        session_id=root.name,
        capture_kind=_session_capture_kind(root),
        sample_count=len(_load_samples(root, root / "samples.csv")),
        samples_sha256=sha256_file(root / "samples.csv"),
        require_approved=True,
    )
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def materialize_signal_labels(session: MissionSession) -> list[dict[str, Any]]:
    annotation = _require_annotation(session)
    signal = _required_mapping(annotation, "signal", "annotation")
    count = len(session.samples)
    labels = [
        {
            "approach": 0.0,
            "visible": 0.0,
            "readable": 0.0,
            **{name: 0.0 for name in LAMP_NAMES},
            "bbox": (0.0, 0.0, 0.0, 0.0),
            "bbox_valid": 0.0,
            "progress": 0.0,
        }
        for _ in range(count)
    ]
    if signal.get("event_present") is False:
        return labels
    approach_start = _required_index(signal, "approach_start", count)
    deadline = _required_index(signal, "decision_deadline", count)
    exit_end = _required_index(signal, "exit_end", count)
    if not approach_start <= deadline <= exit_end:
        raise CompetitionDataError(
            f"invalid signal event order in {session.session_id}"
        )
    span = max(1, deadline - approach_start)
    for index in range(approach_start, exit_end + 1):
        labels[index - 1]["approach"] = 1.0
        labels[index - 1]["progress"] = min(
            1.0,
            max(0.0, (index - approach_start) / span),
        )
    for segment in _required_list(signal, "state_segments", "signal"):
        if not isinstance(segment, Mapping):
            raise CompetitionDataError("signal state segment must be a mapping")
        start = _required_index(segment, "start", count)
        end = _required_index(segment, "end", count)
        if end < start:
            raise CompetitionDataError("signal segment end precedes start")
        readable = _required_bool(segment, "readable", "signal segment")
        for index in range(start, end + 1):
            label = labels[index - 1]
            label["readable"] = float(readable)
            for name in LAMP_NAMES:
                label[name] = float(
                    _required_bool(segment, name, "signal segment")
                )
    keyframes = _bbox_keyframes(signal, count)
    if len(keyframes) < 2:
        raise CompetitionDataError(
            f"positive signal requires at least two bbox keyframes in "
            f"{session.session_id}"
        )
    if any(bbox[3] > (2.0 / 3.0) + 1e-6 for _, bbox in keyframes):
        raise CompetitionDataError(
            f"signal bbox leaves the upper-two-thirds ROI in {session.session_id}"
        )
    if keyframes:
        for left, right in zip(keyframes, keyframes[1:]):
            _interpolate_bbox(labels, left, right)
        sample_index, bbox = keyframes[-1]
        labels[sample_index - 1]["bbox"] = bbox
        labels[sample_index - 1]["bbox_valid"] = 1.0
        first_index, first_bbox = keyframes[0]
        labels[first_index - 1]["bbox"] = first_bbox
        labels[first_index - 1]["bbox_valid"] = 1.0
    for label in labels:
        label["visible"] = label["bbox_valid"]
        if label["readable"] and not label["bbox_valid"]:
            raise CompetitionDataError(
                f"readable signal lacks a bbox in {session.session_id}"
            )
        if any(label[name] for name in LAMP_NAMES) and not label["readable"]:
            raise CompetitionDataError(
                f"active lamp must be readable in {session.session_id}"
            )
    return labels


def materialize_shortcut_labels(session: MissionSession) -> list[dict[str, Any]]:
    annotation = _require_annotation(session)
    shortcut = _required_mapping(annotation, "shortcut", "annotation")
    count = len(session.samples)
    active_start = _required_index(shortcut, "active_start", count)
    active_end = _required_index(shortcut, "active_end", count)
    if active_end < active_start:
        raise CompetitionDataError("shortcut active_end precedes active_start")
    phase_starts = _phase_starts(shortcut, count)
    if phase_starts[0][1] != active_start:
        raise CompetitionDataError(
            "APPROACH phase must start at shortcut active_start"
        )
    if any(
        current[1] >= following[1]
        for current, following in zip(phase_starts, phase_starts[1:])
    ):
        raise CompetitionDataError("shortcut phase starts must increase")
    ready_start = _required_index(shortcut, "handoff_ready_start", count)
    ready_end = _required_index(shortcut, "handoff_ready_end", count)
    if not phase_starts[-1][1] <= ready_start <= ready_end <= active_end:
        raise CompetitionDataError("invalid shortcut handoff-ready range")
    if ready_end - ready_start + 1 < 5:
        raise CompetitionDataError(
            "shortcut handoff-ready range must span at least five frames"
        )
    labels: list[dict[str, Any]] = []
    phase_index = 0
    for sample in session.samples:
        index = sample.sample_index
        while (
            phase_index + 1 < len(phase_starts)
            and index >= phase_starts[phase_index + 1][1]
        ):
            phase_index += 1
        labels.append(
            {
                "active": float(active_start <= index <= active_end),
                "phase": phase_index,
                "handoff_ready": float(ready_start <= index <= ready_end),
            }
        )
    return labels


def load_split_manifest(path: str | Path) -> dict[str, tuple[str, ...]]:
    payload = _load_mapping(Path(path).expanduser().resolve())
    if payload.get("schema_version") != 1:
        raise CompetitionDataError("split schema_version must be 1")
    result: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        raw = payload.get(split)
        if not isinstance(raw, list) or not raw:
            raise CompetitionDataError(f"split {split} must be a non-empty list")
        values = tuple(str(value) for value in raw)
        overlap = seen.intersection(values)
        if overlap:
            raise CompetitionDataError(
                f"session split overlap: {sorted(overlap)}"
            )
        seen.update(values)
        result[split] = values
    return result


def make_split_manifest(
    dataset_root: str | Path,
    output: str | Path,
    *,
    seed: int,
    capture_kind: str | None = None,
) -> Path:
    sessions = discover_sessions(dataset_root, require_approved=True)
    if capture_kind is not None:
        if capture_kind not in CAPTURE_KINDS:
            raise CompetitionDataError(
                f"unsupported split capture_kind: {capture_kind}"
            )
        sessions = tuple(
            session for session in sessions if session.capture_kind == capture_kind
        )
    ordered = sorted(
        sessions,
        key=lambda session: hashlib.sha256(
            f"{seed}:{session.session_id}".encode()
        ).hexdigest(),
    )
    if len(ordered) < 3:
        raise CompetitionDataError("at least three approved sessions are required")
    validation_count = max(1, round(len(ordered) * 0.15))
    test_count = max(1, round(len(ordered) * 0.15))
    train_count = len(ordered) - validation_count - test_count
    if train_count < 1:
        raise CompetitionDataError("split would have no training session")
    payload = {
        "schema_version": 1,
        "seed": seed,
        "train": [session.session_id for session in ordered[:train_count]],
        "validation": [
            session.session_id
            for session in ordered[train_count : train_count + validation_count]
        ],
        "test": [
            session.session_id
            for session in ordered[train_count + validation_count :]
        ],
    }
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise CompetitionDataError(f"refusing to overwrite split: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def _load_samples(root: Path, samples_path: Path) -> tuple[MissionSample, ...]:
    samples: list[MissionSample] = []
    with samples_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "sample_index",
            "image",
            "angle",
            "speed",
            "camera_stamp_sec",
            "camera_stamp_nanosec",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise CompetitionDataError(
                f"samples.csv is missing fields: {sorted(required)}"
            )
        previous_index = 0
        previous_time = -math.inf
        for row in reader:
            try:
                sample_index = int(row["sample_index"])
                angle = float(row["angle"])
                speed = float(row["speed"])
                timestamp = float(row["camera_stamp_sec"]) + (
                    float(row["camera_stamp_nanosec"]) / 1_000_000_000.0
                )
            except (TypeError, ValueError) as exc:
                raise CompetitionDataError(
                    f"invalid samples.csv row in {root.name}"
                ) from exc
            if sample_index != previous_index + 1:
                raise CompetitionDataError(
                    f"non-contiguous sample_index in {root.name}: {sample_index}"
                )
            if not all(math.isfinite(value) for value in (angle, speed, timestamp)):
                raise CompetitionDataError(f"non-finite sample in {root.name}")
            if timestamp < previous_time:
                raise CompetitionDataError(
                    f"camera timestamps move backwards in {root.name}"
                )
            image_relative = Path(row["image"])
            if image_relative.is_absolute() or ".." in image_relative.parts:
                raise CompetitionDataError("unsafe image path in samples.csv")
            image_path = root / image_relative
            if image_path.is_symlink() or not image_path.is_file():
                raise CompetitionDataError(f"missing or unsafe image: {image_path}")
            samples.append(
                MissionSample(
                    sample_index=sample_index,
                    image_path=image_path,
                    angle=angle,
                    speed=speed,
                    timestamp_sec=timestamp,
                )
            )
            previous_index = sample_index
            previous_time = timestamp
    if not samples:
        raise CompetitionDataError(f"session has no samples: {root.name}")
    return tuple(samples)


def _validate_annotation(
    annotation: Mapping[str, Any],
    *,
    session_id: str,
    capture_kind: str,
    sample_count: int,
    samples_sha256: str,
    require_approved: bool,
) -> None:
    if annotation.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise CompetitionDataError("unsupported annotation schema_version")
    if annotation.get("session_id") != session_id:
        raise CompetitionDataError("annotation session_id mismatch")
    if annotation.get("capture_kind") != capture_kind:
        raise CompetitionDataError("annotation capture_kind mismatch")
    if annotation.get("source_samples_sha256") != samples_sha256:
        raise CompetitionDataError("annotation source samples checksum mismatch")
    status = annotation.get("status")
    if status not in {"draft", "approved"}:
        raise CompetitionDataError("annotation status must be draft or approved")
    if require_approved:
        if status != "approved":
            raise CompetitionDataError("training requires approved annotations")
        annotator = str(annotation.get("annotator", "")).strip()
        reviewer = str(annotation.get("reviewer", "")).strip()
        if not annotator:
            raise CompetitionDataError("approved annotation requires annotator")
        if not reviewer:
            raise CompetitionDataError("approved annotation requires reviewer")
        if annotator == reviewer:
            raise CompetitionDataError(
                "annotation reviewer must differ from the annotator"
            )
    signal = _required_mapping(annotation, "signal", "annotation")
    event_present = signal.get("event_present")
    if event_present not in {True, False, None}:
        raise CompetitionDataError("signal.event_present must be boolean or null")
    if require_approved and event_present is None:
        raise CompetitionDataError("approved signal.event_present cannot be null")
    if event_present is True:
        # Materialization performs the detailed interval checks.
        temporary = MissionSession(
            root=Path(session_id),
            capture_kind=capture_kind,
            samples=tuple(
                MissionSample(index, Path("."), 0.0, 0.0, float(index))
                for index in range(1, sample_count + 1)
            ),
            annotation=annotation,
            annotation_sha256=None,
        )
        signal_labels = materialize_signal_labels(temporary)
        if capture_kind == "shortcut" and not any(
            label["left_arrow"] > 0.5 for label in signal_labels
        ):
            raise CompetitionDataError(
                "approved shortcut session must contain a left-arrow signal"
            )
    elif capture_kind == "shortcut" and require_approved:
        raise CompetitionDataError(
            "approved shortcut session must include its left-arrow event"
        )
    if capture_kind == "shortcut" and require_approved:
        temporary = MissionSession(
            root=Path(session_id),
            capture_kind=capture_kind,
            samples=tuple(
                MissionSample(index, Path("."), 0.0, 0.0, float(index))
                for index in range(1, sample_count + 1)
            ),
            annotation=annotation,
            annotation_sha256=None,
        )
        materialize_shortcut_labels(temporary)


def _bbox_keyframes(
    signal: Mapping[str, Any],
    count: int,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    result: list[tuple[int, tuple[float, float, float, float]]] = []
    for raw in _required_list(signal, "bbox_keyframes", "signal"):
        if not isinstance(raw, Mapping):
            raise CompetitionDataError("bbox keyframe must be a mapping")
        sample_index = _required_index(raw, "sample_index", count)
        bbox_raw = raw.get("bbox")
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            raise CompetitionDataError("bbox must be [x1,y1,x2,y2]")
        try:
            bbox = tuple(float(value) for value in bbox_raw)
        except (TypeError, ValueError) as exc:
            raise CompetitionDataError("bbox values must be numeric") from exc
        if (
            not all(math.isfinite(value) for value in bbox)
            or not 0.0 <= bbox[0] < bbox[2] <= 1.0
            or not 0.0 <= bbox[1] < bbox[3] <= 1.0
        ):
            raise CompetitionDataError("bbox must be normalized and ordered")
        result.append((sample_index, bbox))
    result.sort(key=lambda item: item[0])
    if len({item[0] for item in result}) != len(result):
        raise CompetitionDataError("duplicate bbox keyframe sample_index")
    return result


def _interpolate_bbox(
    labels: list[dict[str, Any]],
    left: tuple[int, tuple[float, float, float, float]],
    right: tuple[int, tuple[float, float, float, float]],
) -> None:
    left_index, left_bbox = left
    right_index, right_bbox = right
    if right_index <= left_index:
        raise CompetitionDataError("bbox keyframes must increase")
    span = right_index - left_index
    for sample_index in range(left_index, right_index + 1):
        ratio = (sample_index - left_index) / span
        bbox = tuple(
            a + (b - a) * ratio for a, b in zip(left_bbox, right_bbox)
        )
        labels[sample_index - 1]["bbox"] = bbox
        labels[sample_index - 1]["bbox_valid"] = 1.0


def _phase_starts(
    shortcut: Mapping[str, Any],
    count: int,
) -> list[tuple[str, int]]:
    raw_values = _required_list(shortcut, "phase_starts", "shortcut")
    if len(raw_values) != len(SHORTCUT_PHASES):
        raise CompetitionDataError("all shortcut phases must be labeled")
    result: list[tuple[str, int]] = []
    for expected, raw in zip(SHORTCUT_PHASES, raw_values):
        if not isinstance(raw, Mapping) or raw.get("phase") != expected:
            raise CompetitionDataError(
                f"shortcut phase order must be {list(SHORTCUT_PHASES)}"
            )
        result.append((expected, _required_index(raw, "start", count)))
    return result


def _required_index(
    mapping: Mapping[str, Any],
    key: str,
    count: int,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompetitionDataError(f"{key} must be an integer sample index")
    if not 1 <= value <= count:
        raise CompetitionDataError(f"{key} must be in [1,{count}]")
    return value


def _required_bool(mapping: Mapping[str, Any], key: str, label: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise CompetitionDataError(f"{label}.{key} must be boolean")
    return value


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise CompetitionDataError(f"{label}.{key} must be a mapping")
    return value


def _required_list(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise CompetitionDataError(f"{label}.{key} must be a list")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise CompetitionDataError(f"{label}.{key} must be a non-empty string")
    return value


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompetitionDataError(f"missing or unsafe YAML file: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CompetitionDataError(f"invalid YAML: {path}") from exc
    if not isinstance(value, Mapping):
        raise CompetitionDataError(f"YAML root must be a mapping: {path}")
    return value


def _session_capture_kind(root: Path) -> str:
    metadata = _load_mapping(root / "metadata.yaml")
    mission = _required_mapping(metadata, "mission", "metadata")
    return _required_string(mission, "capture_kind", "mission")


def _require_annotation(session: MissionSession) -> Mapping[str, Any]:
    if session.annotation is None:
        raise CompetitionDataError(
            f"session has no annotation: {session.session_id}"
        )
    return session.annotation


def _command_init(arguments: argparse.Namespace) -> None:
    path = write_annotation_template(
        arguments.session,
        overwrite=arguments.overwrite,
    )
    print(path)


def _command_validate(arguments: argparse.Namespace) -> None:
    if arguments.session:
        sessions = (
            load_mission_session(
                arguments.session,
                require_approved=arguments.require_approved,
            ),
        )
    else:
        sessions = discover_sessions(
            arguments.dataset_root,
            require_approved=arguments.require_approved,
        )
    summary = {
        "sessions": len(sessions),
        "samples": sum(len(session.samples) for session in sessions),
        "capture_kinds": {
            kind: sum(session.capture_kind == kind for session in sessions)
            for kind in sorted(CAPTURE_KINDS)
        },
        "approved": sum(
            session.annotation is not None
            and session.annotation.get("status") == "approved"
            for session in sessions
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _command_approve(arguments: argparse.Namespace) -> None:
    print(approve_annotation(arguments.session, reviewer=arguments.reviewer))


def _command_split(arguments: argparse.Namespace) -> None:
    print(
        make_split_manifest(
            arguments.dataset_root,
            arguments.output,
            seed=arguments.seed,
            capture_kind=arguments.capture_kind,
        )
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage approved competition mission annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init", help="create a draft annotation")
    init.add_argument("--session", required=True)
    init.add_argument("--overwrite", action="store_true")
    init.set_defaults(handler=_command_init)

    validate = subparsers.add_parser("validate", help="validate sessions")
    target = validate.add_mutually_exclusive_group(required=True)
    target.add_argument("--session")
    target.add_argument("--dataset-root")
    validate.add_argument("--require-approved", action="store_true")
    validate.set_defaults(handler=_command_validate)

    approve = subparsers.add_parser("approve", help="approve reviewed labels")
    approve.add_argument("--session", required=True)
    approve.add_argument("--reviewer", required=True)
    approve.set_defaults(handler=_command_approve)

    split = subparsers.add_parser(
        "make-split",
        help="write a deterministic session-disjoint split",
    )
    split.add_argument("--dataset-root", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=20260815)
    split.add_argument("--capture-kind", choices=sorted(CAPTURE_KINDS))
    split.set_defaults(handler=_command_split)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    arguments = build_argument_parser().parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
