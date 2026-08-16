from __future__ import annotations

import csv
from pathlib import Path

import yaml
from PIL import Image

from xycar_ai.competition_data import (
    approve_annotation,
    load_mission_session,
    make_split_manifest,
    materialize_shortcut_labels,
    materialize_signal_labels,
    write_annotation_template,
)


def _write_mission_session(
    root: Path,
    name: str,
    *,
    capture_kind: str,
    count: int = 8,
) -> Path:
    session = root / name
    images = session / "Images"
    images.mkdir(parents=True)
    fields = (
        "sample_index",
        "image",
        "angle",
        "speed",
        "camera_stamp_sec",
        "camera_stamp_nanosec",
    )
    with (session / "samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(1, count + 1):
            Image.new("RGB", (64, 48), (index, 0, 0)).save(
                images / f"{index}.jpg"
            )
            writer.writerow(
                {
                    "sample_index": index,
                    "image": f"Images/{index}.jpg",
                    "angle": index,
                    "speed": 0 if index == 4 else 15,
                    "camera_stamp_sec": 100,
                    "camera_stamp_nanosec": index * 50_000_000,
                }
            )
    (session / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "complete": True,
                "dataset_kind": "competition_mission_sequence",
                "mission": {
                    "capture_kind": capture_kind,
                    "records_stationary_frames": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return session


def test_signal_annotation_requires_review_and_materializes_bbox(tmp_path: Path):
    session = _write_mission_session(
        tmp_path,
        "20260815_010101_001_session",
        capture_kind="signal",
    )
    annotation_path = write_annotation_template(session)
    payload = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
    payload["annotator"] = "codex-draft"
    payload["signal"] = {
        "event_present": True,
        "approach_start": 1,
        "decision_deadline": 5,
        "exit_end": 8,
        "state_segments": [
            {
                "start": 2,
                "end": 4,
                "readable": True,
                "red": True,
                "yellow": False,
                "left_arrow": False,
                "straight_green": False,
            }
        ],
        "bbox_keyframes": [
            {"sample_index": 2, "bbox": [0.4, 0.1, 0.6, 0.2]},
            {"sample_index": 4, "bbox": [0.3, 0.1, 0.7, 0.3]},
        ],
    }
    annotation_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    approve_annotation(session, reviewer="human-reviewer")
    loaded = load_mission_session(session, require_approved=True)
    labels = materialize_signal_labels(loaded)

    assert labels[2]["red"] == 1.0
    assert labels[2]["bbox_valid"] == 1.0
    assert labels[4]["progress"] == 1.0
    assert loaded.samples[3].speed == 0.0


def test_shortcut_annotation_has_persistent_handoff_window(tmp_path: Path):
    session = _write_mission_session(
        tmp_path,
        "20260815_010102_001_session",
        capture_kind="shortcut",
        count=12,
    )
    annotation_path = write_annotation_template(session)
    payload = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
    payload["annotator"] = "codex-draft"
    payload["signal"] = {
        "event_present": True,
        "approach_start": 1,
        "decision_deadline": 3,
        "exit_end": 4,
        "state_segments": [
            {
                "start": 2,
                "end": 3,
                "readable": True,
                "red": False,
                "yellow": False,
                "left_arrow": True,
                "straight_green": False,
            }
        ],
        "bbox_keyframes": [
            {"sample_index": 2, "bbox": [0.4, 0.1, 0.6, 0.2]},
            {"sample_index": 3, "bbox": [0.35, 0.1, 0.65, 0.25]},
        ],
    }
    payload["shortcut"] = {
        "active_start": 1,
        "active_end": 12,
        "phase_starts": [
            {"phase": "APPROACH", "start": 1},
            {"phase": "ENTRY_STRAIGHT", "start": 2},
            {"phase": "FIRST_LEFT", "start": 3},
            {"phase": "CONNECTOR", "start": 4},
            {"phase": "SECOND_LEFT", "start": 5},
            {"phase": "REACQUIRE", "start": 6},
        ],
        "handoff_ready_start": 7,
        "handoff_ready_end": 12,
    }
    annotation_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    approve_annotation(session, reviewer="human-reviewer")
    labels = materialize_shortcut_labels(
        load_mission_session(session, require_approved=True)
    )

    assert [label["handoff_ready"] for label in labels] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_split_is_session_disjoint_and_filterable(tmp_path: Path):
    for index in range(3):
        session = _write_mission_session(
            tmp_path,
            f"20260815_01010{index}_001_session",
            capture_kind="signal",
        )
        annotation_path = write_annotation_template(session)
        payload = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
        payload["annotator"] = "codex-draft"
        payload["signal"]["event_present"] = False
        annotation_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        approve_annotation(session, reviewer="human-reviewer")

    split_path = make_split_manifest(
        tmp_path,
        tmp_path / "split.yaml",
        seed=7,
        capture_kind="signal",
    )
    split = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    values = split["train"] + split["validation"] + split["test"]
    assert len(values) == len(set(values)) == 3
