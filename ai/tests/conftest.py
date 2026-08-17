from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import yaml
from PIL import Image

CSV_HEADER = [
    "sample_index",
    "image",
    "angle",
    "speed",
    "input_key",
    "camera_sequence",
    "camera_stamp_sec",
    "camera_stamp_nanosec",
    "camera_received_wall_time_ns",
    "lidar_valid",
    "lidar",
    "lidar_sequence",
    "lidar_stamp_sec",
    "lidar_stamp_nanosec",
    "lidar_received_wall_time_ns",
    "lidar_skew_sec",
    "history_angle_t_minus_4",
    "history_speed_t_minus_4",
    "history_angle_t_minus_3",
    "history_speed_t_minus_3",
    "history_angle_t_minus_2",
    "history_speed_t_minus_2",
    "history_angle_t_minus_1",
    "history_speed_t_minus_1",
]


def write_session(
    root: Path,
    name: str,
    *,
    labels: Iterable[tuple[float, float]],
    max_forward_speed: float = 25.0,
    control_mode: str = "gamepad",
    complete: bool = True,
    generation: int | None = None,
    initial_history_class_ids: list[list[int]] | None = None,
    recorded_histories: Iterable[tuple[tuple[float, float], ...]] | None = None,
) -> Path:
    session = root / name
    images = session / "Images"
    images.mkdir(parents=True)
    label_list = list(labels)
    history_list = list(recorded_histories or ())
    if history_list and len(history_list) != len(label_list):
        raise ValueError("recorded_histories must match labels")
    with (session / "samples.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        for index, (angle, speed) in enumerate(label_list, start=1):
            Image.new("RGB", (32, 24), color=(index, 20, 30)).save(
                images / f"{index}.png"
            )
            history = history_list[index - 1] if history_list else ()
            writer.writerow(
                [
                    index,
                    f"Images/{index}.png",
                    angle,
                    speed,
                    "gamepad",
                    index,
                    1,
                    2,
                    3,
                    "false",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    *(value for pair in history for value in pair),
                    *("" for _ in range(8 - 2 * len(history))),
                ]
            )
    metadata = {
        "format_version": 1,
        "complete": complete,
        "dataset_kind": "camera_first_teleop_behavior_cloning",
        "control_mode": control_mode,
        "sample_count": len(label_list),
        "gamepad": {"max_forward_speed": max_forward_speed},
    }
    if generation is not None:
        metadata["curriculum"] = {
            "generation": generation,
            "initial_history_class_ids": initial_history_class_ids or [[100, 125]] * 4,
        }
    if history_list:
        metadata["sample_clock"] = "camera_frame"
        metadata["history"] = {
            "frames": 4,
            "time_order": "oldest_to_newest",
            "initial_command": [0, 0],
            "update": "externally_executed_commands",
        }
    (session / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8"
    )
    return session


def write_split_manifest(
    path: Path,
    *,
    train: list[str],
    val: list[str],
    test: list[str],
    schema_version: int = 1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": schema_version,
                "dataset_snapshot": "synthetic",
                "splits": {"train": train, "val": val, "test": test},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
