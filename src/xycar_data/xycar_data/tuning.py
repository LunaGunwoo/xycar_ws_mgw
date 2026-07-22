# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Validated YAML tuning for the camera-first teleop recorder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar

import yaml


@dataclass(frozen=True)
class TopicConfig:
    camera_topic: str = "/image_raw"
    lidar_topic: str = "/scan"
    motor_topic: str = "/xycar_motor"


@dataclass(frozen=True)
class ControlConfig:
    publish_rate_hz: float = 20.0
    key_timeout_sec: float = 0.25
    graph_check_period_sec: float = 0.5
    stop_publish_count: int = 5
    forward_angle: float = 0.0
    forward_speed: float = 5.0
    reverse_angle: float = 0.0
    reverse_speed: float = -3.0
    left_angle: float = -30.0
    left_speed: float = 3.0
    right_angle: float = 30.0
    right_speed: float = 3.0


@dataclass(frozen=True)
class SensorConfig:
    camera_timeout_sec: float = 0.25
    lidar_timeout_sec: float = 0.30
    max_lidar_skew_sec: float = 0.20


@dataclass(frozen=True)
class RecordingConfig:
    root_dir: str = "/home/xytron/xycar_data/teleop"
    png_compression: int = 3
    queue_size: int = 128
    min_free_space_mb: int = 1024


@dataclass(frozen=True)
class TeleopTuning:
    topics: TopicConfig = TopicConfig()
    control: ControlConfig = ControlConfig()
    sensors: SensorConfig = SensorConfig()
    recording: RecordingConfig = RecordingConfig()


ConfigType = TypeVar("ConfigType")


def default_tuning_path() -> str:
    """Return the installed config, with a source-tree fallback for tooling."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("xycar_data"))
        return str(share / "config" / "teleop_recorder.yaml")
    except Exception:
        return str(
            Path(__file__).resolve().parents[1]
            / "config"
            / "teleop_recorder.yaml"
        )


def load_tuning(path: str) -> TeleopTuning:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"tuning file does not exist: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"tuning root must be a mapping: {source}")

    expected = {"topics", "control", "sensors", "recording"}
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise ValueError(f"unknown tuning sections: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing tuning sections: {sorted(missing)}")

    tuning = TeleopTuning(
        topics=_config_from_mapping(TopicConfig, payload["topics"], "topics"),
        control=_config_from_mapping(
            ControlConfig,
            payload["control"],
            "control",
        ),
        sensors=_config_from_mapping(
            SensorConfig,
            payload["sensors"],
            "sensors",
        ),
        recording=_config_from_mapping(
            RecordingConfig,
            payload["recording"],
            "recording",
        ),
    )
    validate_tuning(tuning)
    return tuning


def tuning_as_mapping(tuning: TeleopTuning) -> dict[str, Any]:
    """Return a serializable snapshot to store in each session metadata."""
    return asdict(tuning)


def validate_tuning(tuning: TeleopTuning) -> None:
    for name, value in asdict(tuning.topics).items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"topics.{name} must be a non-empty string")

    control = tuning.control
    _range("control.publish_rate_hz", control.publish_rate_hz, 1.0, 100.0)
    _range("control.key_timeout_sec", control.key_timeout_sec, 0.05, 2.0)
    _range(
        "control.graph_check_period_sec",
        control.graph_check_period_sec,
        0.05,
        10.0,
    )
    _integer_range(
        "control.stop_publish_count",
        control.stop_publish_count,
        1,
        20,
    )
    for field in (
        "forward_angle",
        "reverse_angle",
        "left_angle",
        "right_angle",
    ):
        _range(f"control.{field}", getattr(control, field), -100.0, 100.0)
    for field in (
        "forward_speed",
        "reverse_speed",
        "left_speed",
        "right_speed",
    ):
        _range(f"control.{field}", getattr(control, field), -100.0, 100.0)
    if control.forward_speed <= 0.0:
        raise ValueError("control.forward_speed must be positive")
    if control.reverse_speed >= 0.0:
        raise ValueError("control.reverse_speed must be negative")
    if control.left_speed <= 0.0 or control.right_speed <= 0.0:
        raise ValueError("control.left_speed and right_speed must be positive")
    if control.left_angle >= 0.0 or control.right_angle <= 0.0:
        raise ValueError("control.left_angle must be negative and right_angle positive")

    sensors = tuning.sensors
    _range("sensors.camera_timeout_sec", sensors.camera_timeout_sec, 0.05, 5.0)
    _range("sensors.lidar_timeout_sec", sensors.lidar_timeout_sec, 0.05, 10.0)
    _range("sensors.max_lidar_skew_sec", sensors.max_lidar_skew_sec, 0.0, 5.0)

    recording = tuning.recording
    if not isinstance(recording.root_dir, str) or not recording.root_dir.strip():
        raise ValueError("recording.root_dir must be a non-empty string")
    _integer_range("recording.png_compression", recording.png_compression, 0, 9)
    _integer_range("recording.queue_size", recording.queue_size, 1, 4096)
    _integer_range(
        "recording.min_free_space_mb",
        recording.min_free_space_mb,
        0,
        1024 * 1024,
    )


def _config_from_mapping(
    config_type: Type[ConfigType],
    values: object,
    section: str,
) -> ConfigType:
    if not isinstance(values, Mapping):
        raise ValueError(f"{section} must be a mapping")
    allowed = {item.name for item in fields(config_type)}
    unknown = set(values) - allowed
    missing = allowed - set(values)
    if unknown:
        raise ValueError(f"unknown {section} keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing {section} keys: {sorted(missing)}")
    try:
        return config_type(**dict(values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {section}: {exc}") from exc


def _range(name: str, value: object, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")


def _integer_range(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
