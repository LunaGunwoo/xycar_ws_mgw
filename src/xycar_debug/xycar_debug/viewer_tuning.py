# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class ViewerConfig:
    update_interval_ms: int = 100
    display_abs_x_m: float = 6.0
    display_y_min_m: float = -1.0
    display_y_max_m: float = 12.0
    raw_point_size: float = 8.0
    candidate_point_size: float = 18.0
    cone_point_size: float = 72.0
    trajectory_horizon_m: float = 3.0
    trajectory_step_m: float = 0.10
    window_geometry: str = "1200x720+80+80"


@dataclass(frozen=True)
class DriveToggleConfig:
    path_loss_timeout_sec: float = 0.50
    stop_publish_count: int = 5
    key_debounce_sec: float = 0.30


@dataclass(frozen=True)
class ConeViewerConfig:
    view: ViewerConfig = ViewerConfig()
    drive: DriveToggleConfig = DriveToggleConfig()


def default_viewer_tuning_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("xycar_debug"))
        return str(share / "config" / "cone_viewer.yaml")
    except Exception:
        return str(
            Path(__file__).resolve().parents[1]
            / "config"
            / "cone_viewer.yaml"
        )


def load_viewer_config(path: str) -> ConeViewerConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"viewer tuning file does not exist: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    expected_sections = {"view", "drive"}
    if not isinstance(payload, Mapping) or set(payload) != expected_sections:
        raise ValueError(
            "viewer tuning must contain exactly the 'view' and 'drive' sections"
        )
    view_values = payload["view"]
    drive_values = payload["drive"]
    if not isinstance(view_values, Mapping):
        raise ValueError("view must be a mapping")
    if not isinstance(drive_values, Mapping):
        raise ValueError("drive must be a mapping")
    allowed = {item.name for item in fields(ViewerConfig)}
    unknown = set(view_values) - allowed
    if unknown:
        raise ValueError(f"unknown view keys: {sorted(unknown)}")
    drive_allowed = {item.name for item in fields(DriveToggleConfig)}
    drive_unknown = set(drive_values) - drive_allowed
    if drive_unknown:
        raise ValueError(f"unknown drive keys: {sorted(drive_unknown)}")
    try:
        view_config = ViewerConfig(**dict(view_values))
        drive_config = DriveToggleConfig(**dict(drive_values))
    except TypeError as exc:
        raise ValueError(f"invalid viewer config: {exc}") from exc
    _validate_view(view_config)
    _validate_drive(drive_config)
    return ConeViewerConfig(view=view_config, drive=drive_config)


def _validate_view(config: ViewerConfig) -> None:
    if config.update_interval_ms < 20:
        raise ValueError("view.update_interval_ms must be at least 20")
    if config.display_abs_x_m <= 0.0:
        raise ValueError("view.display_abs_x_m must be positive")
    if config.display_y_max_m <= config.display_y_min_m:
        raise ValueError("view y limits are invalid")
    if min(
        config.raw_point_size,
        config.candidate_point_size,
        config.cone_point_size,
    ) <= 0.0:
        raise ValueError("view point sizes must be positive")
    if config.trajectory_horizon_m <= 0.0:
        raise ValueError("view.trajectory_horizon_m must be positive")
    if not 0.01 <= config.trajectory_step_m <= 0.50:
        raise ValueError("view.trajectory_step_m must be in [0.01, 0.50]")


def _validate_drive(config: DriveToggleConfig) -> None:
    if not isinstance(config.stop_publish_count, int) or isinstance(
        config.stop_publish_count,
        bool,
    ):
        raise ValueError("drive.stop_publish_count must be an integer")
    if config.stop_publish_count < 1:
        raise ValueError("drive.stop_publish_count must be positive")
    for name, value in (
        ("path_loss_timeout_sec", config.path_loss_timeout_sec),
        ("key_debounce_sec", config.key_debounce_sec),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"drive.{name} must be a number")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"drive.{name} must be finite and positive")
    if config.key_debounce_sec > 2.0:
        raise ValueError("drive.key_debounce_sec must not exceed 2.0")
