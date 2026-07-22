# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

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


def load_viewer_config(path: str) -> ViewerConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"viewer tuning file does not exist: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"view"}:
        raise ValueError("viewer tuning must contain exactly the 'view' section")
    values = payload["view"]
    if not isinstance(values, Mapping):
        raise ValueError("view must be a mapping")
    allowed = {item.name for item in fields(ViewerConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown view keys: {sorted(unknown)}")
    try:
        config = ViewerConfig(**dict(values))
    except TypeError as exc:
        raise ValueError(f"invalid view config: {exc}") from exc
    _validate(config)
    return config


def _validate(config: ViewerConfig) -> None:
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
