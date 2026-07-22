# Copyright 2026 Gunwoo Moon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class DebugDriveConfig:
    start_valid_frames: int = 3
    path_loss_timeout_sec: float = 0.50
    stop_publish_count: int = 5


def default_debug_drive_tuning_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("xycar_debug"))
        return str(share / "config" / "cone_debug_drive.yaml")
    except Exception:
        return str(
            Path(__file__).resolve().parents[1]
            / "config"
            / "cone_debug_drive.yaml"
        )


def load_debug_drive_config(path: str) -> DebugDriveConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"debug drive tuning file does not exist: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"mission"}:
        raise ValueError(
            "debug drive tuning must contain exactly the 'mission' section"
        )
    values = payload["mission"]
    if not isinstance(values, Mapping):
        raise ValueError("mission must be a mapping")
    allowed = {item.name for item in fields(DebugDriveConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown mission keys: {sorted(unknown)}")
    try:
        config = DebugDriveConfig(**dict(values))
    except TypeError as exc:
        raise ValueError(f"invalid mission config: {exc}") from exc
    validate_debug_drive_config(config)
    return config


def validate_debug_drive_config(config: DebugDriveConfig) -> None:
    _positive_integer("start_valid_frames", config.start_valid_frames)
    _positive_integer("stop_publish_count", config.stop_publish_count)
    if not isinstance(config.path_loss_timeout_sec, (int, float)):
        raise ValueError("mission.path_loss_timeout_sec must be a number")
    if not math.isfinite(config.path_loss_timeout_sec):
        raise ValueError("mission.path_loss_timeout_sec must be finite")
    if config.path_loss_timeout_sec <= 0.0:
        raise ValueError("mission.path_loss_timeout_sec must be positive")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"mission.{name} must be a positive integer")
