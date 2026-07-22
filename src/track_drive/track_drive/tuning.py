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
from typing import Any, Dict, Mapping, Type, TypeVar

import yaml

from track_drive.cone_following import ConeFilterConfig, ConePathConfig
from track_drive.control import ControlConfig, HARDWARE_STEERING_LIMIT


@dataclass(frozen=True)
class TopicConfig:
    scan_topic: str = "/scan"
    motor_topic: str = "xycar_motor"


@dataclass(frozen=True)
class ConeDriveTuning:
    topics: TopicConfig = TopicConfig()
    cone_filter: ConeFilterConfig = ConeFilterConfig()
    cone_path: ConePathConfig = ConePathConfig()
    control: ControlConfig = ControlConfig()


ConfigType = TypeVar("ConfigType")


def default_tuning_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("track_drive"))
        return str(share / "config" / "cone_drive.yaml")
    except Exception:
        return str(
            Path(__file__).resolve().parents[1] / "config" / "cone_drive.yaml"
        )


def load_tuning(path: str) -> ConeDriveTuning:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"tuning file does not exist: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"tuning root must be a mapping: {source}")

    expected = {"topics", "cone_filter", "cone_path", "control"}
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise ValueError(f"unknown tuning sections: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing tuning sections: {sorted(missing)}")

    tuning = ConeDriveTuning(
        topics=_config_from_mapping(TopicConfig, payload["topics"], "topics"),
        cone_filter=_config_from_mapping(
            ConeFilterConfig,
            payload["cone_filter"],
            "cone_filter",
        ),
        cone_path=_config_from_mapping(
            ConePathConfig,
            payload["cone_path"],
            "cone_path",
        ),
        control=_config_from_mapping(
            ControlConfig,
            payload["control"],
            "control",
        ),
    )
    validate_tuning(tuning)
    return tuning


def validate_tuning(tuning: ConeDriveTuning) -> None:
    topics = tuning.topics
    cone_filter = tuning.cone_filter
    path = tuning.cone_path
    control = tuning.control

    if not topics.scan_topic.strip() or not topics.motor_topic.strip():
        raise ValueError("scan_topic and motor_topic must not be empty")
    _positive("cone_filter.min_range_m", cone_filter.min_range_m)
    if cone_filter.max_range_m <= cone_filter.min_range_m:
        raise ValueError("cone_filter.max_range_m must exceed min_range_m")
    _positive("cone_filter.max_abs_x_m", cone_filter.max_abs_x_m)
    _positive(
        "cone_filter.max_adjacent_gap_m",
        cone_filter.max_adjacent_gap_m,
    )
    _integer("cone_filter.max_index_gap", cone_filter.max_index_gap)
    _integer("cone_filter.min_cluster_points", cone_filter.min_cluster_points)
    _integer("cone_filter.max_cluster_points", cone_filter.max_cluster_points)
    if cone_filter.max_index_gap < 1:
        raise ValueError("cone_filter.max_index_gap must be at least 1")
    if cone_filter.min_cluster_points < 1:
        raise ValueError("cone_filter.min_cluster_points must be at least 1")
    if cone_filter.max_cluster_points < cone_filter.min_cluster_points:
        raise ValueError(
            "cone_filter.max_cluster_points must be >= min_cluster_points"
        )
    _positive(
        "cone_filter.max_cluster_diameter_m",
        cone_filter.max_cluster_diameter_m,
    )

    if cone_filter.min_forward_m < 0.0:
        raise ValueError("cone_filter.min_forward_m must be non-negative")
    if not math.isfinite(cone_filter.front_angle_deg):
        raise ValueError("cone_filter.front_angle_deg must be finite")

    for name, value in (
        ("max_cones_per_side", path.max_cones_per_side),
        ("max_graph_cones", path.max_graph_cones),
        ("max_boundary_path_length", path.max_boundary_path_length),
        ("max_boundary_paths", path.max_boundary_paths),
        ("max_previous_reuse_frames", path.max_previous_reuse_frames),
    ):
        _integer(f"cone_path.{name}", value)
    if path.max_cones_per_side < 2 or path.max_graph_cones < 4:
        raise ValueError("cone path cone limits are too small")
    if path.max_boundary_path_length < 2 or path.max_boundary_paths < 1:
        raise ValueError("cone path graph limits are invalid")
    _ordered_positive(
        "cone_path edge length",
        path.min_edge_length_m,
        path.max_edge_length_m,
    )
    _ordered_positive(
        "cone_path lane width",
        path.min_lane_width_m,
        path.max_lane_width_m,
    )
    _positive("cone_path.max_width_delta_m", path.max_width_delta_m)
    _positive(
        "cone_path.min_forward_progress_m",
        path.min_forward_progress_m,
    )
    if not 0.0 < path.max_boundary_turn_deg <= 180.0:
        raise ValueError("cone_path.max_boundary_turn_deg must be in (0, 180]")
    _positive("cone_path.centerline_step_m", path.centerline_step_m)
    _positive("cone_path.min_plan_horizon_m", path.min_plan_horizon_m)
    _ordered_positive(
        "cone_path half width",
        path.min_half_width_m,
        path.max_half_width_m,
    )
    if not path.min_half_width_m <= path.default_half_width_m <= path.max_half_width_m:
        raise ValueError("cone_path.default_half_width_m is outside its limits")
    if not 0.0 <= path.half_width_ema_alpha <= 1.0:
        raise ValueError("cone_path.half_width_ema_alpha must be in [0, 1]")
    _ordered_positive(
        "cone_path lookahead",
        path.min_lookahead_m,
        path.max_lookahead_m,
    )
    if not path.min_lookahead_m <= path.mid_lookahead_m <= path.max_lookahead_m:
        raise ValueError("cone_path.mid_lookahead_m is outside its limits")
    samples = path.fallback_sample_y_m
    if len(samples) < 2 or any(value <= 0.0 for value in samples):
        raise ValueError("cone_path.fallback_sample_y_m needs positive samples")
    if tuple(sorted(samples)) != tuple(samples):
        raise ValueError("cone_path.fallback_sample_y_m must be increasing")
    if path.max_previous_reuse_frames < 0:
        raise ValueError("cone_path.max_previous_reuse_frames must be >= 0")
    if not 0.0 <= path.reused_confidence_decay <= 1.0:
        raise ValueError("cone_path.reused_confidence_decay must be in [0, 1]")
    _ordered_positive(
        "cone_path lookahead curvature",
        path.lookahead_medium_curvature,
        path.lookahead_sharp_curvature,
    )

    _positive("control.control_rate_hz", control.control_rate_hz)
    _positive("control.scan_timeout_sec", control.scan_timeout_sec)
    for name, value in (
        ("min_pair_confidence", control.min_pair_confidence),
        ("min_fallback_both_confidence", control.min_fallback_both_confidence),
        ("min_single_side_confidence", control.min_single_side_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"control.{name} must be in [0, 1]")
    _integer(
        "control.min_cones_per_visible_side",
        control.min_cones_per_visible_side,
    )
    if control.min_cones_per_visible_side < 2:
        raise ValueError("control.min_cones_per_visible_side must be >= 2")
    _positive("control.min_drive_horizon_m", control.min_drive_horizon_m)
    _positive("control.max_speed", control.max_speed)
    if control.max_speed > 5.0:
        raise ValueError("control.max_speed must not exceed 5 for cone debug")
    if not 0.0 < control.sharp_turn_speed <= control.max_speed:
        raise ValueError("control.sharp_turn_speed must be in (0, max_speed]")
    if not 0.0 < control.single_side_speed <= control.max_speed:
        raise ValueError("control.single_side_speed must be in (0, max_speed]")
    if control.sharp_turn_speed < 3.0 or control.single_side_speed < 3.0:
        raise ValueError("control cone driving speeds must be at least 3")
    _ordered_positive(
        "control curvature",
        control.medium_curvature,
        control.sharp_curvature,
    )
    if control.curvature_gain < 0.0 or control.heading_gain < 0.0:
        raise ValueError("control steering gains must be non-negative")
    if control.steering_sign not in {-1.0, 1.0}:
        raise ValueError("control.steering_sign must be -1 or 1")
    if not 0.0 < control.steering_limit <= HARDWARE_STEERING_LIMIT:
        raise ValueError(
            "control.steering_limit must be in (0, 100]"
        )


def _config_from_mapping(
    config_type: Type[ConfigType],
    value: Any,
    section_name: str,
) -> ConfigType:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section_name} must be a mapping")
    allowed = {item.name for item in fields(config_type)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"unknown keys in {section_name}: {sorted(unknown)}"
        )
    kwargs: Dict[str, Any] = dict(value)
    if "fallback_sample_y_m" in kwargs:
        samples = kwargs["fallback_sample_y_m"]
        if not isinstance(samples, (list, tuple)):
            raise ValueError(
                "cone_path.fallback_sample_y_m must be a sequence"
            )
        kwargs["fallback_sample_y_m"] = tuple(float(item) for item in samples)
    try:
        return config_type(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {section_name}: {exc}") from exc


def _positive(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _ordered_positive(name: str, minimum: float, maximum: float) -> None:
    _positive(f"{name} minimum", minimum)
    _positive(f"{name} maximum", maximum)
    if maximum <= minimum:
        raise ValueError(f"{name} minimum/maximum are invalid")


def _integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
