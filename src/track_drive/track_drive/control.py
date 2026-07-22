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

from dataclasses import dataclass

from track_drive.cone_following import ConeLanePlan


HARDWARE_STEERING_LIMIT = 100.0


@dataclass(frozen=True)
class ControlConfig:
    control_rate_hz: float = 10.0
    scan_timeout_sec: float = 0.30
    min_pair_confidence: float = 0.50
    min_fallback_both_confidence: float = 0.35
    min_single_side_confidence: float = 0.22
    min_cones_per_visible_side: int = 2
    min_drive_horizon_m: float = 1.60
    max_speed: float = 5.0
    sharp_turn_speed: float = 3.0
    single_side_speed: float = 3.0
    medium_curvature: float = 0.025
    sharp_curvature: float = 0.045
    curvature_gain: float = 160.0
    heading_gain: float = 220.0
    steering_sign: float = 1.0
    steering_limit: float = 30.0


@dataclass(frozen=True)
class DriveDecision:
    angle: float = 0.0
    speed: float = 0.0
    can_drive: bool = False
    reason: str = "stopped"
    plan_mode: str = "none"


def decide_drive(
    plan: ConeLanePlan,
    config: ControlConfig,
    drive_enabled: bool,
    scan_fresh: bool,
) -> DriveDecision:
    """Turn a current cone plan into the exact command used by both nodes."""
    if not drive_enabled:
        return _stop("drive disabled", plan.mode)
    if not scan_fresh:
        return _stop("scan stale", plan.mode)
    if plan.is_reused:
        return _stop("previous path is display-only", plan.mode)
    if not plan.has_target or plan.curvature is None:
        return _stop("no cone path", plan.mode)
    if plan.horizon_m < config.min_drive_horizon_m:
        return _stop("path horizon too short", plan.mode)

    left_count = len(plan.left_boundary)
    right_count = len(plan.right_boundary)
    minimum = config.min_cones_per_visible_side

    if plan.mode == "paired":
        if left_count < minimum or right_count < minimum:
            return _stop("paired boundary has too few cones", plan.mode)
        if plan.confidence < config.min_pair_confidence:
            return _stop("paired path confidence too low", plan.mode)
        speed = _paired_speed(plan.curvature, config)
    elif plan.mode == "fallback_both":
        if left_count < minimum or right_count < minimum:
            return _stop("fallback boundary has too few cones", plan.mode)
        if plan.confidence < config.min_fallback_both_confidence:
            return _stop("fallback confidence too low", plan.mode)
        speed = config.sharp_turn_speed
    elif plan.mode in {"single_left", "single_right"}:
        visible_count = left_count if plan.mode == "single_left" else right_count
        if visible_count < minimum:
            return _stop("single boundary has too few cones", plan.mode)
        if plan.confidence < config.min_single_side_confidence:
            return _stop("single boundary confidence too low", plan.mode)
        speed = config.single_side_speed
    else:
        return _stop("unsupported path mode", plan.mode)

    raw_angle = config.steering_sign * (
        plan.curvature * config.curvature_gain
        + plan.path_heading_rad * config.heading_gain
    )
    limit = min(abs(config.steering_limit), HARDWARE_STEERING_LIMIT)
    angle = _clamp(raw_angle, -limit, limit)
    return DriveDecision(
        angle=float(angle),
        speed=float(min(config.max_speed, speed)),
        can_drive=True,
        reason="current cone path is driveable",
        plan_mode=plan.mode,
    )


def _paired_speed(curvature: float, config: ControlConfig) -> float:
    magnitude = abs(curvature)
    if magnitude <= config.medium_curvature:
        return config.max_speed
    if magnitude >= config.sharp_curvature:
        return config.sharp_turn_speed
    span = config.sharp_curvature - config.medium_curvature
    ratio = (magnitude - config.medium_curvature) / span
    return config.max_speed + ratio * (
        config.sharp_turn_speed - config.max_speed
    )


def _stop(reason: str, mode: str) -> DriveDecision:
    return DriveDecision(reason=reason, plan_mode=mode)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
