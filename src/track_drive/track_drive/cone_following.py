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
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class LaserScanSnapshot:
    """ROS-independent copy of the fields used from sensor_msgs/LaserScan."""

    ranges: Sequence[float]
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float


@dataclass(frozen=True)
class ConeFilterConfig:
    min_range_m: float = 0.30
    max_range_m: float = 10.00
    min_forward_m: float = 0.20
    max_abs_x_m: float = 4.00
    max_adjacent_gap_m: float = 0.35
    max_index_gap: int = 2
    min_cluster_points: int = 2
    max_cluster_points: int = 8
    max_cluster_diameter_m: float = 0.35
    front_angle_deg: float = 0.0


@dataclass(frozen=True)
class ConePathConfig:
    max_cones_per_side: int = 5
    max_graph_cones: int = 12
    max_boundary_path_length: int = 5
    max_boundary_paths: int = 120
    min_edge_length_m: float = 0.70
    max_edge_length_m: float = 5.00
    min_forward_progress_m: float = 0.25
    min_lane_width_m: float = 2.00
    max_lane_width_m: float = 6.20
    max_width_delta_m: float = 1.70
    max_boundary_turn_deg: float = 105.0
    centerline_step_m: float = 0.45
    min_plan_horizon_m: float = 1.60
    min_lookahead_m: float = 1.20
    mid_lookahead_m: float = 1.80
    max_lookahead_m: float = 2.60
    default_half_width_m: float = 2.25
    min_half_width_m: float = 1.40
    max_half_width_m: float = 3.00
    half_width_ema_alpha: float = 0.20
    fallback_sample_y_m: Tuple[float, ...] = (
        1.20,
        1.80,
        2.60,
        3.40,
        4.20,
    )
    max_previous_reuse_frames: int = 3
    reused_confidence_decay: float = 0.25
    lookahead_medium_curvature: float = 0.045
    lookahead_sharp_curvature: float = 0.12


@dataclass(frozen=True)
class LidarPoint:
    index: int
    range_m: float
    x_m: float
    y_m: float


@dataclass(frozen=True)
class ConeCluster:
    start_index: int
    end_index: int
    point_count: int
    center_x_m: float
    center_y_m: float
    min_range_m: float
    max_range_m: float
    diameter_m: float

    @property
    def side(self) -> str:
        if self.center_x_m < 0.0:
            return "left"
        if self.center_x_m > 0.0:
            return "right"
        return "center"


@dataclass(frozen=True)
class ConeLanePlan:
    waypoints: Tuple[Point2D, ...] = ()
    lookahead_point: Optional[Point2D] = None
    curvature: Optional[float] = None
    path_heading_rad: float = 0.0
    confidence: float = 0.0
    width_m: Optional[float] = None
    mode: str = "none"
    left_boundary: Tuple[ConeCluster, ...] = ()
    right_boundary: Tuple[ConeCluster, ...] = ()
    candidate_cones: Tuple[ConeCluster, ...] = ()
    reused_frames: int = 0

    @property
    def has_target(self) -> bool:
        return self.lookahead_point is not None and self.curvature is not None

    @property
    def is_reused(self) -> bool:
        return self.reused_frames > 0 or self.mode == "reused"

    @property
    def horizon_m(self) -> float:
        if not self.waypoints:
            return 0.0
        return max(point[1] for point in self.waypoints)


@dataclass(frozen=True)
class _BoundaryPath:
    clusters: Tuple[ConeCluster, ...]
    score: float


class ConePathTracker:
    """Build a cone corridor from one LaserScan without ROS dependencies."""

    def __init__(
        self,
        filter_config: ConeFilterConfig = ConeFilterConfig(),
        path_config: ConePathConfig = ConePathConfig(),
    ) -> None:
        self.filter_config = filter_config
        self.path_config = path_config
        self.estimated_half_width_m = path_config.default_half_width_m
        self.previous_plan: Optional[ConeLanePlan] = None
        self.previous_reuse_count = 0

    def reset(self) -> None:
        self.estimated_half_width_m = self.path_config.default_half_width_m
        self.previous_plan = None
        self.previous_reuse_count = 0

    def update(self, scan: LaserScanSnapshot) -> ConeLanePlan:
        clusters = extract_cone_clusters(scan, self.filter_config)
        clusters = sorted(
            clusters,
            key=lambda item: math.hypot(item.center_x_m, item.center_y_m),
        )[: self.path_config.max_graph_cones]
        clusters = sorted(clusters, key=lambda item: item.center_y_m)

        plan = self._paired_plan(clusters)
        if not plan.has_target:
            plan = self._curve_fallback_plan(clusters)
        if plan.has_target:
            self._remember(plan)
            return plan

        reused = self._reuse_previous(tuple(clusters))
        if reused is not None:
            return reused
        return ConeLanePlan(candidate_cones=tuple(clusters))

    def _paired_plan(
        self,
        clusters: Sequence[ConeCluster],
    ) -> ConeLanePlan:
        left = [item for item in clusters if item.side == "left"]
        right = [item for item in clusters if item.side == "right"]
        left_paths = _boundary_paths(left, self.path_config)
        right_paths = _boundary_paths(right, self.path_config)
        best_plan: Optional[ConeLanePlan] = None
        best_score = -math.inf

        for left_path in left_paths:
            for right_path in right_paths:
                result = _centerline_between_boundaries(
                    left_path.clusters,
                    right_path.clusters,
                    self.path_config,
                )
                waypoints, width_m, width_delta = result
                if not waypoints or width_m is None:
                    continue
                lookahead = _lookahead_point(
                    tuple(waypoints),
                    self.path_config,
                )
                curvature = pure_pursuit_curvature(lookahead)
                if curvature is None:
                    continue

                min_side_count = min(
                    len(left_path.clusters),
                    len(right_path.clusters),
                )
                horizon_m = waypoints[-1][1]
                turn_cost = _centerline_turn_cost(tuple(waypoints))
                score = (
                    left_path.score
                    + right_path.score
                    + 0.45 * horizon_m
                    + 0.85 * min_side_count
                    - 0.70 * width_delta
                    - 0.25 * turn_cost
                    + self._previous_plan_score(lookahead)
                )
                confidence = _clamp(
                    0.42
                    + 0.08 * min_side_count
                    + 0.035 * horizon_m
                    - 0.08 * width_delta
                    - 0.04 * turn_cost,
                    0.0,
                    1.0,
                )
                candidate = ConeLanePlan(
                    waypoints=tuple(waypoints),
                    lookahead_point=lookahead,
                    curvature=curvature,
                    path_heading_rad=_path_heading(tuple(waypoints)),
                    confidence=confidence,
                    width_m=width_m,
                    mode="paired",
                    left_boundary=left_path.clusters,
                    right_boundary=right_path.clusters,
                    candidate_cones=tuple(clusters),
                )
                if score > best_score:
                    best_plan = candidate
                    best_score = score

        if best_plan is None:
            return ConeLanePlan(candidate_cones=tuple(clusters))
        self._update_half_width(best_plan.width_m)
        return best_plan

    def _curve_fallback_plan(
        self,
        clusters: Sequence[ConeCluster],
    ) -> ConeLanePlan:
        left = _side_cones(clusters, "left", self.path_config)
        right = _side_cones(clusters, "right", self.path_config)
        left_curve = _fit_curve(left)
        right_curve = _fit_curve(right)
        if left_curve is None and right_curve is None:
            return ConeLanePlan(candidate_cones=tuple(clusters))

        width_m = _median_curve_width(left_curve, right_curve)
        if width_m is not None:
            if not (
                self.path_config.min_lane_width_m
                <= width_m
                <= self.path_config.max_lane_width_m
            ):
                return ConeLanePlan(candidate_cones=tuple(clusters))
            self._update_half_width(width_m)

        evidence_horizon = max(
            [item.center_y_m for item in left + right],
            default=0.0,
        )
        waypoints: List[Point2D] = []
        for y_m in self.path_config.fallback_sample_y_m:
            if y_m > evidence_horizon + 0.30:
                continue
            target_x = _fallback_target_x(
                left_curve,
                right_curve,
                y_m,
                self.estimated_half_width_m,
            )
            if target_x is not None:
                waypoints.append((float(target_x), float(y_m)))

        if not waypoints:
            return ConeLanePlan(candidate_cones=tuple(clusters))
        lookahead = _lookahead_point(tuple(waypoints), self.path_config)
        curvature = pure_pursuit_curvature(lookahead)
        if lookahead is None or curvature is None:
            return ConeLanePlan(candidate_cones=tuple(clusters))

        if left_curve is not None and right_curve is not None:
            mode = "fallback_both"
            confidence = 0.35
        elif left_curve is not None:
            mode = "single_left"
            confidence = 0.22
        else:
            mode = "single_right"
            confidence = 0.22

        return ConeLanePlan(
            waypoints=tuple(waypoints),
            lookahead_point=lookahead,
            curvature=curvature,
            path_heading_rad=_path_heading(tuple(waypoints)),
            confidence=confidence,
            width_m=width_m,
            mode=mode,
            left_boundary=tuple(left),
            right_boundary=tuple(right),
            candidate_cones=tuple(clusters),
        )

    def _remember(self, plan: ConeLanePlan) -> None:
        self.previous_plan = plan
        self.previous_reuse_count = 0

    def _reuse_previous(
        self,
        candidates: Tuple[ConeCluster, ...],
    ) -> Optional[ConeLanePlan]:
        if self.previous_plan is None:
            return None
        if (
            self.previous_reuse_count
            >= self.path_config.max_previous_reuse_frames
        ):
            return None
        self.previous_reuse_count += 1
        confidence = max(
            0.0,
            self.previous_plan.confidence
            * (
                1.0
                - self.path_config.reused_confidence_decay
                * self.previous_reuse_count
            ),
        )
        return replace(
            self.previous_plan,
            confidence=confidence,
            mode="reused",
            candidate_cones=candidates,
            reused_frames=self.previous_reuse_count,
        )

    def _update_half_width(self, width_m: Optional[float]) -> None:
        if width_m is None:
            return
        half_width = _clamp(
            width_m / 2.0,
            self.path_config.min_half_width_m,
            self.path_config.max_half_width_m,
        )
        alpha = _clamp(self.path_config.half_width_ema_alpha, 0.0, 1.0)
        self.estimated_half_width_m = (
            alpha * half_width
            + (1.0 - alpha) * self.estimated_half_width_m
        )

    def _previous_plan_score(self, lookahead: Point2D) -> float:
        if self.previous_plan is None:
            return 0.0
        previous = self.previous_plan.lookahead_point
        if previous is None:
            return 0.0
        distance = math.hypot(
            previous[0] - lookahead[0],
            previous[1] - lookahead[1],
        )
        return 0.55 * max(0.0, 1.0 - distance / 1.25)


def scan_points(
    scan: LaserScanSnapshot,
    config: ConeFilterConfig = ConeFilterConfig(),
) -> List[LidarPoint]:
    range_max = scan.range_max if scan.range_max > 0.0 else math.inf
    front_angle_rad = math.radians(config.front_angle_deg)
    points: List[LidarPoint] = []
    for index, raw_range in enumerate(scan.ranges):
        distance = _finite_float(raw_range)
        if distance is None:
            continue
        if distance < scan.range_min or distance > range_max:
            continue
        angle = scan.angle_min + scan.angle_increment * index
        relative_angle = angle - front_angle_rad
        points.append(
            LidarPoint(
                index=index,
                range_m=distance,
                x_m=-distance * math.sin(relative_angle),
                y_m=distance * math.cos(relative_angle),
            )
        )
    return points


def cone_candidate_points(
    scan: LaserScanSnapshot,
    config: ConeFilterConfig = ConeFilterConfig(),
) -> List[LidarPoint]:
    return [
        point
        for point in scan_points(scan, config)
        if config.min_range_m <= point.range_m <= config.max_range_m
        and point.y_m >= config.min_forward_m
        and abs(point.x_m) <= config.max_abs_x_m
    ]


def extract_cone_clusters(
    scan: LaserScanSnapshot,
    config: ConeFilterConfig = ConeFilterConfig(),
) -> List[ConeCluster]:
    points = cone_candidate_points(scan, config)
    if not points:
        return []
    groups = _cluster_scan_order(points, len(scan.ranges), config)
    clusters = [_cluster_from_points(group) for group in groups]
    return [
        cluster
        for cluster in clusters
        if config.min_cluster_points
        <= cluster.point_count
        <= config.max_cluster_points
        and cluster.diameter_m <= config.max_cluster_diameter_m
        and cluster.side in {"left", "right"}
    ]


def pure_pursuit_curvature(
    lookahead_point: Optional[Point2D],
) -> Optional[float]:
    if lookahead_point is None:
        return None
    x_m, y_m = lookahead_point
    distance_squared = x_m * x_m + y_m * y_m
    if distance_squared < 1e-6:
        return None
    return 2.0 * x_m / distance_squared


def _boundary_paths(
    clusters: Sequence[ConeCluster],
    config: ConePathConfig,
) -> List[_BoundaryPath]:
    ordered = sorted(clusters, key=lambda item: item.center_y_m)
    adjacency = {index: [] for index in range(len(ordered))}
    for start_index, start in enumerate(ordered):
        for end_index, end in enumerate(ordered):
            if end_index == start_index:
                continue
            forward = end.center_y_m - start.center_y_m
            distance = _cluster_distance(start, end)
            if forward < config.min_forward_progress_m:
                continue
            if config.min_edge_length_m <= distance <= config.max_edge_length_m:
                adjacency[start_index].append(end_index)

    paths: List[_BoundaryPath] = []

    def visit(indexes: Tuple[int, ...]) -> None:
        selected = tuple(ordered[index] for index in indexes)
        if len(selected) >= 2:
            turn_cost = _boundary_turn_cost(selected)
            if turn_cost <= math.radians(config.max_boundary_turn_deg):
                horizon = selected[-1].center_y_m - selected[0].center_y_m
                score = 0.85 * len(selected) + 0.35 * horizon
                score -= 0.30 * turn_cost
                paths.append(_BoundaryPath(selected, score))
        if len(indexes) >= config.max_boundary_path_length:
            return
        for next_index in adjacency[indexes[-1]]:
            if next_index in indexes:
                continue
            candidate = tuple(ordered[index] for index in indexes + (next_index,))
            if _boundary_turn_cost(candidate) > math.radians(
                config.max_boundary_turn_deg
            ):
                continue
            visit(indexes + (next_index,))

    for index in range(len(ordered)):
        visit((index,))
    paths.sort(key=lambda item: item.score, reverse=True)
    return paths[: config.max_boundary_paths]


def _centerline_between_boundaries(
    left: Sequence[ConeCluster],
    right: Sequence[ConeCluster],
    config: ConePathConfig,
) -> Tuple[List[Point2D], Optional[float], float]:
    y_start = max(left[0].center_y_m, right[0].center_y_m)
    y_end = min(left[-1].center_y_m, right[-1].center_y_m)
    if y_end < config.min_plan_horizon_m:
        return [], None, 0.0
    if y_end - y_start < config.centerline_step_m:
        return [], None, 0.0
    sample_count = max(
        3,
        int(math.ceil((y_end - y_start) / config.centerline_step_m)) + 1,
    )
    waypoints: List[Point2D] = []
    widths: List[float] = []
    for y_value in np.linspace(y_start, y_end, sample_count):
        y_m = float(y_value)
        left_x = _boundary_x_at_y(left, y_m)
        right_x = _boundary_x_at_y(right, y_m)
        width_m = right_x - left_x
        if not config.min_lane_width_m <= width_m <= config.max_lane_width_m:
            return [], None, 0.0
        widths.append(width_m)
        waypoints.append(((left_x + right_x) / 2.0, y_m))
    width_delta = max(widths) - min(widths)
    if width_delta > config.max_width_delta_m:
        return [], None, width_delta
    return waypoints, float(np.median(widths)), width_delta


def _side_cones(
    clusters: Sequence[ConeCluster],
    side: str,
    config: ConePathConfig,
) -> List[ConeCluster]:
    result = [item for item in clusters if item.side == side]
    result.sort(key=lambda item: item.center_y_m)
    return result[: config.max_cones_per_side]


def _fit_curve(
    clusters: Sequence[ConeCluster],
) -> Optional[Tuple[float, float, float]]:
    if not clusters:
        return None
    xs = np.asarray([item.center_x_m for item in clusters], dtype=float)
    ys = np.asarray([item.center_y_m for item in clusters], dtype=float)
    if len(clusters) >= 3:
        values = np.polyfit(ys, xs, 2)
        return tuple(float(value) for value in values)
    if len(clusters) == 2:
        slope, intercept = np.polyfit(ys, xs, 1)
        return 0.0, float(slope), float(intercept)
    return 0.0, 0.0, float(xs[0])


def _curve_x(curve: Tuple[float, float, float], y_m: float) -> float:
    a, b, c = curve
    return a * y_m * y_m + b * y_m + c


def _median_curve_width(
    left: Optional[Tuple[float, float, float]],
    right: Optional[Tuple[float, float, float]],
) -> Optional[float]:
    if left is None or right is None:
        return None
    widths = [
        _curve_x(right, y_m) - _curve_x(left, y_m)
        for y_m in (1.5, 2.8, 4.0)
    ]
    positive = [width for width in widths if width > 0.0]
    if not positive:
        return None
    return float(np.median(positive))


def _fallback_target_x(
    left: Optional[Tuple[float, float, float]],
    right: Optional[Tuple[float, float, float]],
    y_m: float,
    half_width_m: float,
) -> Optional[float]:
    if left is not None and right is not None:
        return (_curve_x(left, y_m) + _curve_x(right, y_m)) / 2.0
    if left is not None:
        return _curve_x(left, y_m) + half_width_m
    if right is not None:
        return _curve_x(right, y_m) - half_width_m
    return None


def _lookahead_point(
    waypoints: Tuple[Point2D, ...],
    config: ConePathConfig,
) -> Optional[Point2D]:
    if not waypoints:
        return None
    path_curvature = abs(_centerline_curvature(waypoints))
    if path_curvature >= config.lookahead_sharp_curvature:
        lookahead_m = config.min_lookahead_m
    elif path_curvature >= config.lookahead_medium_curvature:
        lookahead_m = config.mid_lookahead_m
    else:
        lookahead_m = config.max_lookahead_m
    lookahead_m = _clamp(
        lookahead_m,
        config.min_lookahead_m,
        config.max_lookahead_m,
    )
    return _point_at_arc_length(((0.0, 0.0),) + waypoints, lookahead_m)


def _point_at_arc_length(
    points: Tuple[Point2D, ...],
    target_m: float,
) -> Optional[Point2D]:
    if len(points) < 2:
        return None
    walked = 0.0
    previous = points[0]
    for current in points[1:]:
        length = math.hypot(
            current[0] - previous[0],
            current[1] - previous[1],
        )
        if length < 1e-6:
            previous = current
            continue
        if walked + length >= target_m:
            ratio = (target_m - walked) / length
            return (
                previous[0] + ratio * (current[0] - previous[0]),
                previous[1] + ratio * (current[1] - previous[1]),
            )
        walked += length
        previous = current
    return points[-1]


def _path_heading(waypoints: Tuple[Point2D, ...]) -> float:
    if len(waypoints) < 2:
        return 0.0
    first = waypoints[0]
    usable = [point for point in waypoints[1:] if point[1] <= first[1] + 2.2]
    last = usable[-1] if usable else waypoints[1]
    return math.atan2(last[0] - first[0], max(last[1] - first[1], 1e-6))


def _boundary_x_at_y(
    clusters: Sequence[ConeCluster],
    y_m: float,
) -> float:
    ordered = sorted(clusters, key=lambda item: item.center_y_m)
    if len(ordered) == 1:
        return ordered[0].center_x_m
    if y_m <= ordered[0].center_y_m:
        return _interpolate_x(ordered[0], ordered[1], y_m)
    if y_m >= ordered[-1].center_y_m:
        return _interpolate_x(ordered[-2], ordered[-1], y_m)
    for first, second in zip(ordered, ordered[1:]):
        if first.center_y_m <= y_m <= second.center_y_m:
            return _interpolate_x(first, second, y_m)
    return ordered[-1].center_x_m


def _interpolate_x(
    first: ConeCluster,
    second: ConeCluster,
    y_m: float,
) -> float:
    dy = second.center_y_m - first.center_y_m
    if abs(dy) < 1e-6:
        return (first.center_x_m + second.center_x_m) / 2.0
    ratio = (y_m - first.center_y_m) / dy
    return first.center_x_m + ratio * (second.center_x_m - first.center_x_m)


def _centerline_curvature(points: Tuple[Point2D, ...]) -> float:
    if len(points) < 3:
        return 0.0
    values = [
        _triplet_curvature(first, second, third)
        for first, second, third in zip(points, points[1:], points[2:])
    ]
    return max(values, key=lambda value: abs(value), default=0.0)


def _centerline_turn_cost(points: Tuple[Point2D, ...]) -> float:
    if len(points) < 3:
        return 0.0
    return sum(
        abs(_point_turn_angle(first, second, third))
        for first, second, third in zip(points, points[1:], points[2:])
    )


def _triplet_curvature(
    first: Point2D,
    second: Point2D,
    third: Point2D,
) -> float:
    a = math.dist(first, second)
    b = math.dist(second, third)
    c = math.dist(first, third)
    denominator = a * b * c
    if denominator < 1e-6:
        return 0.0
    area_twice = (second[0] - first[0]) * (third[1] - first[1])
    area_twice -= (second[1] - first[1]) * (third[0] - first[0])
    return 2.0 * area_twice / denominator


def _boundary_turn_cost(clusters: Sequence[ConeCluster]) -> float:
    if len(clusters) < 3:
        return 0.0
    points = [(item.center_x_m, item.center_y_m) for item in clusters]
    return sum(
        abs(_point_turn_angle(first, second, third))
        for first, second, third in zip(points, points[1:], points[2:])
    )


def _point_turn_angle(
    first: Point2D,
    second: Point2D,
    third: Point2D,
) -> float:
    first_vector = (second[0] - first[0], second[1] - first[1])
    second_vector = (third[0] - second[0], third[1] - second[1])
    first_norm = math.hypot(*first_vector)
    second_norm = math.hypot(*second_vector)
    if first_norm < 1e-6 or second_norm < 1e-6:
        return 0.0
    dot = (
        first_vector[0] * second_vector[0]
        + first_vector[1] * second_vector[1]
    )
    cross = (
        first_vector[0] * second_vector[1]
        - first_vector[1] * second_vector[0]
    )
    return math.atan2(cross, dot)


def _cluster_scan_order(
    points: Sequence[LidarPoint],
    range_count: int,
    config: ConeFilterConfig,
) -> List[List[LidarPoint]]:
    groups: List[List[LidarPoint]] = []
    current: List[LidarPoint] = []
    previous: Optional[LidarPoint] = None
    for point in sorted(points, key=lambda item: item.index):
        if previous is None or _should_join(previous, point, config):
            current.append(point)
        else:
            groups.append(current)
            current = [point]
        previous = point
    if current:
        groups.append(current)
    if len(groups) > 1:
        last = groups[-1][-1]
        first = groups[0][0]
        wrapped_index_gap = first.index + range_count - last.index
        if (
            wrapped_index_gap <= config.max_index_gap
            and _point_gap(last, first) <= config.max_adjacent_gap_m
        ):
            groups = [groups[-1] + groups[0]] + groups[1:-1]
    return groups


def _should_join(
    previous: LidarPoint,
    current: LidarPoint,
    config: ConeFilterConfig,
) -> bool:
    return (
        current.index - previous.index <= config.max_index_gap
        and _point_gap(previous, current) <= config.max_adjacent_gap_m
    )


def _cluster_from_points(points: Sequence[LidarPoint]) -> ConeCluster:
    xs = [point.x_m for point in points]
    ys = [point.y_m for point in points]
    ranges = [point.range_m for point in points]
    return ConeCluster(
        start_index=min(point.index for point in points),
        end_index=max(point.index for point in points),
        point_count=len(points),
        center_x_m=sum(xs) / len(xs),
        center_y_m=sum(ys) / len(ys),
        min_range_m=min(ranges),
        max_range_m=max(ranges),
        diameter_m=math.hypot(max(xs) - min(xs), max(ys) - min(ys)),
    )


def _cluster_distance(first: ConeCluster, second: ConeCluster) -> float:
    return math.hypot(
        first.center_x_m - second.center_x_m,
        first.center_y_m - second.center_y_m,
    )


def _point_gap(first: LidarPoint, second: LidarPoint) -> float:
    return math.hypot(first.x_m - second.x_m, first.y_m - second.y_m)


def _finite_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
