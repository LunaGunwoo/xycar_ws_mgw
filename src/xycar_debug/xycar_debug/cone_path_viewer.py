# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from track_drive.cone_following import (
    ConeLanePlan,
    ConePathTracker,
    LaserScanSnapshot,
    cone_candidate_points,
    scan_points,
)
from track_drive.control import DriveDecision, decide_drive
from track_drive.tuning import default_tuning_path, load_tuning
from xycar_debug.viewer_tuning import (
    ViewerConfig,
    default_viewer_tuning_path,
    load_viewer_config,
)


@dataclass
class LatestScan:
    snapshot: Optional[LaserScanSnapshot] = None
    received_monotonic: float = 0.0
    sequence: int = 0


class ConePathViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("cone_path_viewer")
        self.declare_parameter("tuning_file", default_tuning_path())
        self.declare_parameter(
            "viewer_tuning_file",
            default_viewer_tuning_path(),
        )
        tuning_path = str(self.get_parameter("tuning_file").value)
        viewer_path = str(self.get_parameter("viewer_tuning_file").value)
        self.tuning = load_tuning(tuning_path)
        self.viewer_config = load_viewer_config(viewer_path)
        self.latest = LatestScan()
        self.subscription = self.create_subscription(
            LaserScan,
            self.tuning.topics.scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "View-only cone path GUI ready on "
            f"{self.tuning.topics.scan_topic}; no motor output exists"
        )

    def _on_scan(self, message: LaserScan) -> None:
        self.latest.snapshot = LaserScanSnapshot(
            ranges=tuple(message.ranges),
            angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),
            range_min=float(message.range_min),
            range_max=float(message.range_max),
        )
        self.latest.received_monotonic = time.monotonic()
        self.latest.sequence += 1


class ConePathViewer:
    def __init__(self, node: ConePathViewerNode) -> None:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.node = node
        self.config: ViewerConfig = node.viewer_config
        self.tracker = ConePathTracker(
            node.tuning.cone_filter,
            node.tuning.cone_path,
        )
        self.last_sequence = -1
        self.plan = ConeLanePlan()
        self.raw_points = _empty_points()
        self.candidate_points = _empty_points()

        self.fig, (self.path_axis, self.status_axis) = plt.subplots(
            1,
            2,
            figsize=(12, 7),
            gridspec_kw={"width_ratios": [2.3, 1.0]},
        )
        self.fig.canvas.manager.set_window_title(
            "xycar_debug - LiDAR cone path preview"
        )
        self._set_window_geometry()
        self._configure_path_axis()
        self._configure_status_axis()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(
            interval=self.config.update_interval_ms
        )
        self.timer.add_callback(self._update)
        self.timer.start()

    def show(self) -> None:
        self.plt.show()

    def _configure_path_axis(self) -> None:
        axis = self.path_axis
        axis.set_title("LiDAR cone detection and predicted motion")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(
            -self.config.display_abs_x_m,
            self.config.display_abs_x_m,
        )
        axis.set_ylim(
            self.config.display_y_min_m,
            self.config.display_y_max_m,
        )
        axis.set_xlabel("x [m]  (negative: left, positive: right)")
        axis.set_ylabel("y forward [m]")
        axis.grid(True, color="0.88")
        self.raw_scatter = axis.scatter(
            [],
            [],
            s=self.config.raw_point_size,
            color="0.72",
            alpha=0.55,
            label="raw scan",
        )
        self.candidate_scatter = axis.scatter(
            [],
            [],
            s=self.config.candidate_point_size,
            color="tab:purple",
            alpha=0.75,
            label="filtered points",
        )
        self.left_scatter = axis.scatter(
            [],
            [],
            s=self.config.cone_point_size,
            color="tab:blue",
            label="left cones",
        )
        self.right_scatter = axis.scatter(
            [],
            [],
            s=self.config.cone_point_size,
            color="tab:orange",
            label="right cones",
        )
        (self.left_line,) = axis.plot(
            [], [], color="tab:blue", linewidth=2.0, label="left boundary"
        )
        (self.right_line,) = axis.plot(
            [], [], color="tab:orange", linewidth=2.0, label="right boundary"
        )
        (self.center_line,) = axis.plot(
            [], [], color="tab:green", linewidth=2.5, label="center path"
        )
        (self.trajectory_line,) = axis.plot(
            [],
            [],
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label="predicted arc",
        )
        self.target_scatter = axis.scatter(
            [],
            [],
            s=120,
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            label="lookahead",
        )
        axis.plot([0.0, 0.0], [0.0, 0.45], color="black", linewidth=4)
        axis.scatter([0.0], [0.0], s=55, color="black", label="vehicle")
        axis.legend(loc="upper left", fontsize=8)

    def _configure_status_axis(self) -> None:
        self.status_axis.set_axis_off()
        self.status_axis.set_title("PREVIEW ONLY - no motor publisher")
        self.status_text = self.status_axis.text(
            0.02,
            0.98,
            "Waiting for /scan",
            transform=self.status_axis.transAxes,
            family="monospace",
            fontsize=10,
            va="top",
        )

    def _update(self) -> None:
        if not rclpy.ok():
            self.plt.close(self.fig)
            return
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.0)

        latest = self.node.latest
        if latest.snapshot is None:
            self.status_text.set_text(
                "PREVIEW ONLY\n\nstate: WAITING_FOR_SCAN\ncommand: [0.0, 0.0]"
            )
            self.fig.canvas.draw_idle()
            return

        if latest.sequence != self.last_sequence:
            self.last_sequence = latest.sequence
            self.plan = self.tracker.update(latest.snapshot)
            self.raw_points = _visible_points(
                scan_points(latest.snapshot, self.node.tuning.cone_filter),
                self.config,
            )
            self.candidate_points = _visible_points(
                cone_candidate_points(
                    latest.snapshot,
                    self.node.tuning.cone_filter,
                ),
                self.config,
            )

        age = max(0.0, time.monotonic() - latest.received_monotonic)
        fresh = age <= self.node.tuning.control.scan_timeout_sec
        decision = decide_drive(
            self.plan,
            self.node.tuning.control,
            drive_enabled=True,
            scan_fresh=fresh,
        )
        self._update_artists(decision, age)
        self.fig.canvas.draw_idle()

    def _update_artists(
        self,
        decision: DriveDecision,
        scan_age: float,
    ) -> None:
        self.raw_scatter.set_offsets(self.raw_points)
        self.candidate_scatter.set_offsets(self.candidate_points)
        left = _clusters_array(self.plan.left_boundary)
        right = _clusters_array(self.plan.right_boundary)
        center = _points_array(self.plan.waypoints)
        self.left_scatter.set_offsets(left)
        self.right_scatter.set_offsets(right)
        self.left_line.set_data(_xy(left))
        self.right_line.set_data(_xy(right))
        self.center_line.set_data(_xy(center))
        target = self.plan.lookahead_point
        self.target_scatter.set_offsets(
            _points_array((target,)) if target is not None else _empty_points()
        )
        trajectory = _trajectory_points(
            self.plan.curvature,
            self.config.trajectory_horizon_m,
            self.config.trajectory_step_m,
        )
        self.trajectory_line.set_data(_xy(trajectory))
        width_text = "n/a" if self.plan.width_m is None else f"{self.plan.width_m:.2f} m"
        curvature_text = (
            "n/a"
            if self.plan.curvature is None
            else f"{self.plan.curvature:+.4f} 1/m"
        )
        heading_deg = math.degrees(self.plan.path_heading_rad)
        state = "WOULD_DRIVE" if decision.can_drive else "WOULD_STOP"
        self.status_text.set_text(
            "PREVIEW ONLY\n"
            "============\n\n"
            f"state:       {state}\n"
            f"reason:      {decision.reason}\n"
            f"plan mode:   {self.plan.mode}\n"
            f"confidence:  {self.plan.confidence:.3f}\n"
            f"scan age:    {scan_age:.3f} s\n"
            f"reused:      {self.plan.reused_frames}\n\n"
            f"left cones:  {len(self.plan.left_boundary)}\n"
            f"right cones: {len(self.plan.right_boundary)}\n"
            f"lane width:  {width_text}\n"
            f"curvature:   {curvature_text}\n"
            f"heading:     {heading_deg:+.1f} deg\n\n"
            "predicted command\n"
            f"angle:       {decision.angle:+.2f}\n"
            f"speed:       {decision.speed:.2f}\n\n"
            "Press Q or Esc to close"
        )

    def _set_window_geometry(self) -> None:
        manager = self.fig.canvas.manager
        window = getattr(manager, "window", None)
        if window is not None and hasattr(window, "wm_geometry"):
            window.wm_geometry(self.config.window_geometry)

    def _on_key(self, event) -> None:
        if event.key in {"q", "escape"}:
            self.plt.close(self.fig)


def _visible_points(points, config: ViewerConfig) -> np.ndarray:
    return _points_array(
        tuple(
            (point.x_m, point.y_m)
            for point in points
            if abs(point.x_m) <= config.display_abs_x_m
            and config.display_y_min_m <= point.y_m <= config.display_y_max_m
        )
    )


def _clusters_array(clusters) -> np.ndarray:
    return _points_array(
        tuple((item.center_x_m, item.center_y_m) for item in clusters)
    )


def _points_array(points) -> np.ndarray:
    if not points:
        return _empty_points()
    return np.asarray(points, dtype=float).reshape((-1, 2))


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=float)


def _xy(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.asarray([]), np.asarray([])
    return points[:, 0], points[:, 1]


def _trajectory_points(
    curvature: Optional[float],
    horizon_m: float,
    step_m: float,
) -> np.ndarray:
    if curvature is None:
        return _empty_points()
    distances = np.arange(0.0, horizon_m + step_m, step_m)
    if abs(curvature) < 1e-6:
        return np.column_stack((np.zeros_like(distances), distances))
    x_values = (1.0 - np.cos(curvature * distances)) / curvature
    y_values = np.sin(curvature * distances) / curvature
    return np.column_stack((x_values, y_values))


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ConePathViewerNode] = None
    try:
        node = ConePathViewerNode()
        viewer = ConePathViewer(node)
        viewer.show()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
