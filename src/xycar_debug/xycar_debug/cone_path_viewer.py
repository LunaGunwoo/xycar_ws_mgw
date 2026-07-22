# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

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
    DriveToggleConfig,
    ViewerConfig,
    default_viewer_tuning_path,
    load_viewer_config,
)


@dataclass
class LatestScan:
    snapshot: Optional[LaserScanSnapshot] = None
    received_monotonic: float = 0.0
    sequence: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class DriveGateStep:
    should_drive: bool
    enabled: bool
    forced_off: bool
    reason: str


class SpaceDriveGate:
    """Pure state gate for explicit Space-key drive control."""

    def __init__(self, config: DriveToggleConfig) -> None:
        self.config = config
        self.enabled = False
        self.loss_started_monotonic: Optional[float] = None
        self.reason = "DRIVE OFF: press Space when the cone path is valid"

    def toggle(
        self,
        *,
        now_monotonic: float,
        driveable: bool,
        decision_reason: str,
        competitors: Sequence[str],
    ) -> DriveGateStep:
        if self.enabled:
            self.enabled = False
            self.loss_started_monotonic = None
            self.reason = "DRIVE OFF: stopped by Space key"
            return DriveGateStep(False, False, True, self.reason)

        if competitors:
            names = ", ".join(competitors)
            self.reason = f"DRIVE OFF: competing motor publisher(s): {names}"
            return DriveGateStep(False, False, False, self.reason)
        if not driveable:
            self.reason = f"DRIVE OFF: activation rejected: {decision_reason}"
            return DriveGateStep(False, False, False, self.reason)

        self.enabled = True
        self.loss_started_monotonic = None
        self.reason = "DRIVE ON: enabled by Space key"
        return DriveGateStep(True, True, False, self.reason)

    def evaluate(
        self,
        *,
        now_monotonic: float,
        driveable: bool,
        decision_reason: str,
        competitors: Sequence[str],
    ) -> DriveGateStep:
        if not self.enabled:
            return DriveGateStep(False, False, False, self.reason)

        if competitors:
            names = ", ".join(competitors)
            self.enabled = False
            self.loss_started_monotonic = None
            self.reason = f"DRIVE OFF: competing motor publisher(s): {names}"
            return DriveGateStep(False, False, True, self.reason)

        if driveable:
            self.loss_started_monotonic = None
            self.reason = "DRIVE ON: following current cone path"
            return DriveGateStep(True, True, False, self.reason)

        if self.loss_started_monotonic is None:
            self.loss_started_monotonic = now_monotonic
        loss_age = now_monotonic - self.loss_started_monotonic
        if loss_age >= self.config.path_loss_timeout_sec:
            self.enabled = False
            self.loss_started_monotonic = None
            self.reason = (
                "DRIVE OFF: path unavailable for "
                f"{self.config.path_loss_timeout_sec:.2f} s"
            )
            return DriveGateStep(False, False, True, self.reason)

        self.reason = (
            "DRIVE ON / STOPPED: waiting for path recovery "
            f"({loss_age:.2f}/{self.config.path_loss_timeout_sec:.2f} s): "
            f"{decision_reason}"
        )
        return DriveGateStep(False, True, False, self.reason)

    def force_off(self, reason: str) -> DriveGateStep:
        was_enabled = self.enabled
        self.enabled = False
        self.loss_started_monotonic = None
        self.reason = f"DRIVE OFF: {reason}"
        return DriveGateStep(False, False, was_enabled, self.reason)


class ConePathViewerNode(Node):
    """Receive LaserScan data and own the optional debug motor output."""

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
        self.viewer_settings = load_viewer_config(viewer_path)
        self.latest = LatestScan()
        self.motor_publisher = self.create_publisher(
            Float32MultiArray,
            self.tuning.topics.motor_topic,
            1,
        )
        self.subscription = self.create_subscription(
            LaserScan,
            self.tuning.topics.scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.get_logger().warning(
            "Cone GUI ready with DRIVE OFF: "
            f"scan={self.tuning.topics.scan_topic}, "
            f"motor={self.tuning.topics.motor_topic}. "
            "Space toggles actual driving; Q/Esc stops and exits."
        )

    def _on_scan(self, message: LaserScan) -> None:
        try:
            self.latest.snapshot = LaserScanSnapshot(
                ranges=tuple(message.ranges),
                angle_min=float(message.angle_min),
                angle_increment=float(message.angle_increment),
                range_min=float(message.range_min),
                range_max=float(message.range_max),
            )
            self.latest.received_monotonic = time.monotonic()
            self.latest.sequence += 1
            self.latest.error = None
        except Exception as exc:
            self.latest.error = f"scan conversion exception: {exc}"

    def competing_motor_publishers(self) -> Tuple[str, ...]:
        topic = self.resolve_topic_name(self.tuning.topics.motor_topic)
        competitors = []
        for publisher in self.get_publishers_info_by_topic(topic):
            if (
                publisher.node_name == self.get_name()
                and publisher.node_namespace == self.get_namespace()
            ):
                continue
            namespace = publisher.node_namespace.rstrip("/")
            label = f"{namespace}/{publisher.node_name}"
            competitors.append(label if label.startswith("/") else f"/{label}")
        return tuple(sorted(set(competitors)))

    def publish_command(self, decision: DriveDecision) -> None:
        message = Float32MultiArray()
        message.data = [float(decision.angle), float(decision.speed)]
        self.motor_publisher.publish(message)

    def publish_stop(self) -> None:
        message = Float32MultiArray()
        message.data = [0.0, 0.0]
        self.motor_publisher.publish(message)

    def publish_stop_burst(self) -> None:
        for _ in range(self.viewer_settings.drive.stop_publish_count):
            self.publish_stop()


class ConePathViewer:
    def __init__(self, node: ConePathViewerNode) -> None:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.node = node
        self.config: ViewerConfig = node.viewer_settings.view
        self.drive_gate = SpaceDriveGate(node.viewer_settings.drive)
        self.tracker = ConePathTracker(
            node.tuning.cone_filter,
            node.tuning.cone_path,
        )
        self.last_sequence = -1
        self.last_space_monotonic = -math.inf
        self.last_control_monotonic = -math.inf
        self.control_period_sec = 1.0 / node.tuning.control.control_rate_hz
        self.plan = ConeLanePlan()
        self.raw_points = _empty_points()
        self.candidate_points = _empty_points()
        self.preview_decision = DriveDecision(reason="waiting for scan")
        self.actual_decision = DriveDecision(reason="drive disabled")
        self.competitors: Tuple[str, ...] = ()
        self.last_scan_age = math.inf
        self.closed = False
        self.last_gate_step = DriveGateStep(
            False,
            False,
            False,
            self.drive_gate.reason,
        )

        self.fig, (self.path_axis, self.status_axis) = plt.subplots(
            1,
            2,
            figsize=(12, 7),
            gridspec_kw={"width_ratios": [2.3, 1.0]},
        )
        self.fig.canvas.manager.set_window_title(
            "xycar_debug - LiDAR cone drive"
        )
        self._set_window_geometry()
        self._configure_path_axis()
        self._configure_status_axis()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        control_interval_ms = max(20, round(self.control_period_sec * 1000.0))
        self.timer = self.fig.canvas.new_timer(
            interval=min(self.config.update_interval_ms, control_interval_ms)
        )
        self.timer.add_callback(self._update)
        self.timer.start()

    def show(self) -> None:
        self.plt.show()

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.drive_gate.force_off("viewer closed")
        self.node.publish_stop_burst()

    def _configure_path_axis(self) -> None:
        axis = self.path_axis
        axis.set_title("LiDAR cone detection and commanded motion")
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
        self.status_axis.set_title("DRIVE OFF")
        self.status_text = self.status_axis.text(
            0.02,
            0.98,
            "Waiting for /scan",
            transform=self.status_axis.transAxes,
            family="monospace",
            fontsize=9.5,
            va="top",
        )

    def _update(self) -> None:
        if self.closed:
            return
        if not rclpy.ok():
            self.shutdown()
            self.plt.close(self.fig)
            return
        try:
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.0)
            self._process_latest_scan()
            self.preview_decision = self._current_preview_decision()
            self.competitors = self.node.competing_motor_publishers()
            now = time.monotonic()
            if now - self.last_control_monotonic >= self.control_period_sec:
                self.last_control_monotonic = now
                self.last_gate_step = self.drive_gate.evaluate(
                    now_monotonic=now,
                    driveable=self.preview_decision.can_drive,
                    decision_reason=self.preview_decision.reason,
                    competitors=self.competitors,
                )
                self._publish_gate_step(self.last_gate_step)
            step = self.last_gate_step
            self._update_artists(step)
        except Exception as exc:
            step = self.drive_gate.force_off(f"processing exception: {exc}")
            if step.forced_off:
                self.node.publish_stop_burst()
            self.actual_decision = DriveDecision(reason=step.reason)
            self.last_gate_step = step
            self.node.get_logger().error(step.reason)
            self._update_artists(step)
        self.fig.canvas.draw_idle()

    def _publish_gate_step(self, step: DriveGateStep) -> None:
        if step.forced_off:
            self.node.publish_stop_burst()
            self.node.get_logger().error(step.reason)
        elif step.should_drive:
            self.node.publish_command(self.preview_decision)
        elif step.enabled:
            self.node.publish_stop()
        self.actual_decision = (
            self.preview_decision
            if step.should_drive
            else DriveDecision(reason=step.reason)
        )

    def _process_latest_scan(self) -> None:
        latest = self.node.latest
        if latest.error is not None:
            raise RuntimeError(latest.error)
        if latest.snapshot is None or latest.sequence == self.last_sequence:
            return
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

    def _current_preview_decision(self) -> DriveDecision:
        latest = self.node.latest
        if latest.snapshot is None:
            self.last_scan_age = math.inf
            return DriveDecision(reason="waiting for scan", plan_mode=self.plan.mode)
        self.last_scan_age = max(
            0.0,
            time.monotonic() - latest.received_monotonic,
        )
        fresh = self.last_scan_age <= self.node.tuning.control.scan_timeout_sec
        return decide_drive(
            self.plan,
            self.node.tuning.control,
            drive_enabled=True,
            scan_fresh=fresh,
        )

    def _update_artists(self, step: DriveGateStep) -> None:
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
        width_text = (
            "n/a" if self.plan.width_m is None else f"{self.plan.width_m:.2f} m"
        )
        curvature_text = (
            "n/a"
            if self.plan.curvature is None
            else f"{self.plan.curvature:+.4f} 1/m"
        )
        scan_age_text = (
            "n/a" if not math.isfinite(self.last_scan_age) else f"{self.last_scan_age:.3f} s"
        )
        heading_deg = math.degrees(self.plan.path_heading_rad)
        drive_label = "DRIVE ON" if step.enabled else "DRIVE OFF"
        output_label = "MOVING" if step.should_drive else "STOPPED"
        competitor_text = ", ".join(self.competitors) or "none"
        self.status_axis.set_title(f"{drive_label} / {output_label}")
        self.status_text.set_text(
            f"{drive_label} / {output_label}\n"
            "====================\n\n"
            f"state:       {step.reason}\n"
            f"plan reason: {self.preview_decision.reason}\n"
            f"plan mode:   {self.plan.mode}\n"
            f"confidence:  {self.plan.confidence:.3f}\n"
            f"scan age:    {scan_age_text}\n"
            f"reused:      {self.plan.reused_frames}\n"
            f"competitor:  {competitor_text}\n\n"
            f"left cones:  {len(self.plan.left_boundary)}\n"
            f"right cones: {len(self.plan.right_boundary)}\n"
            f"lane width:  {width_text}\n"
            f"curvature:   {curvature_text}\n"
            f"heading:     {heading_deg:+.1f} deg\n\n"
            "preview [angle, speed]\n"
            f"[{self.preview_decision.angle:+.2f}, "
            f"{self.preview_decision.speed:.2f}]\n"
            "actual [angle, speed]\n"
            f"[{self.actual_decision.angle:+.2f}, "
            f"{self.actual_decision.speed:.2f}]\n\n"
            "Space: toggle actual drive\n"
            "Q / Esc: stop and close"
        )

    def _set_window_geometry(self) -> None:
        manager = self.fig.canvas.manager
        window = getattr(manager, "window", None)
        if window is not None and hasattr(window, "wm_geometry"):
            window.wm_geometry(self.config.window_geometry)

    def _on_key(self, event) -> None:
        if event.key in {"q", "escape"}:
            self.shutdown()
            self.plt.close(self.fig)
            return
        if event.key not in {" ", "space"}:
            return
        now = time.monotonic()
        if (
            now - self.last_space_monotonic
            < self.node.viewer_settings.drive.key_debounce_sec
        ):
            return
        self.last_space_monotonic = now
        self.preview_decision = self._current_preview_decision()
        self.competitors = self.node.competing_motor_publishers()
        step = self.drive_gate.toggle(
            now_monotonic=now,
            driveable=self.preview_decision.can_drive,
            decision_reason=self.preview_decision.reason,
            competitors=self.competitors,
        )
        self.last_control_monotonic = now
        self.last_gate_step = step
        if step.forced_off:
            self.node.publish_stop_burst()
            self.node.get_logger().warning(step.reason)
        elif step.should_drive:
            self.node.publish_command(self.preview_decision)
            self.node.get_logger().warning(step.reason)
        else:
            self.node.get_logger().error(step.reason)
        self.actual_decision = (
            self.preview_decision
            if step.should_drive
            else DriveDecision(reason=step.reason)
        )
        self._update_artists(step)
        self.fig.canvas.draw_idle()

    def _on_close(self, _event) -> None:
        self.shutdown()


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
    viewer: Optional[ConePathViewer] = None
    try:
        node = ConePathViewerNode()
        viewer = ConePathViewer(node)
        viewer.show()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.shutdown()
        elif node is not None:
            node.publish_stop_burst()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
