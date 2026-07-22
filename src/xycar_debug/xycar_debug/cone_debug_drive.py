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

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from track_drive.cone_following import (
    ConeLanePlan,
    ConePathTracker,
    LaserScanSnapshot,
)
from track_drive.control import DriveDecision, decide_drive
from track_drive.tuning import default_tuning_path, load_tuning
from xycar_debug.debug_drive_tuning import (
    DebugDriveConfig,
    default_debug_drive_tuning_path,
    load_debug_drive_config,
)


class MissionPhase(str, Enum):
    WAITING = "waiting_for_cones"
    DRIVING = "following_cones"
    COMPLETE = "mission_complete"


@dataclass(frozen=True)
class MissionStep:
    should_drive: bool
    completed: bool
    reason: str


class ConeMissionState:
    """Pure state gate for one automatically started cone mission."""

    def __init__(self, config: DebugDriveConfig) -> None:
        self.config = config
        self.phase = MissionPhase.WAITING
        self.valid_frame_count = 0
        self.loss_started_monotonic: Optional[float] = None

    def update(
        self,
        driveable: bool,
        now_monotonic: float,
        new_scan_frame: bool,
    ) -> MissionStep:
        if self.phase == MissionPhase.COMPLETE:
            return MissionStep(False, True, "cone mission already complete")

        if self.phase == MissionPhase.WAITING:
            if new_scan_frame:
                if driveable:
                    self.valid_frame_count += 1
                else:
                    self.valid_frame_count = 0
            elif not driveable:
                self.valid_frame_count = 0

            if self.valid_frame_count >= self.config.start_valid_frames:
                self.phase = MissionPhase.DRIVING
                self.loss_started_monotonic = None
                return MissionStep(True, False, "cone mission started")
            return MissionStep(
                False,
                False,
                "waiting for consecutive valid cone frames",
            )

        if driveable:
            self.loss_started_monotonic = None
            return MissionStep(True, False, "following current cone path")

        if self.loss_started_monotonic is None:
            self.loss_started_monotonic = now_monotonic
        elapsed = now_monotonic - self.loss_started_monotonic
        if elapsed >= self.config.path_loss_timeout_sec:
            self.phase = MissionPhase.COMPLETE
            return MissionStep(False, True, "cone path ended")
        return MissionStep(False, False, "cone path temporarily unavailable")


class ConeDebugDriveNode(Node):
    """Run one low-speed cone-following mission and then stop and exit."""

    def __init__(self) -> None:
        super().__init__("cone_debug_drive")
        self.declare_parameter("tuning_file", default_tuning_path())
        self.declare_parameter(
            "debug_drive_tuning_file",
            default_debug_drive_tuning_path(),
        )
        tuning_path = str(self.get_parameter("tuning_file").value)
        mission_path = str(
            self.get_parameter("debug_drive_tuning_file").value
        )
        self.tuning = load_tuning(tuning_path)
        self.mission_config = load_debug_drive_config(mission_path)
        self.tracker = ConePathTracker(
            self.tuning.cone_filter,
            self.tuning.cone_path,
        )
        self.mission = ConeMissionState(self.mission_config)
        self.latest_plan = ConeLanePlan()
        self.latest_scan_monotonic: Optional[float] = None
        self.scan_sequence = 0
        self.processed_scan_sequence = 0
        self.finished = False
        self.exit_code = 0

        self.motor_publisher = self.create_publisher(
            Float32MultiArray,
            self.tuning.topics.motor_topic,
            1,
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            self.tuning.topics.scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        period = 1.0 / self.tuning.control.control_rate_hz
        self.control_timer = self.create_timer(period, self._on_control_timer)
        self._publish_stop()
        self.get_logger().warning(
            "Cone debug mission ready for AUTO START: "
            f"scan={self.tuning.topics.scan_topic}, "
            f"motor={self.tuning.topics.motor_topic}, "
            f"speed={self.tuning.control.sharp_turn_speed:.1f}.."
            f"{self.tuning.control.max_speed:.1f}. "
            f"Driving begins after {self.mission_config.start_valid_frames} "
            "consecutive valid LaserScan frames."
        )

    def _on_scan(self, message: LaserScan) -> None:
        if self.finished:
            return
        try:
            snapshot = LaserScanSnapshot(
                ranges=tuple(message.ranges),
                angle_min=float(message.angle_min),
                angle_increment=float(message.angle_increment),
                range_min=float(message.range_min),
                range_max=float(message.range_max),
            )
            self.latest_plan = self.tracker.update(snapshot)
            self.latest_scan_monotonic = time.monotonic()
            self.scan_sequence += 1
        except Exception as exc:
            self._finish(False, f"scan processing exception: {exc}")

    def _on_control_timer(self) -> None:
        if self.finished:
            return
        try:
            competitors = self._competing_motor_publishers()
            if competitors:
                names = ", ".join(competitors)
                self._finish(False, f"competing motor publisher(s): {names}")
                return

            now = time.monotonic()
            decision = self._current_decision()
            new_scan = self.scan_sequence != self.processed_scan_sequence
            if new_scan:
                self.processed_scan_sequence = self.scan_sequence
            previous_phase = self.mission.phase
            step = self.mission.update(
                decision.can_drive,
                now,
                new_scan,
            )
            if step.completed:
                self._finish(True, step.reason)
                return
            if step.should_drive:
                if previous_phase == MissionPhase.WAITING:
                    self.get_logger().warning("Cone debug mission AUTO STARTED")
                self._publish_command(decision)
            else:
                self._publish_stop()
        except Exception as exc:
            self._finish(False, f"control exception: {exc}")

    def _current_decision(self) -> DriveDecision:
        return decide_drive(
            self.latest_plan,
            self.tuning.control,
            drive_enabled=True,
            scan_fresh=self._scan_is_fresh(),
        )

    def _scan_is_fresh(self) -> bool:
        if self.latest_scan_monotonic is None:
            return False
        age = time.monotonic() - self.latest_scan_monotonic
        return age <= self.tuning.control.scan_timeout_sec

    def _competing_motor_publishers(self):
        topic = self.resolve_topic_name(self.tuning.topics.motor_topic)
        publishers = self.get_publishers_info_by_topic(topic)
        competitors = []
        own_name = self.get_name()
        own_namespace = self.get_namespace()
        for publisher in publishers:
            if (
                publisher.node_name == own_name
                and publisher.node_namespace == own_namespace
            ):
                continue
            competitors.append(
                f"{publisher.node_namespace.rstrip('/')}/{publisher.node_name}"
            )
        return sorted(set(competitors))

    def _publish_command(self, decision: DriveDecision) -> None:
        message = Float32MultiArray()
        message.data = [float(decision.angle), float(decision.speed)]
        self.motor_publisher.publish(message)

    def _publish_stop(self) -> None:
        message = Float32MultiArray()
        message.data = [0.0, 0.0]
        self.motor_publisher.publish(message)

    def publish_stop_burst(self) -> None:
        for _ in range(self.mission_config.stop_publish_count):
            self._publish_stop()

    def _finish(self, success: bool, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.exit_code = 0 if success else 1
        self.publish_stop_burst()
        if success:
            self.get_logger().info(f"Cone debug mission complete: {reason}")
        else:
            self.get_logger().error(f"Cone debug mission failed: {reason}")
        self.control_timer.cancel()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ConeDebugDriveNode] = None
    exit_code = 0
    try:
        node = ConeDebugDriveNode()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.10)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if not node.finished:
                node.publish_stop_burst()
            exit_code = node.exit_code
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
