# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""TTY keyboard teleop with camera-first behavior-cloning dataset capture."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import time
from typing import Optional, Sequence

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray

from xycar_data.session_writer import (
    AsyncSessionWriter,
    CameraSample,
    LidarSnapshot,
)
from xycar_data.terminal_input import TerminalKeyReader
from xycar_data.tuning import (
    ControlConfig,
    TeleopTuning,
    default_tuning_path,
    load_tuning,
    tuning_as_mapping,
)


@dataclass(frozen=True)
class DriveCommand:
    angle: float = 0.0
    speed: float = 0.0
    input_key: str = ""

    @property
    def is_stop(self) -> bool:
        return self.speed == 0.0 and self.angle == 0.0


class KeyboardCommandState:
    """Independent speed/steering state with terminal timeout release."""

    def __init__(self, config: ControlConfig) -> None:
        self.config = config
        self.command = DriveCommand()
        self.last_input_monotonic = -math.inf

    def set_drive_key(self, key: str, now_monotonic: float) -> DriveCommand:
        angle = self.command.angle
        speed = self.command.speed
        if key == "w":
            speed = self.config.forward_speed
        elif key == "W":
            speed = (
                self.config.forward_speed
                * self.config.forward_boost_multiplier
            )
        elif key == "s":
            speed = self.config.reverse_speed
        elif key == "a":
            angle = _clamp(
                angle - self.config.angle_step,
                self.config.min_angle,
                self.config.max_angle,
            )
        elif key == "d":
            angle = _clamp(
                angle + self.config.angle_step,
                self.config.min_angle,
                self.config.max_angle,
            )
        else:
            raise ValueError(f"unsupported drive key: {key}")
        self.command = DriveCommand(float(angle), float(speed), key)
        self.last_input_monotonic = now_monotonic
        return self.command

    def is_active(self, now_monotonic: float) -> bool:
        return (
            not self.command.is_stop
            and now_monotonic - self.last_input_monotonic
            <= self.config.key_timeout_sec
        )

    def clear(self) -> DriveCommand:
        self.command = DriveCommand()
        self.last_input_monotonic = -math.inf
        return self.command

    def expire_if_stale(self, now_monotonic: float) -> bool:
        if self.command.is_stop or self.is_active(now_monotonic):
            return False
        self.clear()
        return True


@dataclass(frozen=True)
class _CameraFrame:
    image: np.ndarray
    sequence: int
    stamp_sec: int
    stamp_nanosec: int
    received_monotonic: float
    received_wall_time_ns: int


@dataclass(frozen=True)
class _PendingFinish:
    token: int
    reason: str
    complete: bool


class TeleopRecorderNode(Node):
    """ROS I/O, safety gates, and callback-to-writer data handoff."""

    def __init__(self) -> None:
        super().__init__("teleop_recorder")
        self.declare_parameter("tuning_file", default_tuning_path())
        self.tuning_path = str(self.get_parameter("tuning_file").value)
        self.tuning: TeleopTuning = load_tuning(self.tuning_path)

        self.bridge = CvBridge()
        self.command_state = KeyboardCommandState(self.tuning.control)
        self._last_published = DriveCommand()
        self._last_publish_monotonic = -math.inf
        self._camera: Optional[_CameraFrame] = None
        self._lidar: Optional[LidarSnapshot] = None
        self._camera_sequence = 0
        self._lidar_sequence = 0
        self._competitors: tuple[str, ...] = ()
        self._has_motor_subscriber = False
        self._next_graph_check_monotonic = 0.0
        self._session_token: Optional[int] = None
        self._finishing_token: Optional[int] = None
        self._pending_finish: Optional[_PendingFinish] = None
        self._dropped_sample_count = 0
        self._last_lidar_missing_warning_monotonic = -math.inf
        self.exit_requested = False
        self.exit_code = 0
        self._fatal_reason: Optional[str] = None

        recording = self.tuning.recording
        self.writer = AsyncSessionWriter(
            recording.root_dir,
            png_compression=recording.png_compression,
            queue_size=recording.queue_size,
            min_free_space_mb=recording.min_free_space_mb,
        )
        self.motor_publisher = self.create_publisher(
            Float32MultiArray,
            self.tuning.topics.motor_topic,
            10,
        )
        self.camera_subscription = self.create_subscription(
            Image,
            self.tuning.topics.camera_topic,
            self._on_camera,
            qos_profile_sensor_data,
        )
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            self.tuning.topics.lidar_topic,
            self._on_lidar,
            qos_profile_sensor_data,
        )
        self.control_timer = self.create_timer(
            1.0 / self.tuning.control.publish_rate_hz,
            self._on_control_timer,
        )
        self._refresh_graph(time.monotonic())
        self.get_logger().warning(
            "Teleop recorder is armed but stopped. Camera frames are primary "
            "training samples; LiDAR is optional metadata. "
            "E starts a dataset session, Q saves it, Space stops, Esc exits."
        )
        self.get_logger().info(
            f"camera={self.tuning.topics.camera_topic}, "
            f"lidar={self.tuning.topics.lidar_topic} (optional), "
            f"motor={self.tuning.topics.motor_topic}, "
            f"dataset_root={recording.root_dir}"
        )

    def handle_key(self, key: str, now_monotonic: Optional[float] = None) -> None:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if key in ("w", "W", "s", "a", "d"):
            self._handle_drive_key(key, now)
        elif key == "e":
            self._start_session()
        elif key == "q":
            self._close_active_session(
                "operator ended session with Q",
                complete=True,
            )
        elif key == "space":
            self.stop_motion("stopped by Space", burst=True)
        elif key in ("escape", "ctrl_c"):
            self.request_exit("operator requested exit", complete=True)

    def request_exit(self, reason: str, *, complete: bool) -> None:
        if self.exit_requested:
            return
        self.exit_requested = True
        self.stop_motion(reason, burst=True)
        self._close_active_session(reason, complete=complete)

    def exit_ready(self) -> bool:
        return (
            self.exit_requested
            and self._session_token is None
            and self._finishing_token is None
            and self._pending_finish is None
        )

    def shutdown(self) -> None:
        self.stop_motion("shutdown", burst=True)
        if self._session_token is not None:
            self._close_active_session("shutdown", complete=False)
        self.writer.shutdown()

    def stop_motion(self, reason: str, *, burst: bool = False) -> None:
        self.command_state.clear()
        self.publish_stop()
        if burst:
            for _ in range(self.tuning.control.stop_publish_count - 1):
                self.publish_stop()
        self.get_logger().info(f"Teleop stop: {reason}")

    def publish_stop(self) -> None:
        self._publish(DriveCommand())

    def _handle_drive_key(self, key: str, now_monotonic: float) -> None:
        self._refresh_graph(now_monotonic)
        if self._competitors:
            self.stop_motion("competing motor publisher", burst=True)
            self.get_logger().error(
                "Drive rejected: competing motor publisher(s): "
                + ", ".join(self._competitors)
            )
            return
        if not self._has_motor_subscriber:
            self.stop_motion("no motor subscriber", burst=True)
            self.get_logger().warning(
                "Drive rejected: no subscriber is connected to the motor topic."
            )
            return
        if not self._camera_is_fresh(now_monotonic):
            self.stop_motion("camera is stale", burst=True)
            self.get_logger().warning(
                "Drive rejected: waiting for a fresh camera frame."
            )
            return
        command = self.command_state.set_drive_key(key, now_monotonic)
        self._publish(command)

    def _start_session(self) -> None:
        if self._session_token is not None:
            self.get_logger().warning("A recording session is already active.")
            return
        if self._finishing_token is not None:
            self.get_logger().warning("Previous session is still flushing to disk.")
            return
        metadata = {
            "dataset_kind": "camera_first_teleop_behavior_cloning",
            "camera_is_primary": True,
            "lidar_is_optional": True,
            "topics": tuning_as_mapping(self.tuning)["topics"],
            "tuning": tuning_as_mapping(self.tuning),
            "tuning_file": self.tuning_path,
            "dropped_sample_count": 0,
        }
        token = self.writer.start_session(metadata)
        if token is None:
            self._fail("could not queue a new recording session")
            return
        self._session_token = token
        self._dropped_sample_count = 0
        self.get_logger().info(
            "Recording session armed. Data is saved only while a WASD "
            "command is active and a fresh camera frame arrives."
        )

    def _close_active_session(self, reason: str, *, complete: bool) -> None:
        if self._session_token is None:
            return
        token = self._session_token
        self._session_token = None
        self._finishing_token = token
        self._pending_finish = _PendingFinish(token, reason, complete)
        self._retry_pending_finish()

    def _retry_pending_finish(self) -> None:
        pending = self._pending_finish
        if pending is None:
            return
        if self.writer.failure is not None:
            self._pending_finish = None
            return
        if self.writer.finish(
            pending.token,
            pending.reason,
            complete=pending.complete,
            extra_metadata={"dropped_sample_count": self._dropped_sample_count},
        ):
            self._pending_finish = None

    def _on_camera(self, message: Image) -> None:
        now_monotonic = time.monotonic()
        now_wall_time_ns = time.time_ns()
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(
                f"Camera image conversion failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return
        self._camera_sequence += 1
        stamp = message.header.stamp
        frame = _CameraFrame(
            image=np.ascontiguousarray(image).copy(),
            sequence=self._camera_sequence,
            stamp_sec=int(stamp.sec),
            stamp_nanosec=int(stamp.nanosec),
            received_monotonic=now_monotonic,
            received_wall_time_ns=now_wall_time_ns,
        )
        self._camera = frame
        self._enqueue_camera_sample(frame)

    def _on_lidar(self, message: LaserScan) -> None:
        now_monotonic = time.monotonic()
        now_wall_time_ns = time.time_ns()
        self._lidar_sequence += 1
        stamp = message.header.stamp
        self._lidar = LidarSnapshot(
            sequence=self._lidar_sequence,
            ranges=np.asarray(message.ranges, dtype=np.float32).copy(),
            intensities=np.asarray(message.intensities, dtype=np.float32).copy(),
            angle_min=float(message.angle_min),
            angle_max=float(message.angle_max),
            angle_increment=float(message.angle_increment),
            time_increment=float(message.time_increment),
            scan_time=float(message.scan_time),
            range_min=float(message.range_min),
            range_max=float(message.range_max),
            frame_id=str(message.header.frame_id),
            stamp_sec=int(stamp.sec),
            stamp_nanosec=int(stamp.nanosec),
            received_monotonic=now_monotonic,
            received_wall_time_ns=now_wall_time_ns,
        )

    def _enqueue_camera_sample(self, frame: _CameraFrame) -> None:
        now = frame.received_monotonic
        if self._session_token is None or self.exit_requested:
            return
        if not self.command_state.is_active(now):
            return
        command = self.command_state.command
        if self._last_published != command:
            return
        if not _is_recordable_command(command):
            return
        if not self._has_motor_subscriber or self._competitors:
            return
        lidar, skew = self._matching_lidar(frame)
        if lidar is None and (
            now - self._last_lidar_missing_warning_monotonic >= 2.0
        ):
            self._last_lidar_missing_warning_monotonic = now
            self.get_logger().warning(
                "LiDAR is unavailable or too old for this camera frame; "
                "saving the camera sample with lidar_valid=false."
            )
        sample = CameraSample(
            image=frame.image,
            camera_sequence=frame.sequence,
            camera_stamp_sec=frame.stamp_sec,
            camera_stamp_nanosec=frame.stamp_nanosec,
            camera_received_monotonic=frame.received_monotonic,
            camera_received_wall_time_ns=frame.received_wall_time_ns,
            angle=command.angle,
            speed=command.speed,
            input_key=command.input_key,
            lidar=lidar,
            lidar_skew_sec=skew,
        )
        if not self.writer.submit(self._session_token, sample):
            self._dropped_sample_count += 1
            self._fail("dataset writer queue is full or unavailable")

    def _matching_lidar(
        self,
        frame: _CameraFrame,
    ) -> tuple[Optional[LidarSnapshot], Optional[float]]:
        lidar = self._lidar
        if lidar is None:
            return None, None
        age = frame.received_monotonic - lidar.received_monotonic
        if age < 0.0 or age > self.tuning.sensors.lidar_timeout_sec:
            return None, None
        skew = abs(age)
        if skew > self.tuning.sensors.max_lidar_skew_sec:
            return None, None
        return lidar, skew

    def _on_control_timer(self) -> None:
        now = time.monotonic()
        if self.writer.failure is not None:
            self._fail(self.writer.failure)
        self._poll_writer_results()
        self._retry_pending_finish()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)

        self.command_state.expire_if_stale(now)
        if not self.command_state.is_active(now):
            self.publish_stop()
            return
        if self._competitors:
            self._fail(
                "competing motor publisher appeared during teleop: "
                + ", ".join(self._competitors)
            )
            return
        if not self._has_motor_subscriber:
            self.stop_motion("motor subscriber disappeared", burst=True)
            return
        if not self._camera_is_fresh(now):
            self.stop_motion("camera stream is stale", burst=True)
            return
        self._publish(self.command_state.command)

    def _poll_writer_results(self) -> None:
        for result in self.writer.poll_results():
            if result.token != self._finishing_token:
                continue
            self._finishing_token = None
            if result.completed:
                path_text = "no files (no eligible camera samples)"
                if result.path is not None:
                    path_text = str(result.path)
                self.get_logger().info(
                    "Session saved: "
                    f"{path_text}; samples={result.sample_count}, "
                    f"lidar_linked={result.lidar_linked_count}, "
                    f"lidar_missing={result.lidar_missing_count}"
                )
            else:
                self.exit_code = 1
                self.get_logger().error(
                    "Session marked incomplete: "
                    f"{result.path or 'no files'}; reason={result.reason}"
                )

    def _refresh_graph(self, now_monotonic: float) -> None:
        self._next_graph_check_monotonic = (
            now_monotonic + self.tuning.control.graph_check_period_sec
        )
        topic = self.resolve_topic_name(self.tuning.topics.motor_topic)
        competitors = []
        for publisher in self.get_publishers_info_by_topic(topic):
            if (
                publisher.node_name == self.get_name()
                and publisher.node_namespace == self.get_namespace()
            ):
                continue
            competitors.append(_node_label(publisher.node_namespace, publisher.node_name))
        self._competitors = tuple(sorted(set(competitors)))
        self._has_motor_subscriber = bool(
            self.get_subscriptions_info_by_topic(topic)
        )

    def _camera_is_fresh(self, now_monotonic: float) -> bool:
        return (
            self._camera is not None
            and now_monotonic - self._camera.received_monotonic
            <= self.tuning.sensors.camera_timeout_sec
        )

    def _publish(self, command: DriveCommand) -> None:
        message = Float32MultiArray()
        message.data = [float(command.angle), float(command.speed)]
        self.motor_publisher.publish(message)
        self._last_published = command
        self._last_publish_monotonic = time.monotonic()

    def _fail(self, reason: str) -> None:
        if self._fatal_reason is not None:
            return
        self._fatal_reason = reason
        self.exit_code = 1
        self.get_logger().error(f"Teleop recorder failure: {reason}")
        self.request_exit(reason, complete=False)


def _node_label(namespace: str, name: str) -> str:
    prefix = namespace.rstrip("/")
    label = f"{prefix}/{name}"
    return label if label.startswith("/") else f"/{label}"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), float(minimum)), float(maximum))


def _is_recordable_command(command: DriveCommand) -> bool:
    return command.speed != 0.0


def _print_controls() -> None:
    print(
        "\nCLI teleop controls: W=+8; Shift+W=+12; S=-8; A/D adjust "
        "steering; E starts recording; Q saves recording; Space stops; "
        "Esc exits.\n"
        "Camera frames are the AI inputs. LiDAR is optional metadata only.\n",
        flush=True,
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    TerminalKeyReader.require_tty()
    rclpy.init(args=args)
    node: Optional[TeleopRecorderNode] = None
    exit_wait_started: Optional[float] = None
    try:
        node = TeleopRecorderNode()
        _print_controls()
        with TerminalKeyReader() as reader:
            while rclpy.ok():
                now = time.monotonic()
                try:
                    for key in reader.poll(now):
                        node.handle_key(key, now)
                except EOFError:
                    node.request_exit("terminal input closed", complete=False)
                    node.exit_code = 1

                rclpy.spin_once(node, timeout_sec=0.02)
                if node.exit_requested:
                    if exit_wait_started is None:
                        exit_wait_started = time.monotonic()
                    if node.exit_ready():
                        break
                    if time.monotonic() - exit_wait_started > 15.0:
                        node.exit_code = 1
                        node.get_logger().error(
                            "Timed out waiting for dataset writer finalization."
                        )
                        break
    except KeyboardInterrupt:
        if node is not None:
            node.request_exit("Ctrl+C", complete=True)
            deadline = time.monotonic() + 15.0
            while rclpy.ok() and not node.exit_ready() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
    finally:
        exit_code = 1 if node is None else node.exit_code
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
