# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Collect flat traffic-signal image classes while driving by gamepad."""

from __future__ import annotations

from dataclasses import dataclass
import math
import signal
import time
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float32MultiArray

from xycar_data.class_image_writer import (
    AsyncClassImageWriter,
    ClassImageSample,
)
from xycar_data.gamepad_teleop import (
    STOP_COMMAND,
    DriveCommand,
    GamepadConfig,
    InvalidJoyInput,
    NeutralArmingGate,
    _is_paired_unnamed_relay,
    _node_label,
    _validate_collection_profile,
    _validate_config,
    _validate_runtime_parameters,
    is_input_fresh,
    map_joy_input,
)
from xycar_data.steering_contract import require_steering_contract_name


SIGNAL_CLASSES = (
    'red',
    'yellow',
    'straight_green',
    'left_green',
)


@dataclass(frozen=True)
class SignalSelection:
    """Latest unambiguous class-button state."""

    class_name: Optional[str]
    ambiguous: bool = False


def select_signal_class(
    buttons: Sequence[int],
    button_by_class: Mapping[str, int],
) -> SignalSelection:
    """Select exactly one held class and reject simultaneous labels."""
    _validate_signal_button_mapping(button_by_class)
    required_button = max(button_by_class.values())
    if len(buttons) <= required_button:
        raise InvalidJoyInput(
            f'expected button index {required_button}, received '
            f'{len(buttons)} buttons'
        )
    selected = [
        class_name
        for class_name in SIGNAL_CLASSES
        if bool(buttons[button_by_class[class_name]])
    ]
    if len(selected) > 1:
        return SignalSelection(class_name=None, ambiguous=True)
    if selected:
        return SignalSelection(class_name=selected[0])
    return SignalSelection(class_name=None)


def _validate_signal_button_mapping(
    button_by_class: Mapping[str, int],
) -> None:
    if set(button_by_class) != set(SIGNAL_CLASSES):
        raise ValueError('button mapping must define every signal class')
    indices = tuple(button_by_class.values())
    if any(index < 0 for index in indices):
        raise ValueError('signal button indices must be non-negative')
    if len(set(indices)) != len(indices):
        raise ValueError('signal button indices must be distinct')


class TrafficSignalCollectorNode(Node):
    """Publish safe manual commands and save held-button image classes."""

    def __init__(
        self,
        parameter_overrides: Optional[Sequence[Parameter]] = None,
    ) -> None:
        super().__init__(
            'traffic_signal_collector',
            parameter_overrides=parameter_overrides,
        )

        self.declare_parameter(
            'joy_topic',
            '/traffic_signal_collector/joy',
        )
        self.declare_parameter('motor_topic', '/xycar_motor')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter(
            'preview_topic',
            '/traffic_signal_collector/preview',
        )
        self.declare_parameter('collection_profile_path', '')
        self.declare_parameter('steering_contract', '')
        self.declare_parameter('steering_axis', 0)
        self.declare_parameter('lt_axis', 4)
        self.declare_parameter('rt_axis', 5)
        self.declare_parameter('trigger_axis_mode', 'negative')
        self.declare_parameter('invert_steering', True)
        self.declare_parameter('max_angle', 100.0)
        self.declare_parameter('max_reverse_speed', 5.0)
        self.declare_parameter('max_forward_speed', 5.0)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('joy_timeout_sec', 0.25)
        self.declare_parameter('camera_timeout_sec', 0.50)
        self.declare_parameter('graph_check_period_sec', 0.5)
        self.declare_parameter('neutral_trigger_threshold', 0.05)
        self.declare_parameter('stop_publish_count', 5)
        self.declare_parameter(
            'allowed_motor_relay_nodes',
            ['/ros_bridge'],
        )
        self.declare_parameter('red_button', 2)
        self.declare_parameter('yellow_button', 3)
        self.declare_parameter('straight_green_button', 1)
        self.declare_parameter('left_green_button', 0)
        self.declare_parameter(
            'recording_root_dir',
            '/home/xytron/xycar_data/traffic_signal_images',
        )
        self.declare_parameter('recording_jpeg_quality', 95)
        self.declare_parameter('recording_queue_size', 128)
        self.declare_parameter('recording_min_free_space_mb', 1024)
        self.declare_parameter('preview_enabled', False)

        self.joy_topic = str(self.get_parameter('joy_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.camera_topic = str(self.get_parameter('camera_topic').value)
        self.preview_topic = str(self.get_parameter('preview_topic').value)
        self.collection_profile_path = str(
            self.get_parameter('collection_profile_path').value
        )
        self.steering_contract = str(
            self.get_parameter('steering_contract').value
        ).strip()
        self.config = GamepadConfig(
            steering_axis=int(self.get_parameter('steering_axis').value),
            lt_axis=int(self.get_parameter('lt_axis').value),
            rt_axis=int(self.get_parameter('rt_axis').value),
            trigger_axis_mode=str(
                self.get_parameter('trigger_axis_mode').value
            ),
            invert_steering=bool(
                self.get_parameter('invert_steering').value
            ),
            max_angle=float(self.get_parameter('max_angle').value),
            max_reverse_speed=float(
                self.get_parameter('max_reverse_speed').value
            ),
            max_forward_speed=float(
                self.get_parameter('max_forward_speed').value
            ),
        )
        self.publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        self.joy_timeout_sec = float(
            self.get_parameter('joy_timeout_sec').value
        )
        self.camera_timeout_sec = float(
            self.get_parameter('camera_timeout_sec').value
        )
        self.graph_check_period_sec = float(
            self.get_parameter('graph_check_period_sec').value
        )
        self.neutral_trigger_threshold = float(
            self.get_parameter('neutral_trigger_threshold').value
        )
        self.stop_publish_count = int(
            self.get_parameter('stop_publish_count').value
        )
        self.allowed_motor_relay_nodes = tuple(
            str(value)
            for value in self.get_parameter(
                'allowed_motor_relay_nodes'
            ).value
        )
        self.button_by_class = {
            'red': int(self.get_parameter('red_button').value),
            'yellow': int(self.get_parameter('yellow_button').value),
            'straight_green': int(
                self.get_parameter('straight_green_button').value
            ),
            'left_green': int(
                self.get_parameter('left_green_button').value
            ),
        }
        self.recording_root_dir = str(
            self.get_parameter('recording_root_dir').value
        )
        self.recording_jpeg_quality = int(
            self.get_parameter('recording_jpeg_quality').value
        )
        self.recording_queue_size = int(
            self.get_parameter('recording_queue_size').value
        )
        self.recording_min_free_space_mb = int(
            self.get_parameter('recording_min_free_space_mb').value
        )
        self.preview_enabled = bool(
            self.get_parameter('preview_enabled').value
        )
        self._validate_parameters()

        self.bridge = CvBridge()
        self._command = STOP_COMMAND
        self._last_published_command = STOP_COMMAND
        self._last_joy_monotonic: Optional[float] = None
        self._last_camera_monotonic: Optional[float] = None
        self._input_valid = False
        self._selection = SignalSelection(class_name=None)
        self._arming_gate = NeutralArmingGate(
            self.neutral_trigger_threshold
        )
        self._competitors: tuple[str, ...] = ()
        self._has_motor_subscriber = False
        self._next_graph_check_monotonic = 0.0
        self._stop_reason: Optional[str] = None
        self._capture_sequence = 0
        self._submitted_counts = {name: 0 for name in SIGNAL_CLASSES}
        self._capture_disabled = False
        self._writer_failure_handled = False
        self._shutdown_started = False
        self.writer = AsyncClassImageWriter(
            self.recording_root_dir,
            class_names=SIGNAL_CLASSES,
            jpeg_quality=self.recording_jpeg_quality,
            queue_size=self.recording_queue_size,
            min_free_space_mb=self.recording_min_free_space_mb,
        )

        self.motor_publisher = self.create_publisher(
            Float32MultiArray,
            self.motor_topic,
            10,
        )
        self.preview_publisher = None
        if self.preview_enabled:
            self.preview_publisher = self.create_publisher(
                Image,
                self.preview_topic,
                qos_profile_sensor_data,
            )
        self.joy_subscription = self.create_subscription(
            Joy,
            self.joy_topic,
            self._on_joy,
            10,
        )
        self.camera_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self._on_camera,
            qos_profile_sensor_data,
        )
        self.control_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._on_control_timer,
        )

        self._refresh_graph(time.monotonic())
        self.publish_stop_burst()
        self.get_logger().warning(
            'Traffic-signal collector started disarmed. Release LT and RT '
            'once to arm; hold exactly one ABXY class button to save every '
            'camera frame.'
        )
        self.get_logger().info(
            f'joy={self.joy_topic}, camera={self.camera_topic}, '
            f'motor={self.motor_topic}, '
            f'dataset_root={self.recording_root_dir}, '
            'X=red, Y=yellow, B=straight_green, A=left_green, '
            f'speed=RT*{self.config.max_forward_speed:g}'
            f'-LT*{self.config.max_reverse_speed:g}, '
            f'preview={self.preview_enabled}'
        )

    def _validate_parameters(self) -> None:
        for label, topic in (
            ('joy_topic', self.joy_topic),
            ('motor_topic', self.motor_topic),
            ('camera_topic', self.camera_topic),
        ):
            if not topic:
                raise ValueError(f'{label} must not be empty')
        if self.preview_enabled and not self.preview_topic:
            raise ValueError('preview_topic must not be empty when enabled')
        if not self.recording_root_dir:
            raise ValueError('recording_root_dir must not be empty')
        require_steering_contract_name(self.steering_contract)
        _validate_collection_profile(
            self.collection_profile_path,
            node_name='traffic_signal_collector',
        )
        _validate_config(self.config)
        _validate_runtime_parameters(
            self.publish_rate_hz,
            self.joy_timeout_sec,
            self.graph_check_period_sec,
            self.neutral_trigger_threshold,
            self.stop_publish_count,
        )
        if (
            not math.isfinite(self.camera_timeout_sec)
            or self.camera_timeout_sec <= 0.0
        ):
            raise ValueError('camera_timeout_sec must be finite and positive')
        _validate_signal_button_mapping(self.button_by_class)
        if not 1 <= self.recording_jpeg_quality <= 100:
            raise ValueError('recording_jpeg_quality must be in [1, 100]')
        if self.recording_queue_size < 1:
            raise ValueError('recording_queue_size must be positive')
        if self.recording_min_free_space_mb < 0:
            raise ValueError(
                'recording_min_free_space_mb must be non-negative'
            )
        for node in self.allowed_motor_relay_nodes:
            if not node.startswith('/') or node.endswith('/'):
                raise ValueError(
                    'allowed_motor_relay_nodes entries must be fully '
                    'qualified node names without a trailing slash'
                )

    def _on_joy(self, message: Joy) -> None:
        now = time.monotonic()
        try:
            mapped_input = map_joy_input(message.axes, self.config)
        except InvalidJoyInput as exc:
            self._selection = SignalSelection(class_name=None)
            self._input_valid = False
            self._last_joy_monotonic = None
            self._stop_and_disarm(f'invalid Joy input: {exc}')
            self.get_logger().warning(
                f'Ignoring invalid Joy message: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        try:
            selection = select_signal_class(
                message.buttons,
                self.button_by_class,
            )
        except InvalidJoyInput as exc:
            selection = SignalSelection(class_name=None)
            self.get_logger().warning(
                f'Class capture disabled for invalid Joy buttons: {exc}',
                throttle_duration_sec=2.0,
            )
        self._update_selection(selection)

        if self._capture_disabled:
            self._arming_gate.disarm()
            self._command = STOP_COMMAND
            self._last_joy_monotonic = now
            self._input_valid = True
            return

        if self._has_motor_subscriber and not self._competitors:
            was_armed = self._arming_gate.armed
            self._arming_gate.observe(
                mapped_input.lt_depth,
                mapped_input.rt_depth,
            )
            if self._arming_gate.armed and not was_armed:
                self.get_logger().info(
                    'Traffic-signal collector armed after neutral LT/RT input.'
                )
        self._command = mapped_input.command
        self._last_joy_monotonic = now
        self._input_valid = True

    def _update_selection(self, selection: SignalSelection) -> None:
        if selection == self._selection:
            return
        self._selection = selection
        if selection.ambiguous:
            self.get_logger().warning(
                'Multiple signal class buttons are held; images will not '
                'be saved until exactly one remains.'
            )
            return
        if selection.class_name is None:
            self.get_logger().info(
                f'Signal capture idle; saved_counts={self.writer.counts}'
            )
            return
        self.get_logger().info(
            f'Signal capture selected: {selection.class_name}'
        )

    def _on_camera(self, message: Image) -> None:
        received_monotonic = time.monotonic()
        received_wall_time_ns = time.time_ns()
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8',
            )
        except CvBridgeError as exc:
            self.get_logger().warning(
                f'Camera image conversion failed; frame skipped: {exc}',
                throttle_duration_sec=2.0,
            )
            return
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            self.get_logger().warning(
                'Camera image is not uint8 BGR; frame skipped.',
                throttle_duration_sec=2.0,
            )
            return
        self._last_camera_monotonic = received_monotonic

        class_name = self._capturable_class(received_monotonic)
        if class_name is not None:
            self._capture_sequence += 1
            sample = ClassImageSample(
                class_name=class_name,
                image=np.ascontiguousarray(image).copy(),
                sequence=self._capture_sequence,
                received_wall_time_ns=received_wall_time_ns,
            )
            if not self.writer.submit(sample):
                self._capture_failure(
                    'class image writer queue is full or unavailable'
                )
            else:
                self._submitted_counts[class_name] += 1
                submitted = self._submitted_counts[class_name]
                if submitted % 30 == 0:
                    self.get_logger().info(
                        f'Capturing {class_name}: submitted={submitted}, '
                        f'saved_counts={self.writer.counts}'
                    )

        if self.preview_publisher is not None:
            self._publish_preview(message, image, received_monotonic)

    def _capturable_class(self, now_monotonic: float) -> Optional[str]:
        if self._capture_disabled or self._selection.ambiguous:
            return None
        if not self._input_valid or not is_input_fresh(
            now_monotonic,
            self._last_joy_monotonic,
            self.joy_timeout_sec,
        ):
            return None
        return self._selection.class_name

    def _publish_preview(
        self,
        source: Image,
        image: np.ndarray,
        now_monotonic: float,
    ) -> None:
        preview = np.ascontiguousarray(image).copy()
        status, class_name = self._preview_status(now_monotonic)
        color = {
            'SAVING': (0, 255, 0),
            'AMBIGUOUS': (0, 165, 255),
            'ERROR': (0, 0, 255),
            'IDLE': (255, 255, 255),
        }[status]
        cv2.putText(
            preview,
            f'{status}: {class_name or "none"}',
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        counts = self.writer.counts
        count_text = '  '.join(
            f'{name}={counts[name]}' for name in SIGNAL_CLASSES
        )
        cv2.putText(
            preview,
            count_text,
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        try:
            preview_message = self.bridge.cv2_to_imgmsg(
                preview,
                encoding='bgr8',
            )
        except CvBridgeError as exc:
            self.get_logger().warning(
                f'Preview image conversion failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return
        preview_message.header = source.header
        self.preview_publisher.publish(preview_message)

    def _preview_status(
        self,
        now_monotonic: float,
    ) -> tuple[str, Optional[str]]:
        if self._capture_disabled:
            return 'ERROR', None
        if self._selection.ambiguous:
            return 'AMBIGUOUS', None
        class_name = self._capturable_class(now_monotonic)
        if class_name is not None:
            return 'SAVING', class_name
        return 'IDLE', None

    def _on_control_timer(self) -> None:
        now = time.monotonic()
        self._handle_writer_failure()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)

        if self._capture_disabled:
            self._stop_and_disarm('signal image capture is disabled')
            return
        if self._competitors:
            self._stop_and_disarm(
                'competing motor publisher(s): ' + ', '.join(self._competitors)
            )
            return
        if not self._has_motor_subscriber:
            self._stop_and_disarm('no motor subscriber')
            return
        if not self._input_valid or not is_input_fresh(
            now,
            self._last_joy_monotonic,
            self.joy_timeout_sec,
        ):
            self._stop_and_disarm('Joy input is missing or stale')
            return
        if not is_input_fresh(
            now,
            self._last_camera_monotonic,
            self.camera_timeout_sec,
        ):
            self._stop_and_disarm('camera input is missing or stale')
            return
        if not self._arming_gate.armed:
            self._publish_stop_for_reason(
                'waiting for neutral LT/RT input to arm'
            )
            return

        self._stop_reason = None
        self._publish(self._command)

    def _capture_failure(self, reason: str) -> None:
        if self._capture_disabled:
            return
        self._capture_disabled = True
        self.get_logger().error(f'Traffic-signal capture failure: {reason}')
        self._stop_and_disarm(reason)

    def _handle_writer_failure(self) -> None:
        failure = self.writer.failure
        if failure is None or self._writer_failure_handled:
            return
        self._writer_failure_handled = True
        self._capture_failure(failure)

    def _refresh_graph(self, now_monotonic: float) -> None:
        self._next_graph_check_monotonic = (
            now_monotonic + self.graph_check_period_sec
        )
        topic = self.resolve_topic_name(self.motor_topic)
        subscriptions = self.get_subscriptions_info_by_topic(topic)
        allow_unnamed_bridge = (
            '/ros_bridge' in self.allowed_motor_relay_nodes
        )
        competitors = []
        for publisher in self.get_publishers_info_by_topic(topic):
            if (
                publisher.node_name == self.get_name()
                and publisher.node_namespace == self.get_namespace()
            ):
                continue
            label = _node_label(
                publisher.node_namespace,
                publisher.node_name,
            )
            if label in self.allowed_motor_relay_nodes:
                continue
            if allow_unnamed_bridge and _is_paired_unnamed_relay(
                publisher,
                subscriptions,
            ):
                continue
            competitors.append(label)
        self._competitors = tuple(sorted(set(competitors)))
        self._has_motor_subscriber = bool(subscriptions)

    def _publish_stop_for_reason(self, reason: str) -> None:
        self._publish(STOP_COMMAND)
        if reason == self._stop_reason:
            return
        self._stop_reason = reason
        self.get_logger().warning(
            f'Traffic-signal collector stopped: {reason}'
        )

    def _stop_and_disarm(self, reason: str) -> None:
        self._arming_gate.disarm()
        self._command = STOP_COMMAND
        self._publish_stop_for_reason(reason)

    def _publish(self, command: DriveCommand) -> None:
        message = Float32MultiArray()
        message.data = [float(command.angle), float(command.speed)]
        self.motor_publisher.publish(message)
        self._last_published_command = command

    def publish_stop_burst(self) -> None:
        for _ in range(self.stop_publish_count):
            self._publish(STOP_COMMAND)

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.publish_stop_burst()
        if not self.writer.shutdown():
            self.get_logger().error(
                'Traffic-signal image writer did not stop cleanly.'
            )
        self.get_logger().info(
            f'Traffic-signal capture saved_counts={self.writer.counts}'
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[TrafficSignalCollectorNode] = None
    stop_requested = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        node = TrafficSignalCollectorNode()
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None and rclpy.ok():
            node.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
