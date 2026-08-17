# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Camera-clocked manual collection over the native ROS 2 motor gateway."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import signal
import time
from typing import Sequence

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool
from xycar_msgs.msg import XycarMotor

from xycar_data.gamepad_teleop import (
    DriveCommand,
    GamepadConfig,
    InvalidJoyInput,
    NeutralArmingGate,
    RecordingAction,
    RecordingGate,
    STOP_COMMAND,
    _collection_profile_metadata,
    is_input_fresh,
    map_joy_input,
)
from xycar_data.session_writer import AsyncSessionWriter, CameraSample


@dataclass(frozen=True)
class _Frame:
    image_bgr: np.ndarray
    sequence: int
    stamp_sec: int
    stamp_nanosec: int
    received_monotonic: float
    received_wall_time_ns: int


@dataclass(frozen=True)
class _PendingExecution:
    frame: _Frame
    desired: DriveCommand
    history: tuple[tuple[float, float], ...]
    sent_monotonic: float


@dataclass(frozen=True)
class _PendingFinish:
    token: int
    reason: str
    complete: bool
    discarded: int
    samples: tuple[CameraSample, ...]


def _latest_reliable_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
    )


class HistoryGamepadCollectorNode(Node):
    """Publish at most one manual command for each accepted camera frame."""

    def __init__(self) -> None:
        super().__init__('history_gamepad_collector')
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()
        self.bridge = CvBridge()
        self.config = GamepadConfig(
            steering_axis=self.steering_axis,
            lt_axis=self.lt_axis,
            rt_axis=self.rt_axis,
            trigger_axis_mode=self.trigger_axis_mode,
            invert_steering=self.invert_steering,
            max_angle=self.max_angle,
            max_reverse_speed=self.max_reverse_speed,
            max_forward_speed=self.max_forward_speed,
        )
        self.collection_profile_metadata = _collection_profile_metadata(
            self.collection_profile_path
        )
        self.writer = AsyncSessionWriter(
            self.recording_root_dir,
            png_compression=self.recording_png_compression,
            queue_size=self.recording_queue_size,
            min_free_space_mb=self.recording_min_free_space_mb,
            image_format=self.recording_image_format,
            jpeg_quality=self.recording_jpeg_quality,
        )
        self._arming_gate = NeutralArmingGate(self.neutral_trigger_threshold)
        self._recording_gate = RecordingGate()
        self._command = STOP_COMMAND
        self._last_joy_monotonic: float | None = None
        self._joy_valid = False
        self._native_ready = False
        self._has_motor_subscriber = False
        self._competitors: tuple[str, ...] = ()
        self._next_graph_check_monotonic = 0.0
        self._stop_reason: str | None = None
        self._pending: _PendingExecution | None = None
        self._last_execution_stamp: tuple[int, int] | None = None
        self._ignored_execution_stamps: set[tuple[int, int]] = set()
        self._latest_waiting_frame: _Frame | None = None
        self._history: deque[tuple[float, float]] = deque(
            [(0.0, 0.0)] * 4,
            maxlen=4,
        )
        self._camera_sequence = 0
        self._session_token: int | None = None
        self._finishing_token: int | None = None
        self._pending_finish: _PendingFinish | None = None
        self._recording_tail: deque[CameraSample] = deque()
        self._recording_disabled = False
        self._writer_failure_handled = False
        self._shutdown_started = False
        self._camera_times: deque[float] = deque()
        self._command_times: deque[float] = deque()
        self._executed_times: deque[float] = deque()
        self._execution_latencies_ms: deque[float] = deque(maxlen=2048)
        self._image_to_executed_ms: deque[float] = deque(maxlen=2048)
        self._skipped_frames = 0
        self._duplicate_executions = 0
        self._out_of_order_executions = 0

        qos = _latest_reliable_qos()
        self.motor_publisher = self.create_publisher(
            XycarMotor,
            self.motor_topic,
            qos,
        )
        self.executed_subscription = self.create_subscription(
            XycarMotor,
            self.executed_topic,
            self._on_executed,
            qos,
        )
        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_subscription = self.create_subscription(
            Bool,
            self.ready_topic,
            self._on_ready,
            ready_qos,
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
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.safety_timer = self.create_timer(
            1.0 / self.watchdog_rate_hz,
            self._on_safety_timer,
        )
        self.metrics_timer = self.create_timer(
            self.metrics_period_sec,
            self._report_metrics,
        )
        self._refresh_graph(time.monotonic())
        self.publish_stop_burst()
        self.get_logger().warning(
            'History manual collector started DRIVE OFF. Release LT/RT to '
            'arm; hold A with forward speed to record and press B to save.'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('motor_topic', '/xycar_motor_command')
        self.declare_parameter('executed_topic', '/xycar_motor_executed')
        self.declare_parameter('ready_topic', '/xycar_motor_native/ready')
        self.declare_parameter('collection_profile_path', '')
        self.declare_parameter('steering_axis', 0)
        self.declare_parameter('lt_axis', 4)
        self.declare_parameter('rt_axis', 5)
        self.declare_parameter('trigger_axis_mode', 'negative')
        self.declare_parameter('invert_steering', True)
        self.declare_parameter('max_angle', 100.0)
        self.declare_parameter('max_reverse_speed', 7.0)
        self.declare_parameter('max_forward_speed', 15.0)
        self.declare_parameter('joy_timeout_sec', 0.25)
        self.declare_parameter('execution_timeout_sec', 0.25)
        self.declare_parameter('watchdog_rate_hz', 30.0)
        self.declare_parameter('graph_check_period_sec', 0.25)
        self.declare_parameter('metrics_period_sec', 2.0)
        self.declare_parameter('neutral_trigger_threshold', 0.05)
        self.declare_parameter('stop_publish_count', 5)
        self.declare_parameter('record_start_button', 0)
        self.declare_parameter('record_stop_button', 1)
        self.declare_parameter(
            'recording_root_dir',
            '/home/xytron/xycar_data/history_manual',
        )
        self.declare_parameter('emergency_discard_frames', 15)
        self.declare_parameter('recording_image_format', 'jpeg')
        self.declare_parameter('recording_jpeg_quality', 95)
        self.declare_parameter('recording_png_compression', 3)
        self.declare_parameter('recording_queue_size', 128)
        self.declare_parameter('recording_min_free_space_mb', 1024)

    def _read_parameters(self) -> None:
        string_names = (
            'joy_topic',
            'camera_topic',
            'motor_topic',
            'executed_topic',
            'ready_topic',
            'collection_profile_path',
            'trigger_axis_mode',
            'recording_root_dir',
            'recording_image_format',
        )
        for name in string_names:
            setattr(self, name, str(self.get_parameter(name).value))
        int_names = (
            'steering_axis',
            'lt_axis',
            'rt_axis',
            'stop_publish_count',
            'record_start_button',
            'record_stop_button',
            'emergency_discard_frames',
            'recording_jpeg_quality',
            'recording_png_compression',
            'recording_queue_size',
            'recording_min_free_space_mb',
        )
        for name in int_names:
            setattr(self, name, int(self.get_parameter(name).value))
        float_names = (
            'max_angle',
            'max_reverse_speed',
            'max_forward_speed',
            'joy_timeout_sec',
            'execution_timeout_sec',
            'watchdog_rate_hz',
            'graph_check_period_sec',
            'metrics_period_sec',
            'neutral_trigger_threshold',
        )
        for name in float_names:
            setattr(self, name, float(self.get_parameter(name).value))
        self.invert_steering = bool(
            self.get_parameter('invert_steering').value
        )
        self.recording_image_format = self.recording_image_format.lower()

    def _validate_parameters(self) -> None:
        topics = (
            self.joy_topic,
            self.camera_topic,
            self.motor_topic,
            self.executed_topic,
            self.ready_topic,
        )
        if any(not topic.startswith('/') for topic in topics):
            raise ValueError('history collection topics must be absolute')
        if self.collection_profile_path:
            from pathlib import Path

            if not Path(self.collection_profile_path).is_absolute() or not Path(
                self.collection_profile_path
            ).is_file():
                raise ValueError('collection_profile_path must be an existing absolute file')
        for name in (
            'max_angle',
            'max_reverse_speed',
            'max_forward_speed',
            'joy_timeout_sec',
            'execution_timeout_sec',
            'watchdog_rate_hz',
            'graph_check_period_sec',
            'metrics_period_sec',
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0.0 <= self.neutral_trigger_threshold < 1.0:
            raise ValueError('neutral trigger threshold must be in [0,1)')
        if self.record_start_button == self.record_stop_button:
            raise ValueError('record buttons must be distinct')
        if self.recording_image_format not in {'jpeg', 'png'}:
            raise ValueError('recording image format must be jpeg or png')
        if not 1 <= self.recording_jpeg_quality <= 100:
            raise ValueError('recording JPEG quality must be in [1,100]')
        if not 0 <= self.recording_png_compression <= 9:
            raise ValueError('recording PNG compression must be in [0,9]')
        if self.recording_queue_size < 1 or self.stop_publish_count < 1:
            raise ValueError('queue and stop publish counts must be positive')

    def _on_ready(self, message: Bool) -> None:
        self._native_ready = bool(message.data)
        if not self._native_ready:
            self._stop_and_disarm('native motor gateway is not ready')

    def _on_joy(self, message: Joy) -> None:
        now = time.monotonic()
        try:
            mapped = map_joy_input(message.axes, self.config)
        except InvalidJoyInput as exc:
            self._joy_valid = False
            self._last_joy_monotonic = None
            self._stop_and_disarm(f'invalid Joy input: {exc}')
            return
        if self._native_ready and self._has_motor_subscriber and not self._competitors:
            self._arming_gate.observe(mapped.lt_depth, mapped.rt_depth)
        self._command = mapped.command
        self._joy_valid = True
        self._last_joy_monotonic = now
        required = max(self.record_start_button, self.record_stop_button)
        if len(message.buttons) <= required:
            return
        action = self._recording_gate.observe_buttons(
            a_pressed=bool(message.buttons[self.record_start_button]),
            b_pressed=bool(message.buttons[self.record_stop_button]),
        )
        if action == RecordingAction.FINISH_NORMAL:
            self._finish_session('b_button', discard_tail=False, complete=True)

    def _on_camera(self, message: Image) -> None:
        now = time.monotonic()
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self._stop_and_disarm(f'camera conversion failed: {exc}')
            return
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            self._stop_and_disarm('camera image is not uint8')
            return
        stamp = message.header.stamp
        if int(stamp.sec) == 0 and int(stamp.nanosec) == 0:
            self._stop_and_disarm('camera stamp is zero')
            return
        self._camera_sequence += 1
        frame = _Frame(
            image_bgr=np.ascontiguousarray(image).copy(),
            sequence=self._camera_sequence,
            stamp_sec=int(stamp.sec),
            stamp_nanosec=int(stamp.nanosec),
            received_monotonic=now,
            received_wall_time_ns=time.time_ns(),
        )
        self._camera_times.append(now)
        if self._pending is not None:
            if self._latest_waiting_frame is not None:
                self._skipped_frames += 1
            self._latest_waiting_frame = frame
            return
        self._send_frame(frame)

    def _send_frame(self, frame: _Frame) -> None:
        now = time.monotonic()
        reason = self._unsafe_reason(now)
        desired = self._command if reason is None else STOP_COMMAND
        if reason is not None:
            self._stop_and_disarm(reason, publish=False)
        message = XycarMotor()
        message.header.stamp.sec = frame.stamp_sec
        message.header.stamp.nanosec = frame.stamp_nanosec
        message.header.frame_id = 'history_manual_camera'
        message.angle = float(desired.angle)
        message.speed = float(desired.speed)
        self._pending = _PendingExecution(
            frame=frame,
            desired=desired,
            history=tuple(self._history),
            sent_monotonic=now,
        )
        self.motor_publisher.publish(message)
        self._command_times.append(now)

    def _on_executed(self, message: XycarMotor) -> None:
        stamp = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        if stamp in self._ignored_execution_stamps:
            self._ignored_execution_stamps.remove(stamp)
            return
        pending = self._pending
        if pending is None:
            if message.header.frame_id == 'native_motor_watchdog':
                self._history = deque([(0.0, 0.0)] * 4, maxlen=4)
            elif stamp == self._last_execution_stamp:
                self._duplicate_executions += 1
                self._recording_failure('duplicate motor execution echo')
            else:
                self._out_of_order_executions += 1
                self._recording_failure('unexpected motor execution echo')
            return
        expected = (pending.frame.stamp_sec, pending.frame.stamp_nanosec)
        if stamp != expected:
            if stamp == self._last_execution_stamp:
                self._duplicate_executions += 1
            else:
                self._out_of_order_executions += 1
            self._pending = None
            self._recording_failure(
                f'motor execution stamp mismatch: {stamp} != {expected}'
            )
            return
        now = time.monotonic()
        actual = DriveCommand(angle=float(message.angle), speed=float(message.speed))
        if not all(math.isfinite(value) for value in (actual.angle, actual.speed)):
            self._pending = None
            self._recording_failure('motor execution contains NaN or Inf')
            return
        self._executed_times.append(now)
        self._execution_latencies_ms.append((now - pending.sent_monotonic) * 1000.0)
        self._image_to_executed_ms.append(
            (now - pending.frame.received_monotonic) * 1000.0
        )
        self._last_execution_stamp = stamp
        if actual.speed == 0.0:
            self._history = deque([(0.0, 0.0)] * 4, maxlen=4)
        else:
            self._history.append((actual.angle, actual.speed))
        self._stop_reason = None if actual.speed != 0.0 else self._stop_reason
        action = self._recording_gate.observe_published_speed(actual.speed)
        if action == RecordingAction.START_RECORDING:
            self._start_session()
        if self._session_token is not None and actual.speed > 0.0:
            self._record_sample(pending, actual)
        if action == RecordingAction.FINISH_EMERGENCY:
            self._finish_session(
                'speed_nonpositive',
                discard_tail=True,
                complete=True,
            )
        self._pending = None
        waiting = self._latest_waiting_frame
        self._latest_waiting_frame = None
        if waiting is not None:
            self._send_frame(waiting)

    def _unsafe_reason(self, now: float) -> str | None:
        if self._competitors:
            return 'competing native motor publisher(s)'
        if not self._has_motor_subscriber:
            return 'no native motor subscriber'
        if not self._native_ready:
            return 'native motor gateway is not ready'
        if not self._joy_valid or not is_input_fresh(
            now,
            self._last_joy_monotonic,
            self.joy_timeout_sec,
        ):
            return 'Joy input is missing or stale'
        if not self._arming_gate.armed:
            return 'waiting for neutral LT/RT input to arm'
        return None

    def _on_safety_timer(self) -> None:
        now = time.monotonic()
        self._handle_writer_failure()
        self._poll_writer_results()
        self._retry_finish()
        self._flush_recording_prefix()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)
        if (
            self._pending is not None
            and now - self._pending.sent_monotonic
            > self.execution_timeout_sec
        ):
            self._pending = None
            self._recording_failure('native motor execution echo timed out')
        reason = self._unsafe_reason(now)
        if reason is not None and reason != self._stop_reason:
            self._stop_and_disarm(reason)

    def _refresh_graph(self, now: float) -> None:
        subscriptions = self.get_subscriptions_info_by_topic(
            self.resolve_topic_name(self.motor_topic)
        )
        competitors = []
        for endpoint in self.get_publishers_info_by_topic(
            self.resolve_topic_name(self.motor_topic)
        ):
            if (
                endpoint.node_name == self.get_name()
                and endpoint.node_namespace == self.get_namespace()
            ):
                continue
            competitors.append(f'{endpoint.node_namespace}/{endpoint.node_name}')
        self._competitors = tuple(sorted(set(competitors)))
        self._has_motor_subscriber = bool(subscriptions)
        self._next_graph_check_monotonic = now + self.graph_check_period_sec

    def _stop_and_disarm(self, reason: str, *, publish: bool = True) -> None:
        self._arming_gate.disarm()
        self._command = STOP_COMMAND
        self._history = deque([(0.0, 0.0)] * 4, maxlen=4)
        changed = reason != self._stop_reason
        self._stop_reason = reason
        if publish and self._pending is None and changed:
            self._publish_stop()
        if changed:
            self.get_logger().warning(f'History manual stopped: {reason}')

    def _publish_stop(self, *, nanosecond_offset: int = 0) -> None:
        message = XycarMotor()
        stamp = self.get_clock().now().nanoseconds + nanosecond_offset
        message.header.stamp.sec = stamp // 1_000_000_000
        message.header.stamp.nanosec = stamp % 1_000_000_000
        message.header.frame_id = 'history_manual_stop'
        self._ignored_execution_stamps.add(
            (message.header.stamp.sec, message.header.stamp.nanosec)
        )
        self.motor_publisher.publish(message)

    def publish_stop_burst(self) -> None:
        for index in range(self.stop_publish_count):
            self._publish_stop(nanosecond_offset=index)

    def _start_session(self) -> None:
        if self._recording_disabled or self._session_token is not None:
            return
        metadata = {
            'dataset_kind': 'camera_first_teleop_behavior_cloning',
            'camera_is_primary': True,
            'lidar_is_optional': False,
            'control_mode': 'history_gamepad',
            'sample_clock': 'camera_frame',
            'motor_transport': 'ros2_native',
            'curriculum': {'generation': 0},
            'history': {
                'frames': 4,
                'time_order': 'oldest_to_newest',
                'initial_command': [0, 0],
                'update': 'externally_executed_commands',
            },
            'topics': {
                'camera_topic': self.camera_topic,
                'motor_command_topic': self.motor_topic,
                'motor_executed_topic': self.executed_topic,
                'joy_topic': self.joy_topic,
            },
            'gamepad': {
                'steering_axis': self.steering_axis,
                'lt_axis': self.lt_axis,
                'rt_axis': self.rt_axis,
                'trigger_axis_mode': self.trigger_axis_mode,
                'invert_steering': self.invert_steering,
                'max_angle': self.max_angle,
                'max_reverse_speed': self.max_reverse_speed,
                'max_forward_speed': self.max_forward_speed,
                'record_start_button': self.record_start_button,
                'record_stop_button': self.record_stop_button,
            },
            'collection_profile': dict(self.collection_profile_metadata),
            'runtime_safety': {
                'joy_timeout_sec': self.joy_timeout_sec,
                'execution_timeout_sec': self.execution_timeout_sec,
                'watchdog_rate_hz': self.watchdog_rate_hz,
            },
            'recording': {
                'root_dir': self.recording_root_dir,
                'image_format': self.recording_image_format,
                'jpeg_quality': self.recording_jpeg_quality,
                'queue_size': self.recording_queue_size,
                'emergency_discard_frames': self.emergency_discard_frames,
            },
        }
        token = self.writer.start_session(metadata)
        if token is None:
            self._recording_failure('could not start history manual session')
            return
        self._session_token = token
        self._recording_tail.clear()
        self.get_logger().info('History manual recording started.')

    def _record_sample(self, pending: _PendingExecution, actual: DriveCommand) -> None:
        frame = pending.frame
        sample = CameraSample(
            image=frame.image_bgr,
            camera_sequence=frame.sequence,
            camera_stamp_sec=frame.stamp_sec,
            camera_stamp_nanosec=frame.stamp_nanosec,
            camera_received_monotonic=frame.received_monotonic,
            camera_received_wall_time_ns=frame.received_wall_time_ns,
            angle=actual.angle,
            speed=actual.speed,
            input_key='history_gamepad',
            lidar=None,
            lidar_skew_sec=None,
            history_commands=pending.history,
            motor_executed_received_wall_time_ns=time.time_ns(),
        )
        self._recording_tail.append(sample)
        self._flush_recording_prefix()
        if len(self._recording_tail) > self.recording_queue_size + self.emergency_discard_frames:
            self._recording_failure('history manual recording backlog exceeded')

    def _flush_recording_prefix(self) -> None:
        if self._session_token is None:
            return
        while len(self._recording_tail) > self.emergency_discard_frames:
            if not self.writer.submit(self._session_token, self._recording_tail[0]):
                return
            self._recording_tail.popleft()

    def _finish_session(self, reason: str, *, discard_tail: bool, complete: bool) -> None:
        token = self._session_token
        if token is None:
            self._recording_gate.finish_completed()
            return
        buffered = tuple(self._recording_tail)
        self._recording_tail.clear()
        discarded = min(self.emergency_discard_frames, len(buffered)) if discard_tail else 0
        samples = buffered[:-discarded] if discarded else buffered
        self._session_token = None
        self._finishing_token = token
        self._pending_finish = _PendingFinish(token, reason, complete, discarded, samples)
        self._retry_finish()

    def _retry_finish(self) -> None:
        pending = self._pending_finish
        if pending is None or self.writer.failure is not None:
            return
        if self.writer.finish(
            pending.token,
            pending.reason,
            complete=pending.complete,
            extra_metadata={
                'emergency_discard_count': pending.discarded,
                'skipped_camera_frames': self._skipped_frames,
            },
            final_samples=pending.samples,
        ):
            self._pending_finish = None

    def _poll_writer_results(self) -> None:
        for result in self.writer.poll_results():
            if result.token == self._finishing_token:
                self._finishing_token = None
                self._recording_gate.finish_completed()
                self.get_logger().info(
                    f'History manual session completed={result.completed} '
                    f'path={result.path} samples={result.sample_count}'
                )

    def _recording_failure(self, reason: str) -> None:
        if not self._recording_disabled:
            self._recording_disabled = True
            self._finish_session(reason, discard_tail=False, complete=False)
        self._stop_and_disarm(reason)

    def _handle_writer_failure(self) -> None:
        if self.writer.failure is not None and not self._writer_failure_handled:
            self._writer_failure_handled = True
            self._recording_failure(self.writer.failure)

    def _report_metrics(self) -> None:
        now = time.monotonic()
        lower = now - self.metrics_period_sec
        for values in (self._camera_times, self._command_times, self._executed_times):
            while values and values[0] < lower:
                values.popleft()
        p95 = 0.0
        if self._execution_latencies_ms:
            p95 = float(np.percentile(self._execution_latencies_ms, 95))
        end_to_end_p95 = 0.0
        if self._image_to_executed_ms:
            end_to_end_p95 = float(np.percentile(self._image_to_executed_ms, 95))
        self._execution_latencies_ms.clear()
        self._image_to_executed_ms.clear()
        self.get_logger().info(
            'history_manual_metrics '
            f'camera_hz={len(self._camera_times) / self.metrics_period_sec:.2f} '
            f'command_hz={len(self._command_times) / self.metrics_period_sec:.2f} '
            f'executed_hz={len(self._executed_times) / self.metrics_period_sec:.2f} '
            f'execution_echo_p95_ms={p95:.2f} '
            f'image_to_executed_p95_ms={end_to_end_p95:.2f} '
            f'skipped_frames={self._skipped_frames} '
            f'duplicate={self._duplicate_executions} '
            f'out_of_order={self._out_of_order_executions} '
            'control_fifo_backlog=0'
        )

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._finish_session('shutdown', discard_tail=False, complete=False)
        self.publish_stop_burst()
        deadline = time.monotonic() + 1.0
        while self._pending_finish is not None and time.monotonic() < deadline:
            self._retry_finish()
            time.sleep(0.01)
        self.writer.shutdown()
        self._poll_writer_results()


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: HistoryGamepadCollectorNode | None = None
    stop_requested = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        node = HistoryGamepadCollectorNode()
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.02)
    finally:
        if node is not None and rclpy.ok():
            node.shutdown()
        if node is not None:
            node.destroy_node()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
