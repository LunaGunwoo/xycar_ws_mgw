# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Camera-clocked four-command-history policy over native ROS 2 motor I/O."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import signal
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool, Float32MultiArray
from xycar_msgs.msg import XycarMotor

from xycar_ai_drive.artifact import PolicyArtifact
from xycar_ai_drive.control import (
    STOP_COMMAND,
    DriveCommand,
    ToggleAction,
    ToggleDriveGate,
    command_class_ids,
    is_fresh,
)
from xycar_ai_drive.policy_ipc import UnixSocketPolicyClient
from xycar_ai_drive.policy_runtime import TorchScriptPolicy


PolicyFactory = Callable[..., object]


@dataclass(frozen=True)
class _Frame:
    sequence: int
    image_rgb: np.ndarray
    stamp_sec: int
    stamp_nanosec: int
    received_monotonic: float
    received_wall_time_ns: int


@dataclass(frozen=True)
class _PendingExecution:
    frame: _Frame
    requested: DriveCommand
    sent_monotonic: float


def _latest_reliable_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
    )


class HistoryPolicyNode(Node):
    """Run one inference at a time and wait for its actual execution echo."""

    def __init__(
        self,
        parameter_overrides: Sequence[Parameter] | None = None,
        *,
        policy_factory: PolicyFactory = TorchScriptPolicy,
    ) -> None:
        super().__init__(
            'history_policy',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()
        self.bridge = CvBridge()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._toggle = ToggleDriveGate()
        self._latest_frame: _Frame | None = None
        self._frame_sequence = 0
        self._processed_sequence = 0
        self._pending: _PendingExecution | None = None
        self._last_execution_stamp: tuple[int, int] | None = None
        self._ignored_execution_stamps: set[tuple[int, int]] = set()
        self._last_joy_monotonic: float | None = None
        self._joy_valid = False
        self._native_ready = False
        self._has_motor_subscriber = False
        self._competitors: tuple[str, ...] = ()
        self._next_graph_check_monotonic = 0.0
        self._last_inference_source_monotonic: float | None = None
        self._stop_reason: str | None = None
        self._worker_stop = False
        self._shutdown_started = False
        self._camera_times: deque[float] = deque()
        self._inference_times: deque[float] = deque()
        self._command_times: deque[float] = deque()
        self._executed_times: deque[float] = deque()
        self._source_latencies_ms: deque[float] = deque(maxlen=4096)
        self._inference_latencies_ms: deque[float] = deque(maxlen=4096)
        self._image_to_command_ms: deque[float] = deque(maxlen=4096)
        self._execution_echo_ms: deque[float] = deque(maxlen=4096)
        self._skipped_frames = 0
        self._duplicate_executions = 0
        self._out_of_order_executions = 0

        self._policy = self._create_policy(policy_factory)
        self.artifact: PolicyArtifact | None = getattr(
            self._policy,
            'artifact',
            getattr(self._policy, '_artifact', None),
        )
        self._validate_artifact()
        history = self.artifact.history
        assert history is not None
        self._history: deque[tuple[int, int]] = deque(
            [history.initial_class_ids] * history.frames,
            maxlen=history.frames,
        )

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
        self.prediction_publisher = self.create_publisher(
            Float32MultiArray,
            self.prediction_topic,
            1,
        )
        self.enabled_publisher = self.create_publisher(
            Bool,
            self.enabled_topic,
            1,
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
        self._worker = threading.Thread(
            target=self._inference_worker,
            name='history-policy-inference',
            daemon=True,
        )
        self._worker.start()
        self._refresh_graph(time.monotonic())
        self.publish_stop_burst()
        self._publish_enabled(False)
        self.get_logger().warning(
            'History policy started DRIVE OFF. Release A, then press A to '
            'enable only after native motor readiness and fresh inference.'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('artifact_dir', '')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('motor_topic', '/xycar_motor_command')
        self.declare_parameter('executed_topic', '/xycar_motor_executed')
        self.declare_parameter('ready_topic', '/xycar_motor_native/ready')
        self.declare_parameter('prediction_topic', '/history_policy/prediction')
        self.declare_parameter('enabled_topic', '/history_policy/enabled')
        self.declare_parameter('a_button_index', 0)
        self.declare_parameter('allow_motion', True)
        self.declare_parameter('force_speed_zero', False)
        self.declare_parameter('require_schema4', True)
        self.declare_parameter('joy_timeout_sec', 0.25)
        self.declare_parameter('inference_timeout_sec', 0.25)
        self.declare_parameter('execution_timeout_sec', 0.25)
        self.declare_parameter('watchdog_rate_hz', 30.0)
        self.declare_parameter('graph_check_period_sec', 0.25)
        self.declare_parameter('metrics_period_sec', 2.0)
        self.declare_parameter('stop_publish_count', 5)
        self.declare_parameter('inference_backend', 'local')
        self.declare_parameter('inference_device', 'cpu')
        self.declare_parameter(
            'inference_socket_path',
            '/run/user/1000/xycar-ai/policy.sock',
        )
        self.declare_parameter('inference_rpc_timeout_sec', 0.20)
        self.declare_parameter('torch_num_threads', 8)
        self.declare_parameter('warmup_count', 3)

    def _read_parameters(self) -> None:
        for name in (
            'artifact_dir',
            'camera_topic',
            'joy_topic',
            'motor_topic',
            'executed_topic',
            'ready_topic',
            'prediction_topic',
            'enabled_topic',
            'inference_backend',
            'inference_device',
            'inference_socket_path',
        ):
            setattr(self, name, str(self.get_parameter(name).value))
        for name in (
            'joy_timeout_sec',
            'inference_timeout_sec',
            'execution_timeout_sec',
            'watchdog_rate_hz',
            'graph_check_period_sec',
            'metrics_period_sec',
            'inference_rpc_timeout_sec',
        ):
            setattr(self, name, float(self.get_parameter(name).value))
        for name in ('a_button_index', 'stop_publish_count', 'torch_num_threads', 'warmup_count'):
            setattr(self, name, int(self.get_parameter(name).value))
        self.allow_motion = bool(self.get_parameter('allow_motion').value)
        self.force_speed_zero = bool(self.get_parameter('force_speed_zero').value)
        self.require_schema4 = bool(self.get_parameter('require_schema4').value)

    def _validate_parameters(self) -> None:
        for name in (
            'artifact_dir',
            'camera_topic',
            'joy_topic',
            'motor_topic',
            'executed_topic',
            'ready_topic',
            'prediction_topic',
            'enabled_topic',
            'inference_socket_path',
        ):
            if not getattr(self, name):
                raise ValueError(f'{name} must not be empty')
        for topic in (
            self.camera_topic,
            self.joy_topic,
            self.motor_topic,
            self.executed_topic,
            self.ready_topic,
        ):
            if not topic.startswith('/'):
                raise ValueError('history policy I/O topics must be absolute')
        for name in (
            'joy_timeout_sec',
            'inference_timeout_sec',
            'execution_timeout_sec',
            'watchdog_rate_hz',
            'graph_check_period_sec',
            'metrics_period_sec',
            'inference_rpc_timeout_sec',
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.inference_rpc_timeout_sec > self.inference_timeout_sec:
            raise ValueError('RPC timeout must not exceed inference timeout')
        if self.a_button_index < 0 or self.stop_publish_count < 1:
            raise ValueError('button index and stop count are invalid')
        if self.torch_num_threads < 1 or self.warmup_count < 0:
            raise ValueError('Torch thread and warmup values are invalid')
        if self.inference_backend not in {'local', 'unix'}:
            raise ValueError('inference_backend must be local or unix')
        if self.inference_device not in {'cpu', 'cuda'}:
            raise ValueError('inference_device must be cpu or cuda')
        if self.inference_backend == 'local' and self.inference_device != 'cpu':
            raise ValueError('local inference requires cpu')

    def _create_policy(self, policy_factory: PolicyFactory):
        if self.inference_backend == 'unix':
            return UnixSocketPolicyClient(
                artifact_dir=self.artifact_dir,
                socket_path=self.inference_socket_path,
                timeout_sec=self.inference_rpc_timeout_sec,
                required_device=self.inference_device,
            )
        return policy_factory(
            artifact_dir=self.artifact_dir,
            torch_num_threads=self.torch_num_threads,
            warmup_count=self.warmup_count,
            history_reset_timeout_sec=self.inference_timeout_sec,
        )

    def _validate_artifact(self) -> None:
        if self.artifact is None or self.artifact.history is None:
            raise ValueError('history policy requires a four-frame AR artifact')
        history = self.artifact.history
        if history.frames != 4 or history.update != 'externally_executed_commands':
            raise ValueError('history artifact must use four externally executed commands')
        if self.require_schema4 and history.sample_clock != 'camera_frame':
            raise ValueError('history policy requires artifact schema v4')

    def _reset_history_locked(self) -> None:
        assert self.artifact is not None and self.artifact.history is not None
        history = self.artifact.history
        self._history = deque(
            [history.initial_class_ids] * history.frames,
            maxlen=history.frames,
        )
        self._policy.reset_history()

    def _on_ready(self, message: Bool) -> None:
        with self._lock:
            self._native_ready = bool(message.data)
        if not message.data:
            self._force_off('native motor gateway is not ready')

    def _on_joy(self, message: Joy) -> None:
        now = time.monotonic()
        if len(message.buttons) <= self.a_button_index:
            self._force_off('Joy button array is too short')
            return
        with self._lock:
            self._joy_valid = True
            self._last_joy_monotonic = now
            action = self._toggle.observe(
                pressed=bool(message.buttons[self.a_button_index]),
                can_enable=self._can_enable_locked(now),
            )
            if action == ToggleAction.ENABLED:
                self._reset_history_locked()
                self._stop_reason = None
            elif action == ToggleAction.DISABLED:
                self._reset_history_locked()
        if action == ToggleAction.ENABLED:
            self._publish_enabled(True)
            self.get_logger().warning('History policy motion toggled ON by A.')
        elif action == ToggleAction.DISABLED:
            self._force_off('A toggled DRIVE OFF')
        elif action == ToggleAction.REJECTED:
            self._force_off('A toggle rejected because prerequisites are not ready')

    def _on_camera(self, message: Image) -> None:
        now = time.monotonic()
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='rgb8')
        except CvBridgeError as exc:
            self._force_off(f'camera conversion failed: {exc}')
            return
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            self._force_off('camera image is not uint8 RGB')
            return
        stamp = message.header.stamp
        if int(stamp.sec) == 0 and int(stamp.nanosec) == 0:
            self._force_off('camera stamp is zero')
            return
        with self._condition:
            self._frame_sequence += 1
            if (
                self._latest_frame is not None
                and self._latest_frame.sequence > self._processed_sequence
            ):
                self._skipped_frames += 1
            self._latest_frame = _Frame(
                sequence=self._frame_sequence,
                image_rgb=np.ascontiguousarray(image).copy(),
                stamp_sec=int(stamp.sec),
                stamp_nanosec=int(stamp.nanosec),
                received_monotonic=now,
                received_wall_time_ns=time.time_ns(),
            )
            self._camera_times.append(now)
            self._condition.notify_all()

    def _inference_worker(self) -> None:
        while True:
            with self._condition:
                while not self._worker_stop and (
                    self._pending is not None
                    or self._latest_frame is None
                    or self._latest_frame.sequence <= self._processed_sequence
                ):
                    self._condition.wait()
                if self._worker_stop:
                    return
                frame = self._latest_frame
                assert frame is not None
                self._processed_sequence = frame.sequence
                history = tuple(self._history)
            try:
                result = self._policy.infer(frame.image_rgb, history)
            except Exception as exc:  # noqa: BLE001
                self._force_off(f'policy inference failed: {exc}')
                continue
            completed = time.monotonic()
            command = result.command
            if not all(
                math.isfinite(value)
                for value in (command.angle, command.speed, result.inference_ms)
            ):
                self._force_off('policy inference returned NaN or Inf')
                continue
            prediction = Float32MultiArray()
            prediction.data = [
                float(command.angle),
                float(command.speed),
                float(result.inference_ms),
            ]
            self.prediction_publisher.publish(prediction)
            with self._condition:
                self._last_inference_source_monotonic = frame.received_monotonic
                self._inference_times.append(completed)
                self._inference_latencies_ms.append(float(result.inference_ms))
                self._image_to_command_ms.append(
                    (completed - frame.received_monotonic) * 1000.0
                )
                reason = self._unsafe_reason_locked(completed)
                requested = command if reason is None and self._toggle.enabled else STOP_COMMAND
                if self.force_speed_zero:
                    requested = DriveCommand(angle=requested.angle, speed=0.0)
                self._send_frame_locked(frame, requested, now=completed)

    def _send_frame_locked(self, frame: _Frame, command: DriveCommand, *, now: float) -> None:
        message = XycarMotor()
        message.header.stamp.sec = frame.stamp_sec
        message.header.stamp.nanosec = frame.stamp_nanosec
        message.header.frame_id = 'history_policy_camera'
        message.angle = float(command.angle)
        message.speed = float(command.speed)
        self._pending = _PendingExecution(frame, command, now)
        self.motor_publisher.publish(message)
        self._command_times.append(now)

    def _on_executed(self, message: XycarMotor) -> None:
        stamp = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        with self._condition:
            if stamp in self._ignored_execution_stamps:
                self._ignored_execution_stamps.remove(stamp)
                self._condition.notify_all()
                return
            pending = self._pending
            if pending is None:
                if message.header.frame_id == 'native_motor_watchdog':
                    self._reset_history_locked()
                    self._toggle.fault()
                    self._publish_enabled(False)
                    return
                if stamp == self._last_execution_stamp:
                    self._duplicate_executions += 1
                    reason = 'duplicate motor execution echo'
                else:
                    self._out_of_order_executions += 1
                    reason = 'unexpected motor execution echo'
            expected = (
                (pending.frame.stamp_sec, pending.frame.stamp_nanosec)
                if pending is not None
                else None
            )
            if expected is not None and stamp != expected:
                self._out_of_order_executions += 1
                self._ignored_execution_stamps.add(expected)
                self._pending = None
                self._reset_history_locked()
                self._toggle.fault()
                self._condition.notify_all()
                reason = f'motor execution stamp mismatch: {stamp} != {expected}'
            elif pending is not None:
                actual = DriveCommand(
                    angle=float(message.angle),
                    speed=float(message.speed),
                )
                if not all(math.isfinite(value) for value in (actual.angle, actual.speed)):
                    reason = 'motor execution contains NaN or Inf'
                    self._pending = None
                    self._reset_history_locked()
                    self._toggle.fault()
                    self._condition.notify_all()
                else:
                    now = time.monotonic()
                    if self._toggle.enabled:
                        self._history.append(command_class_ids(actual))
                    else:
                        self._reset_history_locked()
                    self._executed_times.append(now)
                    self._source_latencies_ms.append(
                        (now - pending.frame.received_monotonic) * 1000.0
                    )
                    self._execution_echo_ms.append(
                        (now - pending.sent_monotonic) * 1000.0
                    )
                    self._last_execution_stamp = stamp
                    self._pending = None
                    self._condition.notify_all()
                    return
        self._force_off(reason)

    def _unsafe_reason_locked(self, now: float) -> str | None:
        if self._competitors:
            return 'competing native motor publisher(s)'
        if not self._has_motor_subscriber:
            return 'no native motor subscriber'
        if not self._native_ready:
            return 'native motor gateway is not ready'
        if not self._joy_valid or not is_fresh(
            now,
            self._last_joy_monotonic,
            self.joy_timeout_sec,
        ):
            return 'Joy input is missing or stale'
        if not is_fresh(
            now,
            self._last_inference_source_monotonic,
            self.inference_timeout_sec,
        ):
            return 'camera inference is missing or stale'
        return None

    def _can_enable_locked(self, now: float) -> bool:
        return self.allow_motion and self._unsafe_reason_locked(now) is None

    def _on_safety_timer(self) -> None:
        now = time.monotonic()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)
        with self._condition:
            if (
                self._pending is not None
                and now - self._pending.sent_monotonic
                > self.execution_timeout_sec
            ):
                stamp = (
                    self._pending.frame.stamp_sec,
                    self._pending.frame.stamp_nanosec,
                )
                self._ignored_execution_stamps.add(stamp)
                self._pending = None
                self._condition.notify_all()
                reason = 'native motor execution echo timed out'
            else:
                reason = self._unsafe_reason_locked(now) if self._toggle.enabled else None
        if reason is not None:
            self._force_off(reason)

    def _refresh_graph(self, now: float) -> None:
        topic = self.resolve_topic_name(self.motor_topic)
        subscriptions = self.get_subscriptions_info_by_topic(topic)
        competitors = []
        for endpoint in self.get_publishers_info_by_topic(topic):
            if (
                endpoint.node_name == self.get_name()
                and endpoint.node_namespace == self.get_namespace()
            ):
                continue
            competitors.append(f'{endpoint.node_namespace}/{endpoint.node_name}')
        with self._lock:
            self._competitors = tuple(sorted(set(competitors)))
            self._has_motor_subscriber = bool(subscriptions)
            self._next_graph_check_monotonic = now + self.graph_check_period_sec

    def _force_off(self, reason: str) -> None:
        with self._condition:
            was_enabled = self._toggle.fault()
            changed = reason != self._stop_reason
            self._stop_reason = reason
            if self._pending is not None:
                self._ignored_execution_stamps.add(
                    (
                        self._pending.frame.stamp_sec,
                        self._pending.frame.stamp_nanosec,
                    )
                )
                self._pending = None
            self._reset_history_locked()
            self._condition.notify_all()
        if changed or was_enabled:
            self._publish_stop()
        if was_enabled:
            self._publish_enabled(False)
        if changed:
            self.get_logger().warning(f'History policy stopped: {reason}')

    def _publish_stop(self, *, nanosecond_offset: int = 0) -> None:
        stamp = self.get_clock().now().nanoseconds + nanosecond_offset
        message = XycarMotor()
        message.header.stamp.sec = stamp // 1_000_000_000
        message.header.stamp.nanosec = stamp % 1_000_000_000
        message.header.frame_id = 'history_policy_stop'
        key = (message.header.stamp.sec, message.header.stamp.nanosec)
        with self._lock:
            self._ignored_execution_stamps.add(key)
        self.motor_publisher.publish(message)

    def publish_stop_burst(self) -> None:
        for index in range(self.stop_publish_count):
            self._publish_stop(nanosecond_offset=index)

    def _publish_enabled(self, enabled: bool) -> None:
        message = Bool()
        message.data = bool(enabled)
        self.enabled_publisher.publish(message)

    def _report_metrics(self) -> None:
        now = time.monotonic()
        lower = now - self.metrics_period_sec
        with self._lock:
            for values in (
                self._camera_times,
                self._inference_times,
                self._command_times,
                self._executed_times,
            ):
                while values and values[0] < lower:
                    values.popleft()
            inference_p95 = _p95(self._inference_latencies_ms)
            command_p95 = _p95(self._image_to_command_ms)
            execution_p95 = _p95(self._execution_echo_ms)
            end_to_end_p95 = _p95(self._source_latencies_ms)
            self._inference_latencies_ms.clear()
            self._image_to_command_ms.clear()
            self._execution_echo_ms.clear()
            self._source_latencies_ms.clear()
            rates = [
                len(values) / self.metrics_period_sec
                for values in (
                    self._camera_times,
                    self._inference_times,
                    self._command_times,
                    self._executed_times,
                )
            ]
        self.get_logger().info(
            'history_policy_metrics '
            f'camera_hz={rates[0]:.2f} inference_hz={rates[1]:.2f} '
            f'command_hz={rates[2]:.2f} executed_hz={rates[3]:.2f} '
            f'inference_p95_ms={inference_p95:.2f} '
            f'source_frame_age_p95_ms={command_p95:.2f} '
            f'execution_echo_p95_ms={execution_p95:.2f} '
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
        with self._condition:
            self._worker_stop = True
            self._toggle.fault()
            self._condition.notify_all()
        self.publish_stop_burst()
        self._publish_enabled(False)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        close_policy = getattr(self._policy, 'close', None)
        if close_policy is not None:
            close_policy()
        self.publish_stop_burst()


def _p95(values: deque[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: HistoryPolicyNode | None = None
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_shutdown(_signum, _frame) -> None:
        if node is not None:
            node.shutdown()
        context = rclpy.get_default_context()
        if context.ok():
            context.shutdown()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        node = HistoryPolicyNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
