"""Event-driven, fail-closed gateway from Xycar units to native VESC topics."""

from __future__ import annotations

from collections import deque
import math
import signal
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Float64
from vesc_msgs.msg import VescStateStamped
from xycar_msgs.msg import XycarMotor

from xycar_motor_native.control import (
    CommandFreshnessWatchdog,
    MotorCommand,
    NativeMotorContract,
    NativeMotorMapper,
)


def _latest_reliable_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
    )


class NativeMotorGateway(Node):
    """Forward each accepted command once; use a timer only for safety checks."""

    def __init__(self) -> None:
        super().__init__('native_motor_gateway')
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()
        self._mapper = NativeMotorMapper(self._motor_contract())
        self._command_watchdog = CommandFreshnessWatchdog(
            timeout_sec=self.command_timeout_sec,
            check_rate_hz=self.watchdog_rate_hz,
        )
        self._lock = threading.RLock()
        self._started_monotonic = time.monotonic()
        self._last_feedback_monotonic: float | None = None
        self._last_source_stamp: tuple[int, int] | None = None
        self._last_nonzero = False
        self._stopped = True
        self._stop_reason = 'startup'
        self._fault_latched = False
        self._fault_reason: str | None = None
        self._zero_rearm_requested = False
        # Stay fail-closed until ROS discovery has completed at least one
        # publisher-conflict check.
        self._graph_safe = False
        self._shutdown_started = False
        self._command_times: deque[float] = deque()
        self._executed_times: deque[float] = deque()
        self._out_of_order_count = 0
        self._duplicate_count = 0
        self._clamped_count = 0
        self._ramping_count = 0

        qos = _latest_reliable_qos()
        self._erpm_publisher = self.create_publisher(
            Float64,
            self.erpm_topic,
            qos,
        )
        self._servo_publisher = self.create_publisher(
            Float64,
            self.servo_topic,
            qos,
        )
        self._executed_publisher = self.create_publisher(
            XycarMotor,
            self.executed_topic,
            qos,
        )
        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._ready_publisher = self.create_publisher(
            Bool,
            self.ready_topic,
            ready_qos,
        )
        self._command_subscription = self.create_subscription(
            XycarMotor,
            self.command_topic,
            self._on_command,
            qos,
        )
        self._feedback_subscription = self.create_subscription(
            VescStateStamped,
            self.vesc_state_topic,
            self._on_feedback,
            qos,
        )
        self._watchdog_timer = self.create_timer(
            1.0 / self.watchdog_rate_hz,
            self._on_watchdog,
        )
        self._graph_timer = self.create_timer(
            self.graph_check_period_sec,
            self._on_graph_check,
        )
        self._metrics_timer = self.create_timer(
            self.metrics_period_sec,
            self._publish_metrics,
        )
        self.publish_stop_burst(reason='startup')
        self._publish_ready(False)
        self.get_logger().warning(
            'Native motor gateway started stopped. It accepts only the '
            'dedicated ROS 2 command topic and never uses a publish timer '
            'for normal commands.'
        )

    def _declare_parameters(self) -> None:
        defaults = NativeMotorContract()
        self.declare_parameter('command_topic', '/xycar_motor_command')
        self.declare_parameter('executed_topic', '/xycar_motor_executed')
        self.declare_parameter('ready_topic', '/xycar_motor_native/ready')
        self.declare_parameter(
            'erpm_topic',
            '/xycar_native/commands/motor/speed',
        )
        self.declare_parameter(
            'servo_topic',
            '/xycar_native/commands/servo/position',
        )
        self.declare_parameter(
            'vesc_state_topic',
            '/xycar_native/sensors/core',
        )
        self.declare_parameter('command_timeout_sec', 0.25)
        self.declare_parameter('watchdog_rate_hz', 30.0)
        self.declare_parameter('graph_check_period_sec', 0.25)
        self.declare_parameter('feedback_timeout_sec', 0.25)
        self.declare_parameter('feedback_startup_grace_sec', 1.0)
        self.declare_parameter('require_vesc_feedback', True)
        self.declare_parameter('metrics_period_sec', 2.0)
        self.declare_parameter('stop_publish_count', 5)
        for name, value in defaults.__dict__.items():
            self.declare_parameter(name, value)

    def _read_parameters(self) -> None:
        for name in (
            'command_topic',
            'executed_topic',
            'ready_topic',
            'erpm_topic',
            'servo_topic',
            'vesc_state_topic',
        ):
            setattr(self, name, str(self.get_parameter(name).value))
        for name in (
            'command_timeout_sec',
            'watchdog_rate_hz',
            'graph_check_period_sec',
            'feedback_timeout_sec',
            'feedback_startup_grace_sec',
            'metrics_period_sec',
        ):
            setattr(self, name, float(self.get_parameter(name).value))
        self.require_vesc_feedback = bool(
            self.get_parameter('require_vesc_feedback').value
        )
        self.stop_publish_count = int(
            self.get_parameter('stop_publish_count').value
        )

    def _motor_contract(self) -> NativeMotorContract:
        defaults = NativeMotorContract()
        return NativeMotorContract(
            **{
                name: float(self.get_parameter(name).value)
                for name in defaults.__dict__
            }
        )

    def _validate_parameters(self) -> None:
        topics = (
            self.command_topic,
            self.executed_topic,
            self.ready_topic,
            self.erpm_topic,
            self.servo_topic,
            self.vesc_state_topic,
        )
        if any(not topic.startswith('/') for topic in topics):
            raise ValueError('native motor topics must be absolute')
        if len(set(topics)) != len(topics):
            raise ValueError('native motor topics must be distinct')
        for name in (
            'command_timeout_sec',
            'watchdog_rate_hz',
            'graph_check_period_sec',
            'feedback_timeout_sec',
            'feedback_startup_grace_sec',
            'metrics_period_sec',
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.stop_publish_count < 1:
            raise ValueError('stop_publish_count must be positive')

    def _on_command(self, message: XycarMotor) -> None:
        now = time.monotonic()
        stamp = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        if stamp == (0, 0):
            self._trip('command source stamp is zero')
            return
        if not all(math.isfinite(value) for value in (message.angle, message.speed)):
            self._trip('command contains NaN or Inf', header=message.header)
            return
        zero_command = float(message.angle) == 0.0 and float(message.speed) == 0.0
        with self._lock:
            previous_stamp = self._last_source_stamp
            if previous_stamp is not None and stamp == previous_stamp:
                self._duplicate_count += 1
                self._trip_locked('duplicate command source stamp', header=message.header)
                return
            if previous_stamp is not None and stamp < previous_stamp:
                self._out_of_order_count += 1
                self._trip_locked('out-of-order command source stamp', header=message.header)
                return
            self._last_source_stamp = stamp
            self._command_watchdog.observe(now)
            self._command_times.append(now)
            unsafe = self._environment_unsafe_reason_locked(now)
            if unsafe is not None:
                self._trip_locked(
                    unsafe,
                    header=message.header,
                    zero_rearm_requested=zero_command,
                )
                return
            if self._fault_latched:
                if not zero_command:
                    self._trip_locked(
                        'native motor fault is latched; send [0,0] to re-arm',
                        header=message.header,
                    )
                    return
                if unsafe is None:
                    self._fault_latched = False
                    self._fault_reason = None
                    self._zero_rearm_requested = False
            try:
                setpoint = self._mapper.map(
                    MotorCommand(
                        angle=float(message.angle),
                        speed=float(message.speed),
                    ),
                    now=now,
                )
            except ValueError as exc:
                self._trip_locked(str(exc), header=message.header)
                return
            self._publish_setpoint_locked(setpoint, header=message.header)
            self._stopped = setpoint.command.speed == 0.0
            self._last_nonzero = not self._stopped
            self._stop_reason = None if self._last_nonzero else 'source stop'

    def _on_feedback(self, _message: VescStateStamped) -> None:
        with self._lock:
            self._last_feedback_monotonic = time.monotonic()

    def _on_watchdog(self) -> None:
        now = time.monotonic()
        with self._lock:
            if (
                self._fault_latched
                and self._zero_rearm_requested
                and self._environment_unsafe_reason_locked(now) is None
            ):
                self._fault_latched = False
                self._fault_reason = None
                self._zero_rearm_requested = False
            reason = self._unsafe_reason_locked(now)
            if reason is None:
                age = self._command_watchdog.stale_age(now)
                if age is not None:
                    reason = f'command stale for {age:.3f}s'
            if reason is not None and (not self._stopped or reason != self._stop_reason):
                self._trip_locked(reason)
            self._publish_ready(self._ready_locked(now))

    def _on_graph_check(self) -> None:
        command_publishers = self.get_publishers_info_by_topic(
            self.resolve_topic_name(self.command_topic)
        )
        erpm_publishers = self.get_publishers_info_by_topic(
            self.resolve_topic_name(self.erpm_topic)
        )
        external_erpm = [
            endpoint
            for endpoint in erpm_publishers
            if not (
                endpoint.node_name == self.get_name()
                and endpoint.node_namespace == self.get_namespace()
            )
        ]
        with self._lock:
            self._graph_safe = len(command_publishers) <= 1 and not external_erpm
            if not self._graph_safe:
                self._trip_locked(
                    'competing native command or ERPM publisher detected'
                )

    def _unsafe_reason_locked(self, now: float) -> str | None:
        environmental = self._environment_unsafe_reason_locked(now)
        if environmental is not None:
            return environmental
        if self._fault_latched:
            return self._fault_reason or 'native motor fault is latched'
        return None

    def _environment_unsafe_reason_locked(self, now: float) -> str | None:
        if not self._graph_safe:
            return 'native motor graph is unsafe'
        if not self.require_vesc_feedback:
            return None
        feedback = self._last_feedback_monotonic
        if feedback is None:
            if now - self._started_monotonic <= self.feedback_startup_grace_sec:
                return 'waiting for VESC feedback'
            return 'VESC feedback is missing'
        if now - feedback > self.feedback_timeout_sec:
            return 'VESC feedback is stale'
        return None

    def _ready_locked(self, now: float) -> bool:
        return self._unsafe_reason_locked(now) is None

    def _trip(self, reason: str, *, header=None) -> None:
        with self._lock:
            self._trip_locked(reason, header=header)

    def _trip_locked(
        self,
        reason: str,
        *,
        header=None,
        zero_rearm_requested: bool = False,
    ) -> None:
        changed = reason != self._stop_reason or not self._stopped
        self._fault_latched = True
        self._fault_reason = reason
        self._zero_rearm_requested = zero_rearm_requested
        setpoint = self._mapper.stop()
        self._publish_setpoint_locked(setpoint, header=header)
        self._stopped = True
        self._last_nonzero = False
        self._stop_reason = reason
        self._publish_ready(False)
        if changed:
            self.get_logger().warning(f'native motor stopped: {reason}')

    def _publish_setpoint_locked(self, setpoint, *, header=None) -> None:
        erpm = Float64()
        erpm.data = float(setpoint.erpm)
        servo = Float64()
        servo.data = float(setpoint.servo)
        executed = XycarMotor()
        if header is None:
            executed.header.stamp = self.get_clock().now().to_msg()
            executed.header.frame_id = 'native_motor_watchdog'
        else:
            executed.header.stamp.sec = int(header.stamp.sec)
            executed.header.stamp.nanosec = int(header.stamp.nanosec)
            executed.header.frame_id = str(header.frame_id)
        executed.angle = float(setpoint.command.angle)
        executed.speed = float(setpoint.command.speed)
        self._servo_publisher.publish(servo)
        self._erpm_publisher.publish(erpm)
        self._executed_publisher.publish(executed)
        now = time.monotonic()
        self._executed_times.append(now)
        self._clamped_count += int(setpoint.clamped)
        self._ramping_count += int(setpoint.ramping)

    def _publish_ready(self, ready: bool) -> None:
        message = Bool()
        message.data = bool(ready)
        self._ready_publisher.publish(message)

    def _publish_metrics(self) -> None:
        now = time.monotonic()
        window = max(1.0, self.metrics_period_sec)
        with self._lock:
            _trim_times(self._command_times, now - window)
            _trim_times(self._executed_times, now - window)
            command_hz = len(self._command_times) / window
            executed_hz = len(self._executed_times) / window
            ready = self._ready_locked(now)
            self.get_logger().info(
                'native_motor_metrics '
                f'command_hz={command_hz:.2f} executed_hz={executed_hz:.2f} '
                f'erpm_command_hz={executed_hz:.2f} '
                f'duplicate={self._duplicate_count} '
                f'out_of_order={self._out_of_order_count} '
                f'clamped={self._clamped_count} ramping={self._ramping_count} '
                f'control_fifo_backlog=0 ready={ready}'
            )

    def publish_stop_burst(self, *, reason: str) -> None:
        with self._lock:
            setpoint = self._mapper.stop()
            for _ in range(self.stop_publish_count):
                self._publish_setpoint_locked(setpoint)
            self._stopped = True
            self._last_nonzero = False
            self._stop_reason = reason

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.publish_stop_burst(reason='shutdown')
        self._publish_ready(False)


def _trim_times(values: deque[float], lower_bound: float) -> None:
    while values and values[0] < lower_bound:
        values.popleft()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: NativeMotorGateway | None = None
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
        node = NativeMotorGateway()
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
