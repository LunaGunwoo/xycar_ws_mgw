# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Convert a ROS Joy message into safe Xycar motor commands."""

from __future__ import annotations

from dataclasses import dataclass
import math
import signal
import time
from typing import Optional, Sequence

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray


@dataclass(frozen=True)
class GamepadConfig:
    """Axis mapping and output limits for the Remote Gamepad controller."""

    steering_axis: int = 0
    lt_axis: int = 4
    rt_axis: int = 5
    invert_steering: bool = True
    max_angle: float = 100.0
    max_reverse_speed: float = 5.0
    max_forward_speed: float = 7.0


@dataclass(frozen=True)
class DriveCommand:
    """Xycar motor command represented as steering angle and signed speed."""

    angle: float = 0.0
    speed: float = 0.0


STOP_COMMAND = DriveCommand()


class InvalidJoyInput(ValueError):
    """Raised when a Joy message cannot be converted safely."""


def map_joy_axes(
    axes: Sequence[float],
    config: GamepadConfig = GamepadConfig(),
) -> DriveCommand:
    """Map joystick axes to an Xycar command, clamping all input ranges."""
    required_axis = max(
        config.steering_axis,
        config.lt_axis,
        config.rt_axis,
    )
    if len(axes) <= required_axis:
        raise InvalidJoyInput(
            f'expected axis index {required_axis}, received {len(axes)} axes'
        )

    steering = _finite_axis(axes[config.steering_axis], 'steering')
    lt = _finite_axis(axes[config.lt_axis], 'LT')
    rt = _finite_axis(axes[config.rt_axis], 'RT')

    steering = _clamp(steering, -1.0, 1.0)
    lt = _clamp(lt, 0.0, 1.0)
    rt = _clamp(rt, 0.0, 1.0)

    steering_sign = -1.0 if config.invert_steering else 1.0
    return DriveCommand(
        angle=steering * config.max_angle * steering_sign,
        speed=(rt * config.max_forward_speed) - (
            lt * config.max_reverse_speed
        ),
    )


def is_input_fresh(
    now_monotonic: float,
    last_input_monotonic: Optional[float],
    timeout_sec: float,
) -> bool:
    """Return whether the latest valid Joy message is recent enough to use."""
    if last_input_monotonic is None:
        return False
    age = now_monotonic - last_input_monotonic
    return 0.0 <= age <= timeout_sec


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _finite_axis(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidJoyInput(f'{label} axis is not numeric') from exc
    if not math.isfinite(result):
        raise InvalidJoyInput(f'{label} axis is not finite')
    return result


def _validate_config(config: GamepadConfig) -> None:
    for label, index in (
        ('steering_axis', config.steering_axis),
        ('lt_axis', config.lt_axis),
        ('rt_axis', config.rt_axis),
    ):
        if index < 0:
            raise ValueError(f'{label} must be non-negative')
    if config.max_angle <= 0.0:
        raise ValueError('max_angle must be positive')
    if config.max_reverse_speed <= 0.0:
        raise ValueError('max_reverse_speed must be positive')
    if config.max_forward_speed <= 0.0:
        raise ValueError('max_forward_speed must be positive')


class GamepadTeleopNode(Node):
    """Publish Xycar commands from standard ROS game-controller input."""

    def __init__(self) -> None:
        super().__init__('gamepad_teleop')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('motor_topic', '/xycar_motor')
        self.declare_parameter('steering_axis', 0)
        self.declare_parameter('lt_axis', 4)
        self.declare_parameter('rt_axis', 5)
        self.declare_parameter('invert_steering', True)
        self.declare_parameter('max_angle', 100.0)
        self.declare_parameter('max_reverse_speed', 5.0)
        self.declare_parameter('max_forward_speed', 7.0)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('joy_timeout_sec', 0.25)
        self.declare_parameter('graph_check_period_sec', 0.5)
        self.declare_parameter('stop_publish_count', 5)

        self.joy_topic = str(self.get_parameter('joy_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.config = GamepadConfig(
            steering_axis=int(self.get_parameter('steering_axis').value),
            lt_axis=int(self.get_parameter('lt_axis').value),
            rt_axis=int(self.get_parameter('rt_axis').value),
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
        self.graph_check_period_sec = float(
            self.get_parameter('graph_check_period_sec').value
        )
        self.stop_publish_count = int(
            self.get_parameter('stop_publish_count').value
        )
        self._validate_parameters()

        self._command = STOP_COMMAND
        self._last_joy_monotonic: Optional[float] = None
        self._input_valid = False
        self._competitors: tuple[str, ...] = ()
        self._has_motor_subscriber = False
        self._next_graph_check_monotonic = 0.0
        self._stop_reason: Optional[str] = None

        self.motor_publisher = self.create_publisher(
            Float32MultiArray,
            self.motor_topic,
            10,
        )
        self.joy_subscription = self.create_subscription(
            Joy,
            self.joy_topic,
            self._on_joy,
            10,
        )
        self.control_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._on_control_timer,
        )
        self._refresh_graph(time.monotonic())
        self.publish_stop_burst()
        self.get_logger().warning(
            'Gamepad teleop started stopped. A/B are unused; trigger input '
            'becomes active as soon as valid /joy messages arrive.'
        )
        steering_sign = '-' if self.config.invert_steering else ''
        self.get_logger().info(
            f'joy={self.joy_topic}, motor={self.motor_topic}, '
            f'angle=axes[{self.config.steering_axis}]*'
            f'{steering_sign}'
            f'{self.config.max_angle:g}, '
            f'speed=axes[{self.config.rt_axis}]*'
            f'{self.config.max_forward_speed:g}-axes['
            f'{self.config.lt_axis}]*{self.config.max_reverse_speed:g}'
        )

    def _validate_parameters(self) -> None:
        if not self.joy_topic:
            raise ValueError('joy_topic must not be empty')
        if not self.motor_topic:
            raise ValueError('motor_topic must not be empty')
        _validate_config(self.config)
        if self.publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        if self.joy_timeout_sec <= 0.0:
            raise ValueError('joy_timeout_sec must be positive')
        if self.graph_check_period_sec <= 0.0:
            raise ValueError('graph_check_period_sec must be positive')
        if self.stop_publish_count < 1:
            raise ValueError('stop_publish_count must be at least 1')

    def _on_joy(self, message: Joy) -> None:
        now = time.monotonic()
        try:
            command = map_joy_axes(message.axes, self.config)
        except InvalidJoyInput as exc:
            self._command = STOP_COMMAND
            self._input_valid = False
            self._last_joy_monotonic = None
            self._publish_stop_for_reason(f'invalid Joy input: {exc}')
            self.get_logger().warning(
                f'Ignoring invalid Joy message: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        self._command = command
        self._last_joy_monotonic = now
        self._input_valid = True

    def _on_control_timer(self) -> None:
        now = time.monotonic()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)

        if self._competitors:
            self._publish_stop_for_reason(
                'competing motor publisher(s): ' + ', '.join(self._competitors)
            )
            return
        if not self._has_motor_subscriber:
            self._publish_stop_for_reason('no motor subscriber')
            return
        if not self._input_valid or not is_input_fresh(
            now,
            self._last_joy_monotonic,
            self.joy_timeout_sec,
        ):
            self._publish_stop_for_reason('Joy input is missing or stale')
            return

        self._stop_reason = None
        self._publish(self._command)

    def _refresh_graph(self, now_monotonic: float) -> None:
        self._next_graph_check_monotonic = (
            now_monotonic + self.graph_check_period_sec
        )
        topic = self.resolve_topic_name(self.motor_topic)
        competitors = []
        for publisher in self.get_publishers_info_by_topic(topic):
            if (
                publisher.node_name == self.get_name()
                and publisher.node_namespace == self.get_namespace()
            ):
                continue
            competitors.append(
                _node_label(publisher.node_namespace, publisher.node_name)
            )
        self._competitors = tuple(sorted(set(competitors)))
        self._has_motor_subscriber = bool(
            self.get_subscriptions_info_by_topic(topic)
        )

    def _publish_stop_for_reason(self, reason: str) -> None:
        self._publish(STOP_COMMAND)
        if reason == self._stop_reason:
            return
        self._stop_reason = reason
        self.get_logger().warning(f'Gamepad teleop stopped: {reason}')

    def _publish(self, command: DriveCommand) -> None:
        message = Float32MultiArray()
        message.data = [float(command.angle), float(command.speed)]
        self.motor_publisher.publish(message)

    def publish_stop_burst(self) -> None:
        for _ in range(self.stop_publish_count):
            self._publish(STOP_COMMAND)

    def shutdown(self) -> None:
        self.publish_stop_burst()


def _node_label(namespace: str, name: str) -> str:
    prefix = namespace.rstrip('/')
    label = f'{prefix}/{name}'
    return label if label.startswith('/') else f'/{label}'


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[GamepadTeleopNode] = None
    stop_requested = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        node = GamepadTeleopNode()
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
