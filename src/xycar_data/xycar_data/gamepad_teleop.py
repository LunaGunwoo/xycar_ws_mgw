# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Convert a ROS Joy message into safe Xycar motor commands."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
import signal
import time
from typing import Optional, Sequence

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Joy, LaserScan
from std_msgs.msg import Float32MultiArray

from xycar_data.session_writer import (
    AsyncSessionWriter,
    CameraSample,
    LidarSnapshot,
)
from xycar_data.steering_contract import (
    require_steering_contract_name,
    session_steering_contract,
)


_DDS_GUID_PREFIX_SIZE = 12
_UNKNOWN_NODE_NAME = '_NODE_NAME_UNKNOWN_'
_UNKNOWN_NODE_NAMESPACE = '_NODE_NAMESPACE_UNKNOWN_'


def _endpoint_participant_prefix(endpoint) -> bytes | None:
    """Return the DDS participant portion of a topic endpoint GID."""
    try:
        gid = bytes(endpoint.endpoint_gid)
    except (AttributeError, TypeError, ValueError):
        return None
    if len(gid) < _DDS_GUID_PREFIX_SIZE:
        return None
    return gid[:_DDS_GUID_PREFIX_SIZE]


def _is_unnamed_endpoint(endpoint) -> bool:
    return (
        endpoint.node_name == _UNKNOWN_NODE_NAME
        and endpoint.node_namespace == _UNKNOWN_NODE_NAMESPACE
    )


def _is_paired_unnamed_relay(endpoint, subscriptions) -> bool:
    """Match the unnamed publisher half of one DDS bridge participant."""
    if not _is_unnamed_endpoint(endpoint):
        return False
    participant = _endpoint_participant_prefix(endpoint)
    if participant is None:
        return False
    return any(
        _is_unnamed_endpoint(subscription)
        and _endpoint_participant_prefix(subscription) == participant
        for subscription in subscriptions
    )


@dataclass(frozen=True)
class GamepadConfig:
    """Axis mapping and output limits for the Remote Gamepad controller."""

    steering_axis: int = 0
    lt_axis: int = 4
    rt_axis: int = 5
    trigger_axis_mode: str = 'negative'
    invert_steering: bool = True
    max_angle: float = 100.0
    max_reverse_speed: float = 5.0
    max_forward_speed: float = 7.0
    fixed_reverse_button: int = 9
    fixed_reverse_speed: float = 5.0


@dataclass(frozen=True)
class DriveCommand:
    """Xycar motor command represented as steering angle and signed speed."""

    angle: float = 0.0
    speed: float = 0.0


STOP_COMMAND = DriveCommand()


@dataclass(frozen=True)
class MappedJoyInput:
    """Normalized controller input and the resulting Xycar command."""

    command: DriveCommand
    lt_depth: float
    rt_depth: float
    fixed_reverse_pressed: bool = False


@dataclass
class NeutralArmingGate:
    """Require neutral speed controls before drive commands may activate."""

    threshold: float
    armed: bool = False

    def observe(
        self,
        lt_depth: float,
        rt_depth: float,
        fixed_reverse_pressed: bool = False,
    ) -> bool:
        if (
            not self.armed
            and lt_depth <= self.threshold
            and rt_depth <= self.threshold
            and not fixed_reverse_pressed
        ):
            self.armed = True
        return self.armed

    def disarm(self) -> None:
        self.armed = False


class RecordingState(str, Enum):
    """Gamepad dataset recording lifecycle."""

    IDLE = 'idle'
    WAITING_FORWARD = 'waiting_forward'
    RECORDING = 'recording'
    FINISHING = 'finishing'


class RecordingAction(str, Enum):
    """Transition side effects requested by :class:`RecordingGate`."""

    NONE = 'none'
    WAITING_STARTED = 'waiting_started'
    WAITING_CANCELLED = 'waiting_cancelled'
    START_RECORDING = 'start_recording'
    FINISH_NORMAL = 'finish_normal'
    FINISH_EMERGENCY = 'finish_emergency'


@dataclass
class RecordingGate:
    """Turn A/B button levels and published speed into recording actions."""

    state: RecordingState = RecordingState.IDLE
    a_pressed: bool = False
    b_pressed: bool = False
    start_rearmed: bool = True

    def observe_buttons(
        self,
        *,
        a_pressed: bool,
        b_pressed: bool,
    ) -> RecordingAction:
        b_rising = b_pressed and not self.b_pressed
        self.a_pressed = a_pressed
        self.b_pressed = b_pressed

        if not a_pressed:
            self.start_rearmed = True
        if self.state == RecordingState.FINISHING:
            return RecordingAction.NONE

        if b_rising:
            if self.state == RecordingState.RECORDING:
                self.state = RecordingState.FINISHING
                return RecordingAction.FINISH_NORMAL
            if self.state == RecordingState.WAITING_FORWARD:
                self.state = RecordingState.IDLE
                return RecordingAction.WAITING_CANCELLED
            return RecordingAction.NONE

        if (
            self.state == RecordingState.IDLE
            and a_pressed
            and self.start_rearmed
        ):
            self.start_rearmed = False
            self.state = RecordingState.WAITING_FORWARD
            return RecordingAction.WAITING_STARTED
        if (
            self.state == RecordingState.WAITING_FORWARD
            and not a_pressed
        ):
            self.state = RecordingState.IDLE
            return RecordingAction.WAITING_CANCELLED
        return RecordingAction.NONE

    def observe_published_speed(self, speed: float) -> RecordingAction:
        if (
            self.state == RecordingState.WAITING_FORWARD
            and self.a_pressed
            and speed > 0.0
        ):
            self.state = RecordingState.RECORDING
            return RecordingAction.START_RECORDING
        if self.state == RecordingState.RECORDING and speed <= 0.0:
            self.state = RecordingState.FINISHING
            return RecordingAction.FINISH_EMERGENCY
        return RecordingAction.NONE

    def force_finishing(self) -> None:
        self.state = RecordingState.FINISHING

    def finish_completed(self) -> None:
        self.state = RecordingState.IDLE


@dataclass
class MissionSequenceRecordingGate:
    """Record complete mission sequences, including stationary frames."""

    state: RecordingState = RecordingState.IDLE
    a_pressed: bool = False
    b_pressed: bool = False
    start_rearmed: bool = True

    def observe_buttons(
        self,
        *,
        a_pressed: bool,
        b_pressed: bool,
    ) -> RecordingAction:
        a_rising = a_pressed and not self.a_pressed
        b_rising = b_pressed and not self.b_pressed
        self.a_pressed = a_pressed
        self.b_pressed = b_pressed
        if not a_pressed:
            self.start_rearmed = True
        if self.state == RecordingState.FINISHING:
            return RecordingAction.NONE
        if b_rising and self.state == RecordingState.RECORDING:
            self.state = RecordingState.FINISHING
            return RecordingAction.FINISH_NORMAL
        if (
            a_rising
            and self.start_rearmed
            and self.state == RecordingState.IDLE
        ):
            self.start_rearmed = False
            self.state = RecordingState.RECORDING
            return RecordingAction.START_RECORDING
        return RecordingAction.NONE

    def observe_published_speed(self, _speed: float) -> RecordingAction:
        return RecordingAction.NONE

    def force_finishing(self) -> None:
        self.state = RecordingState.FINISHING

    def finish_completed(self) -> None:
        self.state = RecordingState.IDLE


@dataclass(frozen=True)
class _PendingRecordingFinish:
    token: int
    reason: str
    complete: bool
    extra_metadata: dict[str, object]
    final_samples: tuple[CameraSample, ...]


class InvalidJoyInput(ValueError):
    """Raised when a Joy message cannot be converted safely."""


def map_joy_input(
    axes: Sequence[float],
    config: GamepadConfig = GamepadConfig(),
    buttons: Optional[Sequence[int]] = None,
) -> MappedJoyInput:
    """Normalize controller input and map it to an Xycar command."""
    _validate_config(config)
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
    lt_depth = _trigger_depth(
        axes[config.lt_axis],
        'LT',
        config.trigger_axis_mode,
    )
    rt_depth = _trigger_depth(
        axes[config.rt_axis],
        'RT',
        config.trigger_axis_mode,
    )
    fixed_reverse_pressed = bool(
        buttons is not None
        and len(buttons) > config.fixed_reverse_button
        and buttons[config.fixed_reverse_button]
    )

    steering = _clamp(steering, -1.0, 1.0)
    speed = (rt_depth * config.max_forward_speed) - (
        lt_depth * config.max_reverse_speed
    )
    if fixed_reverse_pressed:
        speed -= config.fixed_reverse_speed
    speed = _clamp(
        speed,
        -config.max_reverse_speed,
        config.max_forward_speed,
    )

    steering_sign = -1.0 if config.invert_steering else 1.0
    return MappedJoyInput(
        command=DriveCommand(
            angle=steering * config.max_angle * steering_sign,
            speed=speed,
        ),
        lt_depth=lt_depth,
        rt_depth=rt_depth,
        fixed_reverse_pressed=fixed_reverse_pressed,
    )


def map_joy_axes(
    axes: Sequence[float],
    config: GamepadConfig = GamepadConfig(),
    buttons: Optional[Sequence[int]] = None,
) -> DriveCommand:
    """Map controller input to a command, clamping all input ranges."""
    return map_joy_input(axes, config, buttons).command


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


def matching_lidar_snapshot(
    camera_received_monotonic: float,
    lidar: Optional[LidarSnapshot],
    *,
    lidar_timeout_sec: float,
    max_lidar_skew_sec: float,
) -> tuple[Optional[LidarSnapshot], Optional[float]]:
    """Return the newest preceding LiDAR scan when it is fresh enough."""
    if lidar is None:
        return None, None
    age = camera_received_monotonic - lidar.received_monotonic
    if age < 0.0 or age > lidar_timeout_sec:
        return None, None
    skew = abs(age)
    if skew > max_lidar_skew_sec:
        return None, None
    return lidar, skew


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


def _trigger_depth(value: float, label: str, mode: str) -> float:
    result = _finite_axis(value, label)
    if mode == 'signed':
        return _clamp((1.0 - result) / 2.0, 0.0, 1.0)
    if mode == 'positive':
        return _clamp(result, 0.0, 1.0)
    if mode == 'negative':
        return _clamp(-result, 0.0, 1.0)
    raise ValueError(
        "trigger_axis_mode must be 'signed', 'positive', or 'negative'"
    )


def _validate_finite_positive(label: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'{label} must be finite and positive')


def _validate_config(config: GamepadConfig) -> None:
    for label, index in (
        ('steering_axis', config.steering_axis),
        ('lt_axis', config.lt_axis),
        ('rt_axis', config.rt_axis),
        ('fixed_reverse_button', config.fixed_reverse_button),
    ):
        if index < 0:
            raise ValueError(f'{label} must be non-negative')
    if config.trigger_axis_mode not in ('signed', 'positive', 'negative'):
        raise ValueError(
            "trigger_axis_mode must be 'signed', 'positive', or 'negative'"
        )
    _validate_finite_positive('max_angle', config.max_angle)
    if config.max_angle > 100.0:
        raise ValueError('max_angle must be in (0,100]')
    _validate_finite_positive(
        'max_reverse_speed',
        config.max_reverse_speed,
    )
    _validate_finite_positive(
        'max_forward_speed',
        config.max_forward_speed,
    )
    _validate_finite_positive(
        'fixed_reverse_speed',
        config.fixed_reverse_speed,
    )
    if config.fixed_reverse_speed > config.max_reverse_speed:
        raise ValueError(
            'fixed_reverse_speed must not exceed max_reverse_speed'
        )


def _validate_runtime_parameters(
    publish_rate_hz: float,
    joy_timeout_sec: float,
    graph_check_period_sec: float,
    neutral_trigger_threshold: float,
    stop_publish_count: int,
) -> None:
    _validate_finite_positive('publish_rate_hz', publish_rate_hz)
    _validate_finite_positive('joy_timeout_sec', joy_timeout_sec)
    _validate_finite_positive(
        'graph_check_period_sec',
        graph_check_period_sec,
    )
    if (
        not math.isfinite(neutral_trigger_threshold)
        or not 0.0 <= neutral_trigger_threshold < 1.0
    ):
        raise ValueError(
            'neutral_trigger_threshold must be finite and in [0, 1)'
        )
    if stop_publish_count < 1:
        raise ValueError('stop_publish_count must be at least 1')


def _validate_lidar_parameters(
    lidar_topic: str,
    lidar_timeout_sec: float,
    max_lidar_skew_sec: float,
) -> None:
    if not lidar_topic:
        raise ValueError('lidar_topic must not be empty')
    _validate_finite_positive('lidar_timeout_sec', lidar_timeout_sec)
    _validate_finite_positive(
        'max_lidar_skew_sec',
        max_lidar_skew_sec,
    )
    if max_lidar_skew_sec > lidar_timeout_sec:
        raise ValueError(
            'max_lidar_skew_sec must not exceed lidar_timeout_sec'
        )


def _validate_recording_parameters(
    *,
    camera_topic: str,
    recording_root_dir: str,
    record_start_button: int,
    record_stop_button: int,
    emergency_discard_frames: int,
    recording_image_format: str,
    recording_jpeg_quality: int,
    recording_png_compression: int,
    recording_queue_size: int,
    recording_min_free_space_mb: int,
) -> None:
    if not camera_topic:
        raise ValueError('camera_topic must not be empty')
    if not recording_root_dir:
        raise ValueError('recording_root_dir must not be empty')
    if record_start_button < 0 or record_stop_button < 0:
        raise ValueError('record button indices must be non-negative')
    if record_start_button == record_stop_button:
        raise ValueError('record start and stop buttons must be different')
    if emergency_discard_frames < 0:
        raise ValueError('emergency_discard_frames must be non-negative')
    if recording_image_format not in {'jpeg', 'png'}:
        raise ValueError('recording_image_format must be jpeg or png')
    if not 1 <= recording_jpeg_quality <= 100:
        raise ValueError('recording_jpeg_quality must be in [1, 100]')
    if not 0 <= recording_png_compression <= 9:
        raise ValueError('recording_png_compression must be in [0, 9]')
    if recording_queue_size < 1:
        raise ValueError('recording_queue_size must be positive')
    if recording_min_free_space_mb < 0:
        raise ValueError(
            'recording_min_free_space_mb must be non-negative'
        )


class GamepadTeleopNode(Node):
    """Publish Xycar commands from standard ROS game-controller input."""

    def __init__(
        self,
        parameter_overrides: Optional[Sequence[Parameter]] = None,
    ) -> None:
        super().__init__(
            'gamepad_teleop',
            parameter_overrides=parameter_overrides,
        )

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('motor_topic', '/xycar_motor')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('lidar_timeout_sec', 0.30)
        self.declare_parameter('max_lidar_skew_sec', 0.20)
        self.declare_parameter('collection_profile_path', '')
        self.declare_parameter('steering_contract', '')
        self.declare_parameter('steering_axis', 0)
        self.declare_parameter('lt_axis', 4)
        self.declare_parameter('rt_axis', 5)
        self.declare_parameter('trigger_axis_mode', 'negative')
        self.declare_parameter('invert_steering', True)
        self.declare_parameter('max_angle', 100.0)
        self.declare_parameter('max_reverse_speed', 5.0)
        self.declare_parameter('max_forward_speed', 7.0)
        self.declare_parameter('fixed_reverse_button', 9)
        self.declare_parameter('fixed_reverse_speed', 5.0)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('joy_timeout_sec', 0.25)
        self.declare_parameter('graph_check_period_sec', 0.5)
        self.declare_parameter('neutral_trigger_threshold', 0.05)
        self.declare_parameter('stop_publish_count', 5)
        self.declare_parameter(
            'allowed_motor_relay_nodes',
            ['/ros_bridge'],
        )
        self.declare_parameter('record_start_button', 0)
        self.declare_parameter('record_stop_button', 1)
        self.declare_parameter('recording_mode', 'positive_speed_segment')
        self.declare_parameter('capture_kind', '')
        self.declare_parameter(
            'recording_root_dir',
            '/home/xytron/xycar_data/teleop',
        )
        self.declare_parameter('emergency_discard_frames', 15)
        self.declare_parameter('recording_image_format', 'jpeg')
        self.declare_parameter('recording_jpeg_quality', 95)
        self.declare_parameter('recording_png_compression', 3)
        self.declare_parameter('recording_queue_size', 128)
        self.declare_parameter('recording_min_free_space_mb', 1024)

        self.joy_topic = str(self.get_parameter('joy_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.camera_topic = str(self.get_parameter('camera_topic').value)
        self.lidar_topic = str(self.get_parameter('lidar_topic').value)
        self.lidar_timeout_sec = float(
            self.get_parameter('lidar_timeout_sec').value
        )
        self.max_lidar_skew_sec = float(
            self.get_parameter('max_lidar_skew_sec').value
        )
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
            fixed_reverse_button=int(
                self.get_parameter('fixed_reverse_button').value
            ),
            fixed_reverse_speed=float(
                self.get_parameter('fixed_reverse_speed').value
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
        self.record_start_button = int(
            self.get_parameter('record_start_button').value
        )
        self.record_stop_button = int(
            self.get_parameter('record_stop_button').value
        )
        self.recording_mode = str(
            self.get_parameter('recording_mode').value
        ).strip()
        self.capture_kind = str(
            self.get_parameter('capture_kind').value
        ).strip()
        self.recording_root_dir = str(
            self.get_parameter('recording_root_dir').value
        )
        self.emergency_discard_frames = int(
            self.get_parameter('emergency_discard_frames').value
        )
        self.recording_image_format = str(
            self.get_parameter('recording_image_format').value
        ).strip().lower()
        self.recording_jpeg_quality = int(
            self.get_parameter('recording_jpeg_quality').value
        )
        self.recording_png_compression = int(
            self.get_parameter('recording_png_compression').value
        )
        self.recording_queue_size = int(
            self.get_parameter('recording_queue_size').value
        )
        self.recording_min_free_space_mb = int(
            self.get_parameter('recording_min_free_space_mb').value
        )
        self._validate_parameters()
        self.collection_profile_metadata = _collection_profile_metadata(
            self.collection_profile_path
        )

        self.bridge = CvBridge()
        self._command = STOP_COMMAND
        self._last_published_command = STOP_COMMAND
        self._last_joy_monotonic: Optional[float] = None
        self._input_valid = False
        self._arming_gate = NeutralArmingGate(
            self.neutral_trigger_threshold
        )
        self._competitors: tuple[str, ...] = ()
        self._has_motor_subscriber = False
        self._next_graph_check_monotonic = 0.0
        self._stop_reason: Optional[str] = None
        self._recording_gate = (
            MissionSequenceRecordingGate()
            if self.recording_mode == 'mission_sequence'
            else RecordingGate()
        )
        self._recording_tail: deque[CameraSample] = deque()
        self._camera_sequence = 0
        self._lidar_sequence = 0
        self._lidar: Optional[LidarSnapshot] = None
        self._last_lidar_missing_warning_monotonic = -math.inf
        self._session_token: Optional[int] = None
        self._finishing_token: Optional[int] = None
        self._pending_recording_finish: Optional[
            _PendingRecordingFinish
        ] = None
        self._recording_disabled = False
        self._writer_failure_handled = False
        self._shutdown_started = False
        self.writer = AsyncSessionWriter(
            self.recording_root_dir,
            png_compression=self.recording_png_compression,
            queue_size=self.recording_queue_size,
            min_free_space_mb=self.recording_min_free_space_mb,
            image_format=self.recording_image_format,
            jpeg_quality=self.recording_jpeg_quality,
        )

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
        self.camera_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self._on_camera,
            qos_profile_sensor_data,
        )
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            self.lidar_topic,
            self._on_lidar,
            qos_profile_sensor_data,
        )
        self.control_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._on_control_timer,
        )
        self._refresh_graph(time.monotonic())
        self.publish_stop_burst()
        if self.recording_mode == 'mission_sequence':
            recording_help = (
                'press A once to start a full mission sequence, keep recording '
                'through speed=0 stops, and press B to save'
            )
        else:
            recording_help = (
                'hold A to wait for positive speed recording, and press B to save'
            )
        self.get_logger().warning(
            'Gamepad teleop started disarmed. Release LT, RT, and LB once '
            'to arm; '
            f'{recording_help}.'
        )
        steering_sign = '-' if self.config.invert_steering else ''
        self.get_logger().info(
            f'joy={self.joy_topic}, motor={self.motor_topic}, '
            f'angle=axes[{self.config.steering_axis}]*'
            f'{steering_sign}'
            f'{self.config.max_angle:g}, '
            f'speed=depth(axes[{self.config.rt_axis}])*'
            f'{self.config.max_forward_speed:g}-depth(axes['
            f'{self.config.lt_axis}])*'
            f'{self.config.max_reverse_speed:g}, '
            f'LB=buttons[{self.config.fixed_reverse_button}]->speed-='
            f'{self.config.fixed_reverse_speed:g}, '
            f'trigger_axis_mode={self.config.trigger_axis_mode}'
        )
        self.get_logger().info(
            f'camera={self.camera_topic}, lidar={self.lidar_topic} '
            f'(optional, timeout={self.lidar_timeout_sec:g}s, '
            f'max_skew={self.max_lidar_skew_sec:g}s), '
            f'dataset_root={self.recording_root_dir}, '
            f'A=buttons[{self.record_start_button}], '
            f'B=buttons[{self.record_stop_button}], '
            f'recording_mode={self.recording_mode}, '
            f'capture_kind={self.capture_kind or "none"}, '
            f'emergency_discard_frames={self.emergency_discard_frames}, '
            f'image_format={self.recording_image_format}, '
            f'jpeg_quality={self.recording_jpeg_quality}'
        )

    def _validate_parameters(self) -> None:
        if not self.joy_topic:
            raise ValueError('joy_topic must not be empty')
        if not self.motor_topic:
            raise ValueError('motor_topic must not be empty')
        require_steering_contract_name(self.steering_contract)
        _validate_collection_profile(
            self.collection_profile_path,
            node_name=(
                'mission_sequence_collector'
                if self.recording_mode == 'mission_sequence'
                else 'gamepad_teleop'
            ),
        )
        _validate_config(self.config)
        _validate_runtime_parameters(
            self.publish_rate_hz,
            self.joy_timeout_sec,
            self.graph_check_period_sec,
            self.neutral_trigger_threshold,
            self.stop_publish_count,
        )
        _validate_lidar_parameters(
            self.lidar_topic,
            self.lidar_timeout_sec,
            self.max_lidar_skew_sec,
        )
        _validate_recording_parameters(
            camera_topic=self.camera_topic,
            recording_root_dir=self.recording_root_dir,
            record_start_button=self.record_start_button,
            record_stop_button=self.record_stop_button,
            emergency_discard_frames=self.emergency_discard_frames,
            recording_image_format=self.recording_image_format,
            recording_jpeg_quality=self.recording_jpeg_quality,
            recording_png_compression=self.recording_png_compression,
            recording_queue_size=self.recording_queue_size,
            recording_min_free_space_mb=(
                self.recording_min_free_space_mb
            ),
        )
        if self.config.fixed_reverse_button in {
            self.record_start_button,
            self.record_stop_button,
        }:
            raise ValueError(
                'fixed reverse and record button indices must be distinct'
            )
        if self.recording_mode not in {
            'positive_speed_segment',
            'mission_sequence',
        }:
            raise ValueError(
                'recording_mode must be positive_speed_segment or '
                'mission_sequence'
            )
        if self.recording_mode == 'mission_sequence' and self.capture_kind not in {
            'signal',
            'shortcut',
        }:
            raise ValueError(
                'mission_sequence recording requires capture_kind signal or '
                'shortcut'
            )
        if self.recording_mode != 'mission_sequence' and self.capture_kind:
            raise ValueError(
                'capture_kind is only valid for mission_sequence recording'
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
            mapped_input = map_joy_input(
                message.axes,
                self.config,
                message.buttons,
            )
        except InvalidJoyInput as exc:
            self._command = STOP_COMMAND
            self._input_valid = False
            self._last_joy_monotonic = None
            self._stop_and_disarm(f'invalid Joy input: {exc}')
            self.get_logger().warning(
                f'Ignoring invalid Joy message: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        if len(message.buttons) <= self.config.fixed_reverse_button:
            self.get_logger().warning(
                'Joy button array is too short for the LB fixed reverse; '
                'LT/RT driving remains enabled.',
                throttle_duration_sec=2.0,
            )

        if self._has_motor_subscriber and not self._competitors:
            was_armed = self._arming_gate.armed
            self._arming_gate.observe(
                mapped_input.lt_depth,
                mapped_input.rt_depth,
                mapped_input.fixed_reverse_pressed,
            )
            if self._arming_gate.armed and not was_armed:
                self.get_logger().info(
                    'Gamepad teleop armed after neutral LT/RT/LB input.'
                )
        self._command = mapped_input.command
        self._last_joy_monotonic = now
        self._input_valid = True
        self._handle_record_buttons(message.buttons)

    def _on_control_timer(self) -> None:
        now = time.monotonic()
        self._handle_writer_failure()
        self._poll_writer_results()
        self._retry_pending_recording_finish()
        self._flush_recording_prefix()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)

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
        if not self._arming_gate.armed:
            self._publish_stop_for_reason(
                'waiting for neutral LT/RT/LB input to arm'
            )
            return

        self._stop_reason = None
        self._publish(self._command)

    def _handle_record_buttons(self, buttons: Sequence[int]) -> None:
        if self._recording_disabled:
            return
        required_button = max(
            self.record_start_button,
            self.record_stop_button,
        )
        if len(buttons) <= required_button:
            self.get_logger().warning(
                'Joy button array is too short for A/B recording controls; '
                'driving remains enabled.',
                throttle_duration_sec=2.0,
            )
            return

        action = self._recording_gate.observe_buttons(
            a_pressed=bool(buttons[self.record_start_button]),
            b_pressed=bool(buttons[self.record_stop_button]),
        )
        if action == RecordingAction.WAITING_STARTED:
            self.get_logger().info(
                'Recording waiting: hold A until published speed is positive.'
            )
        elif action == RecordingAction.WAITING_CANCELLED:
            self.get_logger().info('Recording wait cancelled.')
        elif action == RecordingAction.START_RECORDING:
            self._start_recording()
        elif action == RecordingAction.FINISH_NORMAL:
            self._finish_recording(
                reason='b_button',
                discard_emergency_tail=False,
            )

    def _start_recording(self) -> None:
        if (
            self._recording_disabled
            or self._session_token is not None
            or self._finishing_token is not None
        ):
            self._recording_gate.force_finishing()
            return
        mission_sequence = self.recording_mode == 'mission_sequence'
        metadata = {
            'dataset_kind': (
                'competition_mission_sequence'
                if mission_sequence
                else 'camera_first_teleop_behavior_cloning'
            ),
            'camera_is_primary': True,
            'lidar_is_optional': True,
            'control_mode': (
                'gamepad_mission_sequence' if mission_sequence else 'gamepad'
            ),
            'steering_contract': session_steering_contract(
                motor_topic=self.motor_topic
            ),
            'topics': {
                'camera_topic': self.camera_topic,
                'lidar_topic': self.lidar_topic,
                'motor_topic': self.motor_topic,
                'joy_topic': self.joy_topic,
            },
            'sensors': {
                'lidar_timeout_sec': self.lidar_timeout_sec,
                'max_lidar_skew_sec': self.max_lidar_skew_sec,
            },
            'gamepad': {
                **asdict(self.config),
                'record_start_button': self.record_start_button,
                'record_stop_button': self.record_stop_button,
            },
            'recording': {
                'root_dir': self.recording_root_dir,
                'image_format': self.recording_image_format,
                'jpeg_quality': self.recording_jpeg_quality,
                'png_compression': self.recording_png_compression,
                'queue_size': self.recording_queue_size,
                'min_free_space_mb': self.recording_min_free_space_mb,
                'emergency_discard_frames': (
                    self.emergency_discard_frames
                ),
            },
            'collection_profile': dict(self.collection_profile_metadata),
            'runtime_safety': {
                'publish_rate_hz': self.publish_rate_hz,
                'joy_timeout_sec': self.joy_timeout_sec,
                'graph_check_period_sec': self.graph_check_period_sec,
                'neutral_trigger_threshold': (
                    self.neutral_trigger_threshold
                ),
                'stop_publish_count': self.stop_publish_count,
                'allowed_motor_relay_nodes': list(
                    self.allowed_motor_relay_nodes
                ),
            },
        }
        if mission_sequence:
            metadata['mission'] = {
                'capture_kind': self.capture_kind,
                'records_stationary_frames': True,
                'annotation_status': 'unlabeled',
            }
        token = self.writer.start_session(metadata)
        if token is None:
            self._recording_failure(
                'could not queue a new gamepad recording session'
            )
            return
        self._session_token = token
        self._recording_tail.clear()
        if mission_sequence:
            self.get_logger().info(
                f'{self.capture_kind} mission sequence recording started; '
                'stationary frames will be retained.'
            )
        else:
            self.get_logger().info(
                'Gamepad recording started after a positive motor command.'
            )

    def _on_camera(self, message: Image) -> None:
        if (
            self._recording_gate.state != RecordingState.RECORDING
            or self._session_token is None
        ):
            return
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

        command = self._last_published_command
        if (
            self.recording_mode != 'mission_sequence'
            and command.speed <= 0.0
        ):
            return
        self._camera_sequence += 1
        stamp = message.header.stamp
        lidar, lidar_skew_sec = matching_lidar_snapshot(
            received_monotonic,
            self._lidar,
            lidar_timeout_sec=self.lidar_timeout_sec,
            max_lidar_skew_sec=self.max_lidar_skew_sec,
        )
        if lidar is None and (
            received_monotonic
            - self._last_lidar_missing_warning_monotonic
            >= 2.0
        ):
            self._last_lidar_missing_warning_monotonic = (
                received_monotonic
            )
            self.get_logger().warning(
                'LiDAR is unavailable or too old for this camera frame; '
                'saving the camera sample with lidar_valid=false.'
            )
        sample = CameraSample(
            image=np.ascontiguousarray(image).copy(),
            camera_sequence=self._camera_sequence,
            camera_stamp_sec=int(stamp.sec),
            camera_stamp_nanosec=int(stamp.nanosec),
            camera_received_monotonic=received_monotonic,
            camera_received_wall_time_ns=received_wall_time_ns,
            angle=command.angle,
            speed=command.speed,
            input_key='gamepad',
            lidar=lidar,
            lidar_skew_sec=lidar_skew_sec,
        )
        self._recording_tail.append(sample)
        self._flush_recording_prefix()
        backlog_limit = (
            self.recording_queue_size + self.emergency_discard_frames
        )
        if len(self._recording_tail) > backlog_limit:
            self._recording_failure(
                'dataset writer queue remained full; recording backlog limit '
                'was exceeded'
            )

    def _on_lidar(self, message: LaserScan) -> None:
        received_monotonic = time.monotonic()
        received_wall_time_ns = time.time_ns()
        self._lidar_sequence += 1
        stamp = message.header.stamp
        self._lidar = LidarSnapshot(
            sequence=self._lidar_sequence,
            ranges=np.asarray(message.ranges, dtype=np.float32).copy(),
            intensities=np.asarray(
                message.intensities,
                dtype=np.float32,
            ).copy(),
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
            received_monotonic=received_monotonic,
            received_wall_time_ns=received_wall_time_ns,
        )

    def _flush_recording_prefix(self) -> None:
        token = self._session_token
        if token is None:
            return
        while len(self._recording_tail) > self.emergency_discard_frames:
            if not self.writer.submit(token, self._recording_tail[0]):
                return
            self._recording_tail.popleft()

    def _finish_recording(
        self,
        *,
        reason: str,
        discard_emergency_tail: bool,
        complete: bool = True,
    ) -> None:
        token = self._session_token
        if token is None:
            self._recording_tail.clear()
            self._recording_gate.finish_completed()
            return

        buffered = tuple(self._recording_tail)
        self._recording_tail.clear()
        discarded = 0
        final_samples = buffered
        if discard_emergency_tail:
            discarded = min(
                self.emergency_discard_frames,
                len(buffered),
            )
            if discarded:
                final_samples = buffered[:-discarded]

        self._session_token = None
        self._finishing_token = token
        self._pending_recording_finish = _PendingRecordingFinish(
            token=token,
            reason=reason,
            complete=complete,
            extra_metadata={
                'emergency_discard_count': discarded,
                'emergency_discard_frames': (
                    self.emergency_discard_frames
                ),
            },
            final_samples=final_samples,
        )
        self._retry_pending_recording_finish()
        if discard_emergency_tail:
            self.get_logger().warning(
                'Recording stopped because published speed became '
                f'non-positive; discarded latest {discarded} frame(s). '
                'Gamepad driving remains active.'
            )
        else:
            self.get_logger().info(
                'B ended recording normally; final camera frames are '
                'being flushed while driving remains active.'
            )

    def _retry_pending_recording_finish(self) -> None:
        pending = self._pending_recording_finish
        if pending is None or self.writer.failure is not None:
            return
        if self.writer.finish(
            pending.token,
            pending.reason,
            complete=pending.complete,
            extra_metadata=pending.extra_metadata,
            final_samples=pending.final_samples,
        ):
            self._pending_recording_finish = None

    def _poll_writer_results(self) -> None:
        for result in self.writer.poll_results():
            if result.token != self._finishing_token:
                continue
            self._finishing_token = None
            self._recording_gate.finish_completed()
            if result.completed:
                path_text = (
                    str(result.path)
                    if result.path is not None
                    else 'no files (no retained camera samples)'
                )
                self.get_logger().info(
                    f'Gamepad session saved: {path_text}; '
                    f'samples={result.sample_count}; '
                    f'lidar_linked={result.lidar_linked_count}; '
                    f'lidar_missing={result.lidar_missing_count}; '
                    f'reason={result.reason}'
                )
            else:
                self.get_logger().error(
                    'Gamepad session marked incomplete: '
                    f"{result.path or 'no files'}; reason={result.reason}"
                )

    def _recording_failure(self, reason: str) -> None:
        if self._recording_disabled:
            return
        self._recording_disabled = True
        self.get_logger().error(f'Gamepad recording failure: {reason}')
        if self._session_token is not None:
            self._recording_gate.force_finishing()
            self._finish_recording(
                reason=reason,
                discard_emergency_tail=False,
                complete=False,
            )
        else:
            self._recording_gate.force_finishing()
        self._stop_and_disarm(reason)

    def _handle_writer_failure(self) -> None:
        failure = self.writer.failure
        if failure is None or self._writer_failure_handled:
            return
        self._writer_failure_handled = True
        self._recording_disabled = True
        self.get_logger().error(f'Gamepad recording failure: {failure}')
        self._recording_tail.clear()
        self._pending_recording_finish = None
        if self._session_token is not None:
            self._finishing_token = self._session_token
            self._session_token = None
        self._recording_gate.force_finishing()
        self._stop_and_disarm(failure)

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
        self.get_logger().warning(f'Gamepad teleop stopped: {reason}')

    def _stop_and_disarm(self, reason: str) -> None:
        self._arming_gate.disarm()
        self._command = STOP_COMMAND
        self._publish_stop_for_reason(reason)

    def _publish(self, command: DriveCommand) -> None:
        message = Float32MultiArray()
        message.data = [float(command.angle), float(command.speed)]
        self.motor_publisher.publish(message)
        self._last_published_command = command
        if self._recording_disabled:
            return
        action = self._recording_gate.observe_published_speed(command.speed)
        if action == RecordingAction.START_RECORDING:
            self._start_recording()
        elif action == RecordingAction.FINISH_EMERGENCY:
            self._finish_recording(
                reason='speed_nonpositive',
                discard_emergency_tail=True,
            )

    def publish_stop_burst(self) -> None:
        for _ in range(self.stop_publish_count):
            self._publish(STOP_COMMAND)

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.publish_stop_burst()
        deadline = time.monotonic() + 1.0
        while (
            self._pending_recording_finish is not None
            and self.writer.failure is None
            and time.monotonic() < deadline
        ):
            self._retry_pending_recording_finish()
            if self._pending_recording_finish is not None:
                time.sleep(0.01)
        self.writer.shutdown()
        self._poll_writer_results()


def _node_label(namespace: str, name: str) -> str:
    prefix = namespace.rstrip('/')
    label = f'{prefix}/{name}'
    return label if label.startswith('/') else f'/{label}'


def _validate_collection_profile(
    configured_path: str,
    *,
    node_name: str = 'gamepad_teleop',
) -> None:
    if not configured_path:
        raise ValueError('collection_profile_path must not be empty')
    path = Path(configured_path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            'collection_profile_path must be an existing absolute file'
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f'collection profile must be valid UTF-8 YAML: {path}'
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError('collection profile root must be a mapping')
    node_config = payload.get(node_name)
    if not isinstance(node_config, Mapping):
        raise ValueError(f'collection profile must configure {node_name}')
    parameters = node_config.get('ros__parameters')
    if not isinstance(parameters, Mapping):
        raise ValueError(f'{node_name}.ros__parameters must be a mapping')
    require_steering_contract_name(parameters.get('steering_contract'))


def _collection_profile_metadata(configured_path: str) -> dict[str, object]:
    if not configured_path:
        return {'path': None, 'sha256': None}
    path = Path(configured_path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': configured_path, 'sha256': digest.hexdigest()}


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
