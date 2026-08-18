"""Collect intervention-aware policy data with one safe motor publisher."""

from __future__ import annotations

import hashlib
import math
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool, Float32MultiArray
from xycar_data.session_writer import AsyncSessionWriter, CameraSample

from xycar_ai_drive.artifact import PolicyArtifact, load_policy_artifact
from xycar_ai_drive.control import (
    STOP_COMMAND,
    DriveCommand,
    ToggleAction,
    ToggleDriveGate,
    command_class_ids,
    is_fresh,
)
from xycar_ai_drive.front_cam_policy_node import (
    _is_paired_unnamed_relay,
    _node_label,
)
from xycar_ai_drive.policy_ipc import UnixSocketPolicyClient
from xycar_ai_drive.policy_runtime import TorchScriptPolicy
from xycar_ai_drive.steering_contract import (
    require_normalized_steering_contract,
    require_steering_contract_name,
    session_steering_contract,
)

PolicyFactory = Callable[..., object]


@dataclass(frozen=True)
class GuideInput:
    steering_axis: float = 0.0
    steering_takeover: bool = False
    lt_depth: float = 0.0
    rt_depth: float = 0.0


@dataclass(frozen=True)
class FusedCommand:
    executed: DriveCommand
    steering_residual: float
    speed_delta: float
    human_correction: bool


@dataclass(frozen=True)
class GuidedPrediction:
    sequence: int
    command: DriveCommand
    source_monotonic: float
    completed_monotonic: float
    inference_ms: float
    image_bgr: np.ndarray
    stamp_sec: int
    stamp_nanosec: int
    received_wall_time_ns: int


@dataclass(frozen=True)
class _Frame:
    sequence: int
    image_rgb: np.ndarray
    image_bgr: np.ndarray
    received_monotonic: float
    received_wall_time_ns: int
    stamp_sec: int
    stamp_nanosec: int


@dataclass(frozen=True)
class _PendingFinish:
    token: int
    reason: str
    complete: bool
    discarded: int
    final_samples: tuple[CameraSample, ...]
    delete_session: bool = False


def fuse_guided_command(
    model: DriveCommand,
    guide: GuideInput,
    *,
    max_steering_angle: float,
    invert_steering: bool,
    rt_speed_increment: float,
    lt_speed_decrement: float,
    speed_cap: float,
    correction_deadzone: float,
) -> FusedCommand:
    steering_sign = -1.0 if invert_steering else 1.0
    controller_angle = max(
        -100.0,
        min(
            100.0,
            guide.steering_axis * max_steering_angle * steering_sign,
        ),
    )
    selected_angle = (
        controller_angle if guide.steering_takeover else model.angle
    )
    angle = max(-100.0, min(100.0, selected_angle))
    residual = angle - model.angle
    speed_delta = (
        guide.rt_depth * rt_speed_increment
        - guide.lt_depth * lt_speed_decrement
    )
    speed = max(0.0, min(speed_cap, model.speed + speed_delta))
    correction = (
        guide.steering_takeover
        or guide.lt_depth > correction_deadzone
        or guide.rt_depth > correction_deadzone
    )
    return FusedCommand(
        executed=DriveCommand(angle=angle, speed=speed),
        steering_residual=residual,
        speed_delta=speed_delta,
        human_correction=correction,
    )


def trigger_depth(raw: float, mode: str) -> float:
    if not math.isfinite(raw):
        raise ValueError('trigger axis must be finite')
    raw = max(-1.0, min(1.0, raw))
    if mode == 'negative':
        return max(0.0, -raw)
    if mode == 'positive':
        return max(0.0, raw)
    if mode == 'signed':
        return (1.0 - raw) / 2.0
    raise ValueError('trigger_axis_mode must be negative, positive, or signed')


def _validate_control_indices(
    axis_indices: Sequence[int],
    button_indices: Sequence[int],
) -> None:
    if any(index < 0 for index in (*axis_indices, *button_indices)):
        raise ValueError('axis and button indices must be non-negative')
    if len(set(button_indices)) != len(button_indices):
        raise ValueError('A/B/X/Y/RB button indices must be distinct')


class GuidedPolicyCollectorNode(Node):
    """Fuse AI and human commands while recording executed-policy labels."""

    def __init__(
        self,
        parameter_overrides: Sequence[Parameter] | None = None,
        *,
        policy_factory: PolicyFactory = TorchScriptPolicy,
    ) -> None:
        super().__init__(
            'guided_policy_collector',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()
        self.collection_profile_metadata = _collection_profile_metadata(
            self.collection_profile_path
        )

        self.artifact: PolicyArtifact = load_policy_artifact(
            self.artifact_dir
        )
        require_normalized_steering_contract(
            self.artifact.steering_contract,
            context='guided parent artifact steering contract',
        )
        self.bridge = CvBridge()
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._drive_gate = ToggleDriveGate()
        self._latest_frame: _Frame | None = None
        self._frame_sequence = 0
        self._prediction: GuidedPrediction | None = None
        self._guide = GuideInput()
        self._last_joy_monotonic: float | None = None
        self._joy_valid = False
        self._last_buttons: list[bool] = []
        self._competitors: tuple[str, ...] = ()
        self._has_motor_subscriber = False
        self._next_graph_check_monotonic = 0.0
        self._stop_reason: str | None = None
        self._awaiting_post_reset_prediction = False
        self._history_reset_monotonic: float | None = None
        self._history = self._initial_history()
        self._last_executed_prediction_sequence = 0
        self._last_published_command = STOP_COMMAND
        self._shutdown_started = False
        self._worker_stop = False

        self._session_token: int | None = None
        self._finishing_token: int | None = None
        self._pending_finish: _PendingFinish | None = None
        self._recording_tail: deque[CameraSample] = deque()
        self._recording_disabled = False
        self._writer_failure_handled = False
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
        self.prediction_publisher = self.create_publisher(
            Float32MultiArray,
            self.prediction_topic,
            10,
        )
        self.enabled_publisher = self.create_publisher(
            Bool,
            self.enabled_topic,
            10,
        )
        self.publish_stop_burst()
        self._publish_enabled(False)

        self._policy = self._create_policy(policy_factory)
        self._worker = threading.Thread(
            target=self._inference_worker,
            name='guided-policy-inference',
            daemon=True,
        )
        self._worker.start()
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
        self.get_logger().warning(
            'Guided collector started DRIVE OFF. Release and press Y to '
            'toggle motion; hold RB for human steering, A starts, B saves, '
            'and X stops motion and deletes the session.'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('artifact_dir', '')
        self.declare_parameter('collection_profile_path', '')
        self.declare_parameter('steering_contract', '')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('motor_topic', '/xycar_motor')
        self.declare_parameter(
            'allowed_motor_relay_nodes',
            ['/ros_bridge'],
        )
        self.declare_parameter(
            'prediction_topic',
            '/guided_policy/prediction',
        )
        self.declare_parameter('enabled_topic', '/guided_policy/enabled')
        self.declare_parameter('steering_axis', 0)
        self.declare_parameter('lt_axis', 4)
        self.declare_parameter('rt_axis', 5)
        self.declare_parameter('trigger_axis_mode', 'negative')
        self.declare_parameter('invert_steering', True)
        self.declare_parameter('max_steering_angle', 100.0)
        self.declare_parameter('rt_speed_increment', 2.0)
        self.declare_parameter('lt_speed_decrement', 5.0)
        self.declare_parameter('speed_cap', 30.0)
        self.declare_parameter('correction_deadzone', 0.05)
        self.declare_parameter('record_start_button', 0)
        self.declare_parameter('record_stop_button', 1)
        self.declare_parameter('record_discard_button', 2)
        self.declare_parameter('drive_toggle_button', 3)
        self.declare_parameter('steering_takeover_button', 10)
        self.declare_parameter('allow_motion', True)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('joy_timeout_sec', 0.25)
        self.declare_parameter('inference_timeout_sec', 0.25)
        self.declare_parameter('graph_check_period_sec', 0.5)
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
        self.declare_parameter(
            'recording_root_dir',
            '/home/xytron/xycar_data/teleop',
        )
        self.declare_parameter('tail_discard_frames', 10)
        self.declare_parameter('recording_image_format', 'jpeg')
        self.declare_parameter('recording_jpeg_quality', 95)
        self.declare_parameter('recording_png_compression', 3)
        self.declare_parameter('recording_queue_size', 128)
        self.declare_parameter('recording_min_free_space_mb', 1024)
        self.declare_parameter('curriculum_generation', 1)

    def _read_parameters(self) -> None:
        def value(name: str):
            return self.get_parameter(name).value

        for name in (
            'artifact_dir',
            'collection_profile_path',
            'steering_contract',
            'camera_topic',
            'joy_topic',
            'motor_topic',
            'prediction_topic',
            'enabled_topic',
            'trigger_axis_mode',
            'inference_backend',
            'inference_device',
            'inference_socket_path',
            'recording_root_dir',
            'recording_image_format',
        ):
            setattr(self, name, str(value(name)))
        self.recording_image_format = self.recording_image_format.strip().lower()
        self.allowed_motor_relay_nodes = tuple(
            str(item) for item in value('allowed_motor_relay_nodes')
        )
        for name in (
            'steering_axis',
            'lt_axis',
            'rt_axis',
            'record_start_button',
            'record_stop_button',
            'record_discard_button',
            'drive_toggle_button',
            'steering_takeover_button',
            'stop_publish_count',
            'torch_num_threads',
            'warmup_count',
            'tail_discard_frames',
            'recording_jpeg_quality',
            'recording_png_compression',
            'recording_queue_size',
            'recording_min_free_space_mb',
            'curriculum_generation',
        ):
            setattr(self, name, int(value(name)))
        for name in (
            'max_steering_angle',
            'rt_speed_increment',
            'lt_speed_decrement',
            'speed_cap',
            'correction_deadzone',
            'publish_rate_hz',
            'joy_timeout_sec',
            'inference_timeout_sec',
            'graph_check_period_sec',
            'inference_rpc_timeout_sec',
        ):
            setattr(self, name, float(value(name)))
        self.invert_steering = bool(value('invert_steering'))
        self.allow_motion = bool(value('allow_motion'))

    def _validate_parameters(self) -> None:
        for name in (
            'artifact_dir',
            'camera_topic',
            'joy_topic',
            'motor_topic',
            'prediction_topic',
            'enabled_topic',
            'inference_socket_path',
            'recording_root_dir',
        ):
            if not getattr(self, name):
                raise ValueError(f'{name} must not be empty')
        _validate_collection_profile(self.collection_profile_path)
        require_steering_contract_name(self.steering_contract)
        axis_indices = (
            self.steering_axis,
            self.lt_axis,
            self.rt_axis,
        )
        button_indices = (
            self.record_start_button,
            self.record_stop_button,
            self.record_discard_button,
            self.drive_toggle_button,
            self.steering_takeover_button,
        )
        _validate_control_indices(axis_indices, button_indices)
        if self.trigger_axis_mode not in {'negative', 'positive', 'signed'}:
            raise ValueError('unsupported trigger_axis_mode')
        positive_values = (
            self.rt_speed_increment,
            self.lt_speed_decrement,
            self.speed_cap,
            self.publish_rate_hz,
            self.joy_timeout_sec,
            self.inference_timeout_sec,
            self.graph_check_period_sec,
            self.inference_rpc_timeout_sec,
        )
        if any(
            not math.isfinite(item) or item <= 0
            for item in positive_values
        ):
            raise ValueError(
                'runtime gains and timeouts must be finite and positive'
            )
        if not 0 <= self.correction_deadzone < 1:
            raise ValueError('correction_deadzone must be in [0,1)')
        if (
            not math.isfinite(self.max_steering_angle)
            or not 0 < self.max_steering_angle <= 100
        ):
            raise ValueError('max_steering_angle must be in (0,100]')
        if self.speed_cap != 30.0:
            raise ValueError('speed_cap must be exactly 30')
        if self.tail_discard_frames < 0 or self.curriculum_generation < 0:
            raise ValueError(
                'tail discard and generation must be non-negative'
            )
        if self.recording_image_format not in {'jpeg', 'png'}:
            raise ValueError('recording_image_format must be jpeg or png')
        if not 1 <= self.recording_jpeg_quality <= 100:
            raise ValueError('recording_jpeg_quality must be in [1,100]')
        if not 0 <= self.recording_png_compression <= 9:
            raise ValueError('recording_png_compression must be in [0,9]')
        if self.recording_queue_size < 1 or self.stop_publish_count < 1:
            raise ValueError('queue and stop publish counts must be positive')
        if self.recording_min_free_space_mb < 0:
            raise ValueError('minimum free space must be non-negative')
        if self.inference_backend not in {'local', 'unix'}:
            raise ValueError('inference_backend must be local or unix')
        if self.inference_device not in {'cpu', 'cuda'}:
            raise ValueError('inference_device must be cpu or cuda')
        if (
            self.inference_backend == 'local'
            and self.inference_device != 'cpu'
        ):
            raise ValueError('local inference requires cpu')
        if self.inference_rpc_timeout_sec > self.inference_timeout_sec:
            raise ValueError('RPC timeout must not exceed inference timeout')
        for node in self.allowed_motor_relay_nodes:
            if not node.startswith('/') or node.endswith('/'):
                raise ValueError('allowed relay nodes must be fully qualified')

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

    def _initial_history(self) -> deque[tuple[int, int]] | None:
        history = self.artifact.history
        if history is None:
            return None
        return deque(
            [history.initial_class_ids] * history.frames,
            maxlen=history.frames,
        )

    def _on_joy(self, message: Joy) -> None:
        now = time.monotonic()
        required_axis = max(self.steering_axis, self.lt_axis, self.rt_axis)
        required_button = max(
            self.record_start_button,
            self.record_stop_button,
            self.record_discard_button,
            self.drive_toggle_button,
            self.steering_takeover_button,
        )
        if (
            len(message.axes) <= required_axis
            or len(message.buttons) <= required_button
        ):
            self._force_off('Joy axis or button array is too short')
            return
        buttons = [bool(value) for value in message.buttons]
        try:
            steering = float(message.axes[self.steering_axis])
            if not math.isfinite(steering):
                raise ValueError('steering axis must be finite')
            guide = GuideInput(
                steering_axis=max(-1.0, min(1.0, steering)),
                steering_takeover=buttons[self.steering_takeover_button],
                lt_depth=trigger_depth(
                    float(message.axes[self.lt_axis]),
                    self.trigger_axis_mode,
                ),
                rt_depth=trigger_depth(
                    float(message.axes[self.rt_axis]),
                    self.trigger_axis_mode,
                ),
            )
        except (TypeError, ValueError) as exc:
            self._force_off(f'invalid Joy input: {exc}')
            return

        with self._lock:
            previous = self._last_buttons

            def rising(index: int) -> bool:
                return buttons[index] and (
                    len(previous) <= index or not previous[index]
                )

            self._last_buttons = buttons
            self._guide = guide
            self._joy_valid = True
            self._last_joy_monotonic = now
            start_pressed = rising(self.record_start_button)
            stop_pressed = rising(self.record_stop_button)
            discard_pressed = rising(self.record_discard_button)
            action = (
                ToggleAction.NONE
                if discard_pressed
                else self._drive_gate.observe(
                    pressed=buttons[self.drive_toggle_button],
                    can_enable=self._can_enable_locked(now),
                )
            )
            if action == ToggleAction.ENABLED:
                self._policy.reset_history()
                self._history = self._initial_history()
                self._prediction = None
                self._awaiting_post_reset_prediction = True
                self._history_reset_monotonic = now
                self._stop_reason = None
        if discard_pressed:
            self._discard_session_and_stop()
            return
        if action == ToggleAction.ENABLED:
            self._publish_enabled(True)
            self.get_logger().warning('Guided motion toggled ON by Y.')
        elif action == ToggleAction.DISABLED:
            self._publish_stop()
            self._publish_enabled(False)
            self._finish_active_session(
                reason='y_emergency_stop',
                discard_tail=True,
                complete=True,
            )
            self.get_logger().warning('Guided emergency stop triggered by Y.')
        elif action == ToggleAction.REJECTED:
            self._publish_stop()
            self.get_logger().warning(
                'Y toggle rejected; release and retry after prerequisites '
                'recover.',
                throttle_duration_sec=1.0,
            )
        if start_pressed:
            self._start_recording()
        if stop_pressed:
            self._finish_active_session(
                reason='b_button',
                discard_tail=False,
                complete=True,
            )
    def _on_camera(self, message: Image) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding='rgb8')
        except CvBridgeError as exc:
            self._force_off(f'camera image conversion failed: {exc}')
            return
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
        ):
            self._force_off('camera image is not uint8 RGB')
            return
        rgb = np.ascontiguousarray(rgb.copy())
        stamp = message.header.stamp
        with self._frame_condition:
            self._frame_sequence += 1
            self._latest_frame = _Frame(
                sequence=self._frame_sequence,
                image_rgb=rgb,
                image_bgr=np.ascontiguousarray(rgb[:, :, ::-1]),
                received_monotonic=time.monotonic(),
                received_wall_time_ns=time.time_ns(),
                stamp_sec=int(stamp.sec),
                stamp_nanosec=int(stamp.nanosec),
            )
            self._frame_condition.notify()

    def _inference_worker(self) -> None:
        processed_sequence = 0
        while True:
            with self._frame_condition:
                while not self._worker_stop and (
                    self._latest_frame is None
                    or self._latest_frame.sequence <= processed_sequence
                ):
                    self._frame_condition.wait()
                if self._worker_stop:
                    return
                frame = self._latest_frame
                if frame is None:
                    continue
                processed_sequence = frame.sequence
                history = (
                    None
                    if self._history is None
                    else tuple(self._history)
                )
            try:
                contract = self.artifact.history
                if (
                    contract is not None
                    and contract.update == 'externally_executed_commands'
                ):
                    result = self._policy.infer(frame.image_rgb, history)
                else:
                    result = self._policy.infer(frame.image_rgb)
            except Exception as exc:  # noqa: BLE001
                self._force_off(f'policy inference failed: {exc}')
                continue
            prediction = GuidedPrediction(
                sequence=frame.sequence,
                command=result.command,
                source_monotonic=frame.received_monotonic,
                completed_monotonic=time.monotonic(),
                inference_ms=float(result.inference_ms),
                image_bgr=frame.image_bgr,
                stamp_sec=frame.stamp_sec,
                stamp_nanosec=frame.stamp_nanosec,
                received_wall_time_ns=frame.received_wall_time_ns,
            )
            values = (
                prediction.command.angle,
                prediction.command.speed,
                prediction.inference_ms,
            )
            if not all(math.isfinite(value) for value in values):
                self._force_off('policy prediction is non-finite')
                continue
            with self._lock:
                if self._awaiting_post_reset_prediction:
                    reset_at = self._history_reset_monotonic
                    if (
                        reset_at is None
                        or prediction.source_monotonic <= reset_at
                    ):
                        continue
                    self._awaiting_post_reset_prediction = False
                    self._history_reset_monotonic = None
                self._prediction = prediction
            message = Float32MultiArray()
            message.data = [
                float(prediction.command.angle),
                float(prediction.command.speed),
                prediction.inference_ms,
            ]
            self.prediction_publisher.publish(message)

    def _on_control_timer(self) -> None:
        now = time.monotonic()
        self._handle_writer_failure()
        self._poll_writer_results()
        self._retry_pending_finish()
        self._flush_recording_prefix()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)
        with self._lock:
            reason = self._unsafe_reason_locked(now)
            enabled = self._drive_gate.enabled
            prediction = self._prediction
            guide = self._guide
        if reason is not None:
            self._force_off(reason)
            return
        if not enabled or prediction is None:
            self._publish_stop()
            return
        fused = fuse_guided_command(
            prediction.command,
            guide,
            max_steering_angle=self.max_steering_angle,
            invert_steering=self.invert_steering,
            rt_speed_increment=self.rt_speed_increment,
            lt_speed_decrement=self.lt_speed_decrement,
            speed_cap=self.speed_cap,
            correction_deadzone=self.correction_deadzone,
        )
        self._publish(fused.executed)
        if prediction.sequence == self._last_executed_prediction_sequence:
            return
        self._last_executed_prediction_sequence = prediction.sequence
        with self._lock:
            if self._history is not None:
                self._history.append(command_class_ids(fused.executed))
        if self._session_token is not None and fused.executed.speed > 0:
            self._record_prediction(prediction, guide, fused)

    def _unsafe_reason_locked(self, now: float) -> str | None:
        if self._competitors:
            return 'competing motor publisher(s): ' + ', '.join(
                self._competitors
            )
        if not self._has_motor_subscriber:
            return 'no motor subscriber'
        if not self._joy_valid or not is_fresh(
            now,
            self._last_joy_monotonic,
            self.joy_timeout_sec,
        ):
            return 'Joy input is missing or stale'
        if self._awaiting_post_reset_prediction:
            if (
                self._history_reset_monotonic is None
                or now - self._history_reset_monotonic
                > self.inference_timeout_sec
            ):
                return 'post-reset camera inference is missing or stale'
            return None
        if self._prediction is None or not is_fresh(
            now,
            self._prediction.source_monotonic,
            self.inference_timeout_sec,
        ):
            return 'camera inference is missing or stale'
        return None

    def _can_enable_locked(self, now: float) -> bool:
        guide_is_neutral = (
            not self._guide.steering_takeover
            and self._guide.lt_depth <= self.correction_deadzone
            and self._guide.rt_depth <= self.correction_deadzone
        )
        return (
            self.allow_motion
            and guide_is_neutral
            and self._unsafe_reason_locked(now) is None
        )

    def _refresh_graph(self, now: float) -> None:
        topic = self.resolve_topic_name(self.motor_topic)
        subscriptions = self.get_subscriptions_info_by_topic(topic)
        allow_unnamed = '/ros_bridge' in self.allowed_motor_relay_nodes
        competitors = []
        for publisher in self.get_publishers_info_by_topic(topic):
            if (
                publisher.node_name == self.get_name()
                and publisher.node_namespace == self.get_namespace()
            ):
                continue
            label = _node_label(publisher.node_namespace, publisher.node_name)
            if label in self.allowed_motor_relay_nodes:
                continue
            if allow_unnamed and _is_paired_unnamed_relay(
                publisher,
                subscriptions,
            ):
                continue
            competitors.append(label)
        with self._lock:
            self._competitors = tuple(sorted(set(competitors)))
            self._has_motor_subscriber = bool(subscriptions)
            self._next_graph_check_monotonic = (
                now + self.graph_check_period_sec
            )

    def _start_recording(self) -> None:
        if self._recording_disabled or self._session_token is not None:
            return
        if (
            self._finishing_token is not None
            or self._pending_finish is not None
        ):
            self.get_logger().warning('Previous session is still being saved.')
            return
        with self._lock:
            if (
                not self._drive_gate.enabled
                or self._last_published_command.speed <= 0
            ):
                self.get_logger().warning(
                    'A ignored: enable Y motion and publish positive '
                    'speed first.'
                )
                return
            history = (
                None
                if self._history is None
                else [list(pair) for pair in self._history]
            )
        curriculum = {
            'generation': self.curriculum_generation,
            'parent_artifact_id': self.artifact.artifact_id,
            'parent_artifact_sha256s_digest': _sha256_file(
                Path(self.artifact_dir) / 'SHA256SUMS'
            ),
        }
        if history is not None:
            curriculum['initial_history_class_ids'] = history
        metadata = {
            'dataset_kind': 'camera_first_teleop_behavior_cloning',
            'camera_is_primary': True,
            'lidar_is_optional': False,
            'control_mode': 'guided_policy',
            'steering_contract': session_steering_contract(
                motor_topic=self.motor_topic
            ),
            'curriculum': curriculum,
            'topics': {
                'camera_topic': self.camera_topic,
                'motor_topic': self.motor_topic,
                'joy_topic': self.joy_topic,
            },
            'guided_control': {
                'steering_mode': (
                    'model_unless_takeover_button_'
                    'controller_absolute_when_held'
                ),
                'max_steering_angle': self.max_steering_angle,
                'invert_steering': self.invert_steering,
                'rt_speed_increment': self.rt_speed_increment,
                'lt_speed_decrement': self.lt_speed_decrement,
                'speed_cap': self.speed_cap,
                'correction_deadzone': self.correction_deadzone,
                'tail_discard_frames': self.tail_discard_frames,
            },
            'gamepad': {
                'max_forward_speed': self.speed_cap,
                'steering_axis': self.steering_axis,
                'lt_axis': self.lt_axis,
                'rt_axis': self.rt_axis,
                'trigger_axis_mode': self.trigger_axis_mode,
                'record_start_button': self.record_start_button,
                'record_stop_button': self.record_stop_button,
                'record_discard_button': self.record_discard_button,
                'drive_toggle_button': self.drive_toggle_button,
                'steering_takeover_button': (
                    self.steering_takeover_button
                ),
            },
            'collection_profile': dict(self.collection_profile_metadata),
            'runtime_safety': {
                'allow_motion': self.allow_motion,
                'publish_rate_hz': self.publish_rate_hz,
                'joy_timeout_sec': self.joy_timeout_sec,
                'inference_timeout_sec': self.inference_timeout_sec,
                'graph_check_period_sec': self.graph_check_period_sec,
                'stop_publish_count': self.stop_publish_count,
                'allowed_motor_relay_nodes': list(
                    self.allowed_motor_relay_nodes
                ),
            },
            'inference_runtime': {
                'backend': self.inference_backend,
                'device': self.inference_device,
                'socket_path': self.inference_socket_path,
                'rpc_timeout_sec': self.inference_rpc_timeout_sec,
                'torch_num_threads': self.torch_num_threads,
                'warmup_count': self.warmup_count,
            },
            'recording': {
                'root_dir': self.recording_root_dir,
                'tail_discard_frames': self.tail_discard_frames,
                'image_format': self.recording_image_format,
                'jpeg_quality': self.recording_jpeg_quality,
                'png_compression': self.recording_png_compression,
                'queue_size': self.recording_queue_size,
                'min_free_space_mb': self.recording_min_free_space_mb,
            },
        }
        token = self.writer.start_session(metadata)
        if token is None:
            self._recording_failure('could not start guided session')
            return
        self._session_token = token
        self._recording_tail.clear()
        self.get_logger().info('A started guided policy recording.')

    def _record_prediction(
        self,
        prediction: GuidedPrediction,
        guide: GuideInput,
        fused: FusedCommand,
    ) -> None:
        sample = CameraSample(
            image=prediction.image_bgr.copy(),
            camera_sequence=prediction.sequence,
            camera_stamp_sec=prediction.stamp_sec,
            camera_stamp_nanosec=prediction.stamp_nanosec,
            camera_received_monotonic=prediction.source_monotonic,
            camera_received_wall_time_ns=prediction.received_wall_time_ns,
            angle=fused.executed.angle,
            speed=fused.executed.speed,
            input_key='guided_policy',
            lidar=None,
            lidar_skew_sec=None,
            model_angle=prediction.command.angle,
            model_speed=prediction.command.speed,
            steering_axis=guide.steering_axis,
            steering_residual=fused.steering_residual,
            lt_depth=guide.lt_depth,
            rt_depth=guide.rt_depth,
            speed_delta=fused.speed_delta,
            human_correction=fused.human_correction,
            inference_ms=prediction.inference_ms,
        )
        self._recording_tail.append(sample)
        self._flush_recording_prefix()
        if len(self._recording_tail) > (
            self.recording_queue_size + self.tail_discard_frames
        ):
            self._recording_failure('guided recording backlog exceeded limit')

    def _flush_recording_prefix(self) -> None:
        token = self._session_token
        if token is None:
            return
        while len(self._recording_tail) > self.tail_discard_frames:
            if not self.writer.submit(token, self._recording_tail[0]):
                return
            self._recording_tail.popleft()

    def _finish_active_session(
        self,
        *,
        reason: str,
        discard_tail: bool,
        complete: bool,
    ) -> None:
        token = self._session_token
        if token is None:
            return
        buffered = tuple(self._recording_tail)
        self._recording_tail.clear()
        discarded = (
            min(self.tail_discard_frames, len(buffered))
            if discard_tail
            else 0
        )
        final_samples = buffered[:-discarded] if discarded else buffered
        self._session_token = None
        self._finishing_token = token
        self._pending_finish = _PendingFinish(
            token=token,
            reason=reason,
            complete=complete,
            discarded=discarded,
            final_samples=final_samples,
        )
        self._retry_pending_finish()

    def _discard_active_session(self, *, reason: str) -> None:
        token = self._session_token
        if token is None:
            return
        self._recording_tail.clear()
        self._session_token = None
        self._finishing_token = token
        self._pending_finish = _PendingFinish(
            token=token,
            reason=reason,
            complete=True,
            discarded=0,
            final_samples=(),
            delete_session=True,
        )
        self._retry_pending_finish()

    def _discard_session_and_stop(self) -> None:
        self._discard_active_session(reason='x_button')
        self._force_off('X button emergency stop')

    def _retry_pending_finish(self) -> None:
        pending = self._pending_finish
        if pending is None or self.writer.failure is not None:
            return
        if pending.delete_session:
            accepted = self.writer.discard(pending.token, pending.reason)
        else:
            accepted = self.writer.finish(
                pending.token,
                pending.reason,
                complete=pending.complete,
                extra_metadata={
                    'emergency_discard_count': pending.discarded,
                    'emergency_discard_frames': self.tail_discard_frames,
                },
                final_samples=pending.final_samples,
            )
        if accepted:
            self._pending_finish = None

    def _poll_writer_results(self) -> None:
        for result in self.writer.poll_results():
            if result.token != self._finishing_token:
                continue
            self._finishing_token = None
            if result.discarded:
                self.get_logger().warning(
                    'Guided motion stopped and session deleted by X; '
                    f'written_samples={result.sample_count}'
                )
            elif result.completed:
                self.get_logger().info(
                    f'Guided session saved: {result.path or "no files"}; '
                    f'samples={result.sample_count}; reason={result.reason}'
                )
            else:
                self.get_logger().error(
                    f'Guided session incomplete: {result.path or "no files"}; '
                    f'reason={result.reason}'
                )

    def _recording_failure(self, reason: str) -> None:
        if self._recording_disabled:
            return
        self._recording_disabled = True
        self._finish_active_session(
            reason=reason,
            discard_tail=False,
            complete=False,
        )
        self._force_off(reason)

    def _handle_writer_failure(self) -> None:
        failure = self.writer.failure
        if failure is None or self._writer_failure_handled:
            return
        self._writer_failure_handled = True
        self._recording_disabled = True
        self._recording_tail.clear()
        self._pending_finish = None
        self._force_off(failure)

    def _force_off(self, reason: str) -> None:
        with self._lock:
            was_enabled = self._drive_gate.fault()
            changed = reason != self._stop_reason
            self._stop_reason = reason
            self._awaiting_post_reset_prediction = False
            self._history_reset_monotonic = None
        self._publish_stop()
        if was_enabled:
            self._publish_enabled(False)
        if self._session_token is not None:
            self._finish_active_session(
                reason=reason,
                discard_tail=False,
                complete=False,
            )
        if changed:
            self.get_logger().warning(f'Guided motion stopped: {reason}')

    def _publish(self, command: DriveCommand) -> None:
        message = Float32MultiArray()
        message.data = [float(command.angle), float(command.speed)]
        self.motor_publisher.publish(message)
        self._last_published_command = command

    def _publish_stop(self) -> None:
        self._publish(STOP_COMMAND)

    def _publish_enabled(self, enabled: bool) -> None:
        message = Bool()
        message.data = enabled
        self.enabled_publisher.publish(message)

    def publish_stop_burst(self) -> None:
        for _ in range(self.stop_publish_count):
            self._publish_stop()

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._finish_active_session(
            reason='shutdown',
            discard_tail=False,
            complete=False,
        )
        with self._frame_condition:
            self._worker_stop = True
            self._drive_gate.fault()
            self._frame_condition.notify_all()
        self.publish_stop_burst()
        self._publish_enabled(False)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        close_policy = getattr(self._policy, 'close', None)
        if close_policy is not None:
            close_policy()
        deadline = time.monotonic() + 1.0
        while self._pending_finish is not None and time.monotonic() < deadline:
            self._retry_pending_finish()
            time.sleep(0.01)
        self.writer.shutdown()
        self._poll_writer_results()
        self.publish_stop_burst()


def _validate_collection_profile(configured_path: str) -> None:
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
    node_config = payload.get('guided_policy_collector')
    if not isinstance(node_config, Mapping):
        raise ValueError(
            'collection profile must configure guided_policy_collector'
        )
    parameters = node_config.get('ros__parameters')
    if not isinstance(parameters, Mapping):
        raise ValueError(
            'guided_policy_collector.ros__parameters must be a mapping'
        )
    if 'residual_gain' in parameters:
        raise ValueError(
            'legacy residual_gain is not supported; replace it with '
            'max_steering_angle'
        )
    if 'max_steering_angle' not in parameters:
        raise ValueError(
            'collection profile must explicitly set max_steering_angle'
        )
    if 'steering_takeover_button' not in parameters:
        raise ValueError(
            'collection profile must explicitly set '
            'steering_takeover_button'
        )
    require_steering_contract_name(parameters.get('steering_contract'))


def _collection_profile_metadata(configured_path: str) -> dict[str, object]:
    if not configured_path:
        return {'path': None, 'sha256': None}
    return {
        'path': configured_path,
        'sha256': _sha256_file(Path(configured_path)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: GuidedPolicyCollectorNode | None = None
    stop_requested = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        node = GuidedPolicyCollectorNode()
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.05)
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
