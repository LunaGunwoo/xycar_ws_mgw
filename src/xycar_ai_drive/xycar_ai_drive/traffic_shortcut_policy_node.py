"""Single-publisher traffic-light Base/ResNet18 mission controller."""

from __future__ import annotations

import math
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool, Float32MultiArray

from xycar_ai_drive.control import (
    STOP_COMMAND,
    DriveCommand,
    HoldDriveGate,
    ToggleAction,
    cap_command_speed,
    command_history_token_ids,
    is_fresh,
)
from xycar_ai_drive.front_cam_policy_node import (
    _is_paired_unnamed_relay,
    _node_label,
)
from xycar_ai_drive.policy_ipc import UnixSocketPolicyClient
from xycar_ai_drive.traffic_light_detector import (
    LampAction,
    TrafficClassifierCadence,
    TrafficLampLatch,
    TrafficSignalLatch,
)
from xycar_ai_drive.traffic_light_runtime import create_onnx_detector
from xycar_ai_drive.traffic_shortcut_artifact import (
    TrafficShortcutBundle,
    load_traffic_shortcut_bundle,
)
from xycar_ai_drive.traffic_shortcut_fsm import (
    MissionState,
    PolicyChoice,
    TrafficShortcutFsm,
)

PolicyClientFactory = Callable[..., object]
DetectorFactory = Callable[[TrafficShortcutBundle], object]
BundleLoader = Callable[[str], TrafficShortcutBundle]


@dataclass(frozen=True)
class MissionDecision:
    command: DriveCommand
    policy: PolicyChoice
    state: MissionState
    source_monotonic: float
    completed_monotonic: float
    inference_ms: float
    frame_sequence: int


class YoloMissingReleaseCounter:
    """Release a latched stop only after a configured run of YOLO misses."""

    def __init__(
        self,
        *,
        release_frames: int,
        inference_every_n_frames: int,
    ) -> None:
        if (
            release_frames < 1
            or inference_every_n_frames < 1
            or release_frames % inference_every_n_frames != 0
        ):
            raise ValueError('invalid YOLO missing-release frame contract')
        self._release_frames = release_frames
        self._inference_every_n_frames = inference_every_n_frames
        self.missing_frames = 0

    def reset(self) -> None:
        self.missing_frames = 0

    def observe(
        self,
        *,
        red_stop_active: bool,
        detector_observed: bool,
        yolo_box_found: bool,
        detector_frame_span: int | None = None,
    ) -> bool:
        if not red_stop_active:
            self.reset()
            return False
        if not detector_observed:
            return False
        if yolo_box_found:
            self.reset()
            return False
        frame_span = (
            self._inference_every_n_frames
            if detector_frame_span is None
            else detector_frame_span
        )
        if frame_span < 1:
            raise ValueError('detector frame span must be positive')
        self.missing_frames += frame_span
        if self.missing_frames < self._release_frames:
            return False
        self.reset()
        return True


class TrafficShortcutPolicyNode(Node):
    """Detect signals, call one selected policy and own the motor topic."""

    def __init__(
        self,
        parameter_overrides: Sequence[Parameter] | None = None,
        *,
        policy_client_factory: PolicyClientFactory = UnixSocketPolicyClient,
        detector_factory: DetectorFactory | None = None,
        bundle_loader: BundleLoader = load_traffic_shortcut_bundle,
    ) -> None:
        super().__init__(
            'traffic_shortcut_policy',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()
        self.bundle = bundle_loader(self.bundle_dir)
        self._validate_bundle_runtime_contract()
        self.bridge = CvBridge()
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._drive_gate = HoldDriveGate(self.a_release_grace_sec)
        self._fsm = TrafficShortcutFsm(
            shortcut_duration_sec=self.bundle.shortcut_duration_sec,
            seamless_base_handoff=self.bundle.base_shadow_enabled,
        )
        if self.bundle.detector.mode == 'yolo_cnn_classifier':
            self._lamp_latch = TrafficSignalLatch(
                bbox_width_min=self.bundle.detector.bbox_width_min,
                bbox_width_max=self.bundle.detector.bbox_width_max,
                red_consecutive_reads=(
                    self.bundle.detector.red_consecutive_reads
                ),
                left_consecutive_reads=(
                    self.bundle.detector.left_consecutive_reads
                ),
                straight_consecutive_reads=(
                    self.bundle.detector.straight_consecutive_reads
                ),
            )
        else:
            self._lamp_latch = TrafficLampLatch(
                bbox_width_min=self.bundle.detector.bbox_width_min,
                bbox_width_max=self.bundle.detector.bbox_width_max,
                red_consecutive_reads=self.bundle.detector.red_consecutive_reads,
                left_consecutive_reads=(
                    self.bundle.detector.left_consecutive_reads
                ),
                straight_consecutive_reads=(
                    self.bundle.detector.straight_consecutive_reads
                ),
            )
        release_frames = self.bundle.red_stop_yolo_missing_release_frames
        self._yolo_missing_release = (
            None
            if release_frames is None
            else YoloMissingReleaseCounter(
                release_frames=release_frames,
                inference_every_n_frames=(
                    self.bundle.detector.inference_every_n_frames
                ),
            )
        )
        self._detector = (
            detector_factory(self.bundle)
            if detector_factory is not None
            else create_onnx_detector(self.bundle)
        )
        self._signal_cadence = TrafficClassifierCadence(
            detector_every_n_frames=(
                self.bundle.detector.inference_every_n_frames
            ),
            classification_every_n_frames_after_detection=(
                self.bundle.detector
                .classification_every_n_frames_after_detection
            ),
            reuse_detected_bbox_between_yolo_frames=(
                self.bundle.detector
                .reuse_detected_bbox_between_yolo_frames
            ),
        )
        self._base_policy = self._create_policy_client(
            policy_client_factory,
            artifact_dir=str(self.bundle.base.root),
            socket_path=self.base_socket_path,
        )
        self._shortcut_policy = self._create_policy_client(
            policy_client_factory,
            artifact_dir=str(self.bundle.shortcut.root),
            socket_path=self.shortcut_socket_path,
        )
        self._validate_client_artifacts()

        self._latest_frame: tuple[int, np.ndarray, float] | None = None
        self._frame_sequence = 0
        self._minimum_next_frame_sequence = 0
        self._decision: MissionDecision | None = None
        self._last_executed_decision_sequence = 0
        self._history = self._initial_external_history()
        self._base_shadow_history: deque[tuple[int, int]] | None = None
        self._base_shadow_decision: MissionDecision | None = None
        self._base_shadow_epoch = 0
        self._last_joy_monotonic: float | None = None
        self._joy_valid = False
        self._last_camera_monotonic: float | None = None
        self._competitors: tuple[str, ...] = ()
        self._has_motor_subscriber = False
        self._next_graph_check_monotonic = 0.0
        self._stop_reason: str | None = None
        self._awaiting_post_reset_decision = False
        self._history_reset_monotonic: float | None = None
        self._transition_stop_pending = False
        self._mission_generation = 0
        self._last_signal = LampAction.UNKNOWN
        self._shutdown_started = False
        self._worker_stop = False

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

        self._worker = threading.Thread(
            target=self._inference_worker,
            name='traffic-shortcut-inference',
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
            'Traffic shortcut policy started OFF. Hold A to drive; release A '
            'to stop. Red always has priority.'
        )
        self.get_logger().info(
            f'bundle={self.bundle.artifact_id}, base_cap='
            f'{self.bundle.base_speed_cap:g}, shortcut_speed='
            f'{self.bundle.shortcut_speed:g}, shortcut_duration='
            f'{self.bundle.shortcut_duration_sec:g}s, '
            f'base_shadow={self.bundle.base_shadow_enabled}, '
            f'A_release_grace={self.a_release_grace_sec:g}s'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('bundle_dir', '')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('motor_topic', '/xycar_motor')
        self.declare_parameter('allowed_motor_relay_nodes', ['/ros_bridge'])
        self.declare_parameter(
            'prediction_topic',
            '/traffic_shortcut/prediction',
        )
        self.declare_parameter('enabled_topic', '/traffic_shortcut/enabled')
        self.declare_parameter('a_button_index', 0)
        self.declare_parameter('a_release_grace_sec', 0.12)
        self.declare_parameter('require_gamepad_hold', True)
        self.declare_parameter('allow_motion', True)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('joy_timeout_sec', 0.25)
        self.declare_parameter('camera_timeout_sec', 0.25)
        self.declare_parameter('inference_timeout_sec', 0.25)
        self.declare_parameter('graph_check_period_sec', 0.5)
        self.declare_parameter('stop_publish_count', 5)
        self.declare_parameter('inference_device', 'cuda')
        self.declare_parameter(
            'base_socket_path',
            '/run/user/1000/xycar-ai/traffic-base.sock',
        )
        self.declare_parameter(
            'shortcut_socket_path',
            '/run/user/1000/xycar-ai/traffic-shortcut.sock',
        )
        self.declare_parameter('inference_rpc_timeout_sec', 0.20)

    def _read_parameters(self) -> None:
        self.bundle_dir = str(self.get_parameter('bundle_dir').value)
        self.camera_topic = str(self.get_parameter('camera_topic').value)
        self.joy_topic = str(self.get_parameter('joy_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.allowed_motor_relay_nodes = tuple(
            str(value)
            for value in self.get_parameter('allowed_motor_relay_nodes').value
        )
        self.prediction_topic = str(
            self.get_parameter('prediction_topic').value
        )
        self.enabled_topic = str(self.get_parameter('enabled_topic').value)
        self.a_button_index = int(self.get_parameter('a_button_index').value)
        self.a_release_grace_sec = float(
            self.get_parameter('a_release_grace_sec').value
        )
        self.require_gamepad_hold = bool(
            self.get_parameter('require_gamepad_hold').value
        )
        self.allow_motion = bool(self.get_parameter('allow_motion').value)
        self.publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        self.joy_timeout_sec = float(
            self.get_parameter('joy_timeout_sec').value
        )
        self.camera_timeout_sec = float(
            self.get_parameter('camera_timeout_sec').value
        )
        self.inference_timeout_sec = float(
            self.get_parameter('inference_timeout_sec').value
        )
        self.graph_check_period_sec = float(
            self.get_parameter('graph_check_period_sec').value
        )
        self.stop_publish_count = int(
            self.get_parameter('stop_publish_count').value
        )
        self.inference_device = str(
            self.get_parameter('inference_device').value
        )
        self.base_socket_path = str(
            self.get_parameter('base_socket_path').value
        )
        self.shortcut_socket_path = str(
            self.get_parameter('shortcut_socket_path').value
        )
        self.inference_rpc_timeout_sec = float(
            self.get_parameter('inference_rpc_timeout_sec').value
        )

    def _validate_parameters(self) -> None:
        for label, value in (
            ('bundle_dir', self.bundle_dir),
            ('camera_topic', self.camera_topic),
            ('joy_topic', self.joy_topic),
            ('motor_topic', self.motor_topic),
            ('prediction_topic', self.prediction_topic),
            ('enabled_topic', self.enabled_topic),
            ('base_socket_path', self.base_socket_path),
            ('shortcut_socket_path', self.shortcut_socket_path),
        ):
            if not value:
                raise ValueError(f'{label} must not be empty')
        if self.base_socket_path == self.shortcut_socket_path:
            raise ValueError('base and shortcut sockets must differ')
        if self.a_button_index < 0:
            raise ValueError('a_button_index must be non-negative')
        for node in self.allowed_motor_relay_nodes:
            if not node.startswith('/') or node.endswith('/'):
                raise ValueError(
                    'allowed_motor_relay_nodes entries must be fully qualified'
                )
        for label, value in (
            ('a_release_grace_sec', self.a_release_grace_sec),
            ('publish_rate_hz', self.publish_rate_hz),
            ('joy_timeout_sec', self.joy_timeout_sec),
            ('camera_timeout_sec', self.camera_timeout_sec),
            ('inference_timeout_sec', self.inference_timeout_sec),
            ('graph_check_period_sec', self.graph_check_period_sec),
            ('inference_rpc_timeout_sec', self.inference_rpc_timeout_sec),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{label} must be finite and positive')
        if self.inference_rpc_timeout_sec > self.inference_timeout_sec:
            raise ValueError(
                'inference_rpc_timeout_sec must not exceed inference_timeout_sec'
            )
        if self.stop_publish_count < 1:
            raise ValueError('stop_publish_count must be positive')
        if self.inference_device != 'cuda':
            raise ValueError('traffic shortcut runtime requires CUDA IPC')

    def _validate_bundle_runtime_contract(self) -> None:
        detector = self.bundle.detector
        expected_width = (
            (40, 225)
            if self.bundle.schema_version in {6, 7, 8, 9, 10, 11, 12, 13}
            else (45, 200)
        )
        expected_votes = (
            (15, 15, 15)
            if self.bundle.schema_version == 13
            else (10, 30, 30)
            if self.bundle.schema_version == 12
            else (30, 30, 30)
            if self.bundle.schema_version in {10, 11}
            else (10, 15, 15)
            if self.bundle.schema_version == 9
            else (3, 15, 15)
            if self.bundle.schema_version in {7, 8}
            else (2, 2, 2)
            if self.bundle.schema_version >= 4
            else (5, 5, 5)
            if self.bundle.schema_version == 3
            else (3, 1, 1)
        )
        if (
            detector.bbox_width_min != expected_width[0]
            or detector.bbox_width_max != expected_width[1]
            or detector.inference_every_n_frames != 3
            or (
                detector.red_consecutive_reads,
                detector.left_consecutive_reads,
                detector.straight_consecutive_reads,
            )
            != expected_votes
        ):
            raise ValueError('traffic detector width/votes/every contract mismatch')
        expected_classification_every = (
            1
            if self.bundle.schema_version in {8, 9, 10, 11, 12, 13}
            else 3
        )
        expected_reuse_detected_bbox = self.bundle.schema_version in {
            8,
            9,
            10,
            11,
            12,
            13,
        }
        if (
            detector.classification_every_n_frames_after_detection
            != expected_classification_every
            or detector.reuse_detected_bbox_between_yolo_frames
            is not expected_reuse_detected_bbox
        ):
            raise ValueError('traffic classifier cadence contract mismatch')
        expected_yolo_missing_release_frames = (
            30 if self.bundle.schema_version >= 5 else None
        )
        if (
            self.bundle.red_stop_yolo_missing_release_frames
            != expected_yolo_missing_release_frames
        ):
            raise ValueError('traffic YOLO missing-release contract mismatch')
        expected_base_speed_cap = (
            35.0 if self.bundle.schema_version in {11, 12, 13} else 25.0
        )
        if self.bundle.base_speed_cap != expected_base_speed_cap:
            raise ValueError(
                f'bundle Base speed cap must be {expected_base_speed_cap:g}'
            )
        if self.bundle.base_speed_cap > self.bundle.base.speed_output_max:
            raise ValueError('bundle Base speed cap exceeds policy maximum')
        if (
            self.bundle.shortcut_speed != 23.0
            or self.bundle.shortcut.fixed_speed != 23.0
        ):
            raise ValueError('bundle shortcut speed must be 23')
        if self.bundle.shortcut_entry_stop_control_cycles != 1:
            raise ValueError('shortcut entry stop must be one control cycle')
        if self.bundle.base_shadow_enabled:
            if (
                self.bundle.shortcut_exit_stop_control_cycles != 0
                or self.bundle.base_shadow_history_update
                != 'capped_prediction_commands'
                or self.bundle.base_shadow_max_age_sec is None
                or self.bundle.base_shadow_max_age_sec
                > self.inference_timeout_sec
            ):
                raise ValueError('Base shadow handoff contract is invalid')
        elif self.bundle.shortcut_exit_stop_control_cycles != 1:
            raise ValueError('legacy shortcut exit stop must be one cycle')

    def _create_policy_client(
        self,
        factory: PolicyClientFactory,
        *,
        artifact_dir: str,
        socket_path: str,
    ):
        return factory(
            artifact_dir=artifact_dir,
            socket_path=socket_path,
            timeout_sec=self.inference_rpc_timeout_sec,
            required_device=self.inference_device,
        )

    def _validate_client_artifacts(self) -> None:
        if getattr(self._base_policy, '_artifact', None) != self.bundle.base:
            raise ValueError('Base IPC client artifact does not match bundle')
        if (
            getattr(self._shortcut_policy, '_artifact', None)
            != self.bundle.shortcut
        ):
            raise ValueError('shortcut IPC client artifact does not match bundle')
        if self.bundle.base_shadow_enabled and (
            getattr(self._base_policy, 'supports_pair_inference', False)
            is not True
            or getattr(self._base_policy, 'paired_artifact_id', None)
            != getattr(self._shortcut_policy, 'artifact_id', None)
            or getattr(self._base_policy, 'paired_artifact_digest', None)
            != getattr(self._shortcut_policy, 'artifact_digest', None)
        ):
            raise ValueError(
                'Base IPC server paired shortcut identity does not match bundle'
            )

    def _initial_external_history(self):
        history = self.bundle.base.history
        if history is None or history.update != 'externally_executed_commands':
            raise ValueError('Base artifact must use external executed history')
        return deque(
            [history.initial_class_ids] * history.frames,
            maxlen=history.frames,
        )

    def _on_joy(self, message: Joy) -> None:
        if not self.require_gamepad_hold:
            return
        now = time.monotonic()
        if len(message.buttons) <= self.a_button_index:
            with self._lock:
                self._joy_valid = False
                self._last_joy_monotonic = None
            self._force_fault('Joy button array is too short for A')
            return
        pressed = bool(message.buttons[self.a_button_index])
        with self._lock:
            self._joy_valid = True
            self._last_joy_monotonic = now
            action = self._observe_drive_gate_locked(
                pressed=pressed,
                now=now,
            )
        if action == ToggleAction.ENABLED:
            self.get_logger().warning('Traffic shortcut motion enabled by A hold.')
        elif action == ToggleAction.DISABLED:
            self._publish_stop()
            self._publish_enabled(False)
            self.get_logger().warning('Traffic shortcut motion disabled.')
        elif action == ToggleAction.REJECTED:
            self._publish_stop()
            self._publish_enabled(False)
            self.get_logger().warning(
                'A hold rejected; release A, restore prerequisites, then hold again.',
                throttle_duration_sec=1.0,
            )

    def _observe_drive_gate_locked(
        self,
        *,
        pressed: bool,
        now: float,
    ) -> ToggleAction:
        action = self._drive_gate.observe(
            pressed=pressed,
            can_enable=self._can_enable_locked(now),
            now_monotonic=now,
        )
        if action == ToggleAction.ENABLED:
            try:
                self._base_policy.reset_history()
                self._shortcut_policy.reset_history()
            except Exception as exc:  # noqa: BLE001 - IPC boundary
                self._drive_gate.fault()
                self._fsm.fault()
                self._stop_reason = f'policy reset failed: {exc}'
                return ToggleAction.REJECTED
            self._fsm.enable()
            self._lamp_latch.reset()
            self._reset_yolo_missing_release_locked()
            self._signal_cadence.reset(
                frame_sequence=self._frame_sequence,
            )
            self._last_signal = LampAction.UNKNOWN
            self._history = self._initial_external_history()
            self._discard_base_shadow_locked()
            self._decision = None
            self._last_executed_decision_sequence = 0
            self._awaiting_post_reset_decision = True
            self._history_reset_monotonic = now
            self._transition_stop_pending = False
            self._minimum_next_frame_sequence = self._frame_sequence
            self._mission_generation += 1
            self._stop_reason = None
            self._publish_enabled(True)
        elif action == ToggleAction.DISABLED:
            self._fsm.disable()
            self._reset_yolo_missing_release_locked()
            self._signal_cadence.reset(
                frame_sequence=self._frame_sequence,
            )
            self._decision = None
            self._awaiting_post_reset_decision = False
            self._history_reset_monotonic = None
            self._transition_stop_pending = False
            self._discard_base_shadow_locked()
            self._mission_generation += 1
        return action

    def _reset_yolo_missing_release_locked(self) -> None:
        if self._yolo_missing_release is not None:
            self._yolo_missing_release.reset()

    def _on_camera(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='rgb8',
            )
        except CvBridgeError as exc:
            self._force_fault(f'camera image conversion failed: {exc}')
            return
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            self._force_fault('camera image is not uint8 RGB')
            return
        now = time.monotonic()
        with self._frame_condition:
            self._frame_sequence += 1
            self._last_camera_monotonic = now
            self._latest_frame = (
                self._frame_sequence,
                np.ascontiguousarray(image.copy()),
                now,
            )
            self._frame_condition.notify()

    def _inference_worker(self) -> None:
        processed_sequence = 0
        while True:
            with self._frame_condition:
                while not self._worker_stop and (
                    self._latest_frame is None
                    or self._latest_frame[0] <= processed_sequence
                ):
                    self._frame_condition.wait()
                if self._worker_stop:
                    return
                assert self._latest_frame is not None
                sequence, image, source_monotonic = self._latest_frame
                processed_sequence = sequence
                if (
                    not self._drive_gate.enabled
                    or self._transition_stop_pending
                    or sequence <= self._minimum_next_frame_sequence
                ):
                    continue
                generation = self._mission_generation
                signal = self._last_signal
                inference_plan = self._signal_cadence.plan(
                    frame_sequence=sequence,
                )
            try:
                reading_observed = False
                reading = None
                detected_box = None
                detector_observed = inference_plan.run_detector
                if inference_plan.run_detector:
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if self.bundle.detector.mode == 'yolo_cnn_classifier':
                        inspection = self._detector.inspect_signal(bgr)
                        if inspection is not None:
                            reading = inspection.reading
                            detected_box = inspection.bbox
                    else:
                        reading = self._detector.read_lamp(bgr)
                    reading_observed = True
                elif inference_plan.classification_box is not None:
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    inspection = self._detector.classify_signal_box(
                        bgr,
                        inference_plan.classification_box,
                    )
                    reading = inspection.reading
                    reading_observed = True
                with self._lock:
                    if generation != self._mission_generation:
                        continue
                    if inference_plan.run_detector:
                        self._signal_cadence.observe_detection(
                            frame_sequence=sequence,
                            box=detected_box,
                        )
                    elif inference_plan.classification_box is not None:
                        self._signal_cadence.observe_classification(
                            frame_sequence=sequence,
                        )
                    if reading_observed:
                        signal = self._lamp_latch.observe(reading)
                    if self._yolo_missing_release is not None:
                        release_after_yolo_loss = (
                            self._yolo_missing_release.observe(
                                red_stop_active=(
                                    self._fsm.state
                                    == MissionState.RED_STOP
                                ),
                                detector_observed=detector_observed,
                                yolo_box_found=detected_box is not None,
                                detector_frame_span=(
                                    inference_plan.detector_frame_span
                                    if detector_observed
                                    else None
                                ),
                            )
                        )
                        if release_after_yolo_loss:
                            if not isinstance(
                                self._lamp_latch,
                                TrafficSignalLatch,
                            ):
                                raise RuntimeError(
                                    'YOLO-loss release requires classifier '
                                    'traffic latch'
                                )
                            self._lamp_latch.release_stop_latch()
                            signal = LampAction.STRAIGHT
                            self.get_logger().warning(
                                'YOLO found no traffic-light box for 30 '
                                'camera frames while stopped; resuming Base.'
                            )
                    self._last_signal = signal
                    plan = self._fsm.on_frame(
                        signal,
                        now_monotonic=time.monotonic(),
                    )
                    if plan.state != MissionState.RED_STOP:
                        self._reset_yolo_missing_release_locked()
                    if plan.publish_stop:
                        if plan.state == MissionState.SWITCH_TO_SHORTCUT:
                            self._transition_stop_pending = True
                            self._decision = None
                            shadow_work = (
                                self._start_base_shadow_locked()
                            )
                        else:
                            self._discard_base_shadow_locked()
                            self._store_stop_decision_locked(
                                sequence=sequence,
                                source_monotonic=source_monotonic,
                            )
                            shadow_work = None
                        self._accept_post_reset_decision_locked(
                            source_monotonic
                        )
                    else:
                        shadow_work = None
                        if plan.policy == PolicyChoice.BASE:
                            self._discard_base_shadow_locked()
                            history = tuple(self._history)
                        elif plan.policy == PolicyChoice.SHORTCUT:
                            shadow_work = (
                                self._base_shadow_snapshot_locked()
                            )
                        generation = self._mission_generation
                if plan.publish_stop:
                    if shadow_work is not None:
                        self._infer_and_store_base_shadow(
                            image=image,
                            source_monotonic=source_monotonic,
                            sequence=sequence,
                            shadow_work=shadow_work,
                        )
                    continue
                shadow_decision = None
                if plan.policy == PolicyChoice.BASE:
                    result = self._base_policy.infer(image, history)
                    decision = self._decision_from_result(
                        result=result,
                        policy=PolicyChoice.BASE,
                        state=plan.state,
                        source_monotonic=source_monotonic,
                        sequence=sequence,
                    )
                elif plan.policy == PolicyChoice.SHORTCUT:
                    if shadow_work is None:
                        result = self._shortcut_policy.infer(image)
                    else:
                        result, shadow_result = self._base_policy.infer_pair(
                            image,
                            shadow_work[1],
                        )
                        shadow_decision = self._decision_from_result(
                            result=shadow_result,
                            policy=PolicyChoice.BASE,
                            state=MissionState.SHORTCUT,
                            source_monotonic=source_monotonic,
                            sequence=sequence,
                        )
                    decision = self._decision_from_result(
                        result=result,
                        policy=PolicyChoice.SHORTCUT,
                        state=plan.state,
                        source_monotonic=source_monotonic,
                        sequence=sequence,
                    )
                else:
                    raise RuntimeError('FSM selected no policy without STOP')
                with self._lock:
                    if generation != self._mission_generation:
                        continue
                    self._decision = decision
                    if shadow_decision is not None:
                        self._store_base_shadow_decision_locked(
                            epoch=shadow_work[0],
                            decision=shadow_decision,
                        )
                    self._accept_post_reset_decision_locked(source_monotonic)
                self._publish_prediction(decision)
            except Exception as exc:  # noqa: BLE001 - fail-closed boundary
                self._force_fault(f'mission inference failed: {exc}')

    def _decision_from_result(
        self,
        *,
        result,
        policy: PolicyChoice,
        state: MissionState,
        source_monotonic: float,
        sequence: int,
    ) -> MissionDecision:
        completed = time.monotonic()
        command = result.command
        inference_ms = float(result.inference_ms)
        if not all(
            math.isfinite(value)
            for value in (
                command.angle,
                command.speed,
                inference_ms,
                completed - source_monotonic,
            )
        ):
            raise ValueError('policy output contains NaN or Inf')
        if policy == PolicyChoice.BASE:
            command = cap_command_speed(
                command,
                self.bundle.base_speed_cap,
                maximum_speed=self.bundle.base.speed_output_max,
            )
        elif not math.isclose(
            command.speed,
            self.bundle.shortcut_speed,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError('shortcut policy speed is not exactly 23')
        return MissionDecision(
            command=command,
            policy=policy,
            state=state,
            source_monotonic=source_monotonic,
            completed_monotonic=completed,
            inference_ms=inference_ms,
            frame_sequence=sequence,
        )

    def _start_base_shadow_locked(self):
        if not self.bundle.base_shadow_enabled:
            return None
        self._base_shadow_epoch += 1
        self._base_shadow_history = deque(
            self._history,
            maxlen=self._history.maxlen,
        )
        self._base_shadow_decision = None
        return self._base_shadow_snapshot_locked()

    def _base_shadow_snapshot_locked(self):
        if (
            not self.bundle.base_shadow_enabled
            or self._base_shadow_history is None
        ):
            return None
        return self._base_shadow_epoch, tuple(self._base_shadow_history)

    def _infer_and_store_base_shadow(
        self,
        *,
        image: np.ndarray,
        source_monotonic: float,
        sequence: int,
        shadow_work,
    ) -> None:
        epoch, history = shadow_work
        result = self._base_policy.infer(image, history)
        decision = self._decision_from_result(
            result=result,
            policy=PolicyChoice.BASE,
            state=MissionState.SHORTCUT,
            source_monotonic=source_monotonic,
            sequence=sequence,
        )
        with self._lock:
            self._store_base_shadow_decision_locked(
                epoch=epoch,
                decision=decision,
            )

    def _store_base_shadow_decision_locked(
        self,
        *,
        epoch: int,
        decision: MissionDecision,
    ) -> None:
        if (
            epoch != self._base_shadow_epoch
            or self._base_shadow_history is None
            or self._fsm.state
            not in {
                MissionState.SWITCH_TO_SHORTCUT,
                MissionState.SHORTCUT,
            }
        ):
            return
        self._base_shadow_history.append(
            command_history_token_ids(
                decision.command,
                speed_max=self.bundle.base.speed_output_max,
            )
        )
        self._base_shadow_decision = decision

    def _discard_base_shadow_locked(self) -> None:
        self._base_shadow_epoch += 1
        self._base_shadow_history = None
        self._base_shadow_decision = None

    def _accept_post_reset_decision_locked(
        self,
        source_monotonic: float,
    ) -> None:
        if not self._awaiting_post_reset_decision:
            return
        reset = self._history_reset_monotonic
        if reset is None or source_monotonic <= reset:
            return
        self._awaiting_post_reset_decision = False
        self._history_reset_monotonic = None

    def _store_stop_decision_locked(
        self,
        *,
        sequence: int,
        source_monotonic: float,
    ) -> None:
        self._decision = MissionDecision(
            command=STOP_COMMAND,
            policy=PolicyChoice.NONE,
            state=self._fsm.state,
            source_monotonic=source_monotonic,
            completed_monotonic=time.monotonic(),
            inference_ms=0.0,
            frame_sequence=sequence,
        )

    def _on_control_timer(self) -> None:
        now = time.monotonic()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)
        auto_action = None
        with self._lock:
            if (
                not self.require_gamepad_hold
                and not self._drive_gate.enabled
                and self._can_enable_locked(now)
            ):
                auto_action = self._observe_drive_gate_locked(
                    pressed=True,
                    now=now,
                )
            reason = None
            if self._drive_gate.enabled:
                reason = self._unsafe_reason_locked(now)
            if reason is None and self._drive_gate.enabled:
                deadline_plan = self._fsm.on_control_tick(
                    now_monotonic=now
                )
                if deadline_plan is not None:
                    if deadline_plan.promote_base_shadow:
                        reason = self._promote_base_shadow_locked(now)
                        if reason is None:
                            return
                    else:
                        self._mission_generation += 1
                        self._minimum_next_frame_sequence = (
                            self._frame_sequence
                        )
                        self._decision = None
                        self._awaiting_post_reset_decision = True
                        self._history_reset_monotonic = now
                        self._publish_and_record_locked(
                            STOP_COMMAND,
                            decision_sequence=None,
                        )
                        return
                if reason is None and self._transition_stop_pending:
                    self._publish_and_record_locked(
                        STOP_COMMAND,
                        decision_sequence=None,
                    )
                    self._transition_stop_pending = False
                    self._mission_generation += 1
                    self._minimum_next_frame_sequence = self._frame_sequence
                    self._decision = None
                    self._awaiting_post_reset_decision = True
                    self._history_reset_monotonic = now
                    return
                decision = self._decision
                if reason is None and decision is not None:
                    self._publish_and_record_locked(
                        decision.command,
                        decision_sequence=decision.frame_sequence,
                    )
                    if (
                        decision.policy == PolicyChoice.SHORTCUT
                        and self._fsm.shortcut_started_monotonic is None
                    ):
                        self._fsm.on_shortcut_command_published(
                            now_monotonic=now
                        )
                    self._stop_reason = None
                    return
        if auto_action == ToggleAction.ENABLED:
            self.get_logger().warning(
                'Traffic shortcut motion enabled without Gamepad hold.'
            )
        elif auto_action == ToggleAction.REJECTED:
            self._force_fault(
                self._stop_reason
                or 'automatic traffic shortcut start failed'
            )
            return
        if reason is not None:
            self._force_fault(reason)
            return
        self._publish_stop()

    def _promote_base_shadow_locked(self, now: float) -> str | None:
        decision = self._base_shadow_decision
        history = self._base_shadow_history
        max_age = self.bundle.base_shadow_max_age_sec
        if not self.bundle.base_shadow_enabled or max_age is None:
            return 'Base shadow handoff was requested by a legacy bundle'
        if decision is None or history is None:
            return 'Base shadow prediction is missing at shortcut handoff'
        age = now - decision.source_monotonic
        if not math.isfinite(age) or age < 0.0 or age > max_age:
            return (
                'Base shadow prediction is stale at shortcut handoff: '
                f'age={age:.3f}s'
            )
        promoted = MissionDecision(
            command=decision.command,
            policy=PolicyChoice.BASE,
            state=MissionState.BASE,
            source_monotonic=decision.source_monotonic,
            completed_monotonic=decision.completed_monotonic,
            inference_ms=decision.inference_ms,
            frame_sequence=decision.frame_sequence,
        )
        promoted_history = deque(history, maxlen=history.maxlen)
        self._discard_base_shadow_locked()
        self._fsm.on_base_shadow_promoted()
        self._history = promoted_history
        self._decision = promoted
        self._last_executed_decision_sequence = promoted.frame_sequence
        self._mission_generation += 1
        self._minimum_next_frame_sequence = promoted.frame_sequence
        self._awaiting_post_reset_decision = False
        self._history_reset_monotonic = None
        self._transition_stop_pending = False
        self._stop_reason = None
        self._publish(promoted.command)
        self._publish_prediction(promoted)
        return None

    def _publish_and_record_locked(
        self,
        command: DriveCommand,
        *,
        decision_sequence: int | None,
    ) -> None:
        self._publish(command)
        if decision_sequence is None:
            self._history.append(
                command_history_token_ids(
                    command,
                    speed_max=self.bundle.base.speed_output_max,
                )
            )
            return
        if decision_sequence != self._last_executed_decision_sequence:
            self._history.append(
                command_history_token_ids(
                    command,
                    speed_max=self.bundle.base.speed_output_max,
                )
            )
            self._last_executed_decision_sequence = decision_sequence

    def _unsafe_reason_locked(self, now: float) -> str | None:
        if self._competitors:
            return 'competing motor publisher(s): ' + ', '.join(
                self._competitors
            )
        if not self._has_motor_subscriber:
            return 'no motor subscriber'
        if self.require_gamepad_hold and (
            not self._joy_valid
            or not is_fresh(
                now,
                self._last_joy_monotonic,
                self.joy_timeout_sec,
            )
        ):
            return 'Joy input is missing or stale'
        if not is_fresh(
            now,
            self._last_camera_monotonic,
            self.camera_timeout_sec,
        ):
            return 'camera input is missing or stale'
        if self._awaiting_post_reset_decision:
            if (
                self._history_reset_monotonic is None
                or now - self._history_reset_monotonic
                > self.inference_timeout_sec
            ):
                return 'post-reset mission inference is missing or stale'
            return None
        if self._drive_gate.enabled and not self._transition_stop_pending:
            if self._decision is None or not is_fresh(
                now,
                self._decision.source_monotonic,
                self.inference_timeout_sec,
            ):
                return 'selected policy inference is missing or stale'
        return None

    def _can_enable_locked(self, now: float) -> bool:
        if not self.allow_motion:
            return False
        if self._competitors or not self._has_motor_subscriber:
            return False
        if self.require_gamepad_hold and (
            not self._joy_valid
            or not is_fresh(
                now,
                self._last_joy_monotonic,
                self.joy_timeout_sec,
            )
        ):
            return False
        return is_fresh(
            now,
            self._last_camera_monotonic,
            self.camera_timeout_sec,
        )

    def _refresh_graph(self, now: float) -> None:
        topic = self.resolve_topic_name(self.motor_topic)
        subscriptions = self.get_subscriptions_info_by_topic(topic)
        allow_unnamed_bridge = '/ros_bridge' in self.allowed_motor_relay_nodes
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
        with self._lock:
            self._competitors = tuple(sorted(set(competitors)))
            self._has_motor_subscriber = bool(subscriptions)
            self._next_graph_check_monotonic = (
                now + self.graph_check_period_sec
            )

    def _force_fault(self, reason: str) -> None:
        with self._lock:
            was_enabled = self._drive_gate.fault()
            self._fsm.fault()
            changed_reason = reason != self._stop_reason
            self._stop_reason = reason
            self._decision = None
            self._awaiting_post_reset_decision = False
            self._history_reset_monotonic = None
            self._transition_stop_pending = False
            self._discard_base_shadow_locked()
            self._signal_cadence.reset(
                frame_sequence=self._frame_sequence,
            )
            self._mission_generation += 1
            self._publish_stop()
        if was_enabled:
            self._publish_enabled(False)
        if changed_reason:
            self.get_logger().warning(f'Traffic shortcut motion stopped: {reason}')

    def _publish_prediction(self, decision: MissionDecision) -> None:
        message = Float32MultiArray()
        message.data = [
            float(decision.command.angle),
            float(decision.command.speed),
            float(decision.inference_ms),
            1.0 if decision.policy == PolicyChoice.SHORTCUT else 0.0,
        ]
        self.prediction_publisher.publish(message)

    def _publish(self, command: DriveCommand) -> None:
        message = Float32MultiArray()
        message.data = [float(command.angle), float(command.speed)]
        self.motor_publisher.publish(message)

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
        with self._frame_condition:
            self._worker_stop = True
            self._drive_gate.fault()
            self._fsm.fault()
            self._reset_yolo_missing_release_locked()
            self._signal_cadence.reset(
                frame_sequence=self._frame_sequence,
            )
            self._mission_generation += 1
            self._frame_condition.notify_all()
        self.publish_stop_burst()
        self._publish_enabled(False)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        for policy in (self._base_policy, self._shortcut_policy):
            close = getattr(policy, 'close', None)
            if close is not None:
                close()
        self.publish_stop_burst()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: TrafficShortcutPolicyNode | None = None
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
        node = TrafficShortcutPolicyNode()
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
