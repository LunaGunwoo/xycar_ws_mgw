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
    TrafficLampLatch,
    TrafficLightDetector,
)
from xycar_ai_drive.traffic_shortcut_artifact import (
    EXPECTED_NUMPY_VERSION,
    EXPECTED_ONNXRUNTIME_VERSION,
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
        self._lamp_latch = TrafficLampLatch(
            bbox_width_min=self.bundle.detector.bbox_width_min,
            bbox_width_max=self.bundle.detector.bbox_width_max,
            red_consecutive_reads=self.bundle.detector.red_consecutive_reads,
        )
        self._detector = (
            detector_factory(self.bundle)
            if detector_factory is not None
            else _create_onnx_detector(self.bundle)
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
        if self.bundle.base_speed_cap != 25.0:
            raise ValueError('bundle Base speed cap must be 25')
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
                    action = ToggleAction.REJECTED
                    self._stop_reason = f'policy reset failed: {exc}'
                else:
                    self._fsm.enable()
                    self._lamp_latch.reset()
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
                self._decision = None
                self._awaiting_post_reset_decision = False
                self._history_reset_monotonic = None
                self._transition_stop_pending = False
                self._discard_base_shadow_locked()
                self._mission_generation += 1
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
            try:
                reading_observed = False
                reading = None
                if (
                    sequence
                    % self.bundle.detector.inference_every_n_frames
                    == 0
                ):
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    reading = self._detector.read_lamp(bgr)
                    reading_observed = True
                with self._lock:
                    if generation != self._mission_generation:
                        continue
                    if reading_observed:
                        signal = self._lamp_latch.observe(reading)
                    self._last_signal = signal
                    plan = self._fsm.on_frame(
                        signal,
                        now_monotonic=time.monotonic(),
                    )
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
            command_history_token_ids(decision.command)
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
        with self._lock:
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
            self._history.append(command_history_token_ids(command))
            return
        if decision_sequence != self._last_executed_decision_sequence:
            self._history.append(command_history_token_ids(command))
            self._last_executed_decision_sequence = decision_sequence

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
        if not self._joy_valid or not is_fresh(
            now,
            self._last_joy_monotonic,
            self.joy_timeout_sec,
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


def _create_onnx_detector(bundle: TrafficShortcutBundle) -> TrafficLightDetector:
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise ValueError(
            f'host NumPy must be {EXPECTED_NUMPY_VERSION}, got {np.__version__}'
        )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError('onnxruntime is required for traffic detection') from exc
    if ort.__version__ != EXPECTED_ONNXRUNTIME_VERSION:
        raise ValueError(
            f'host ONNX Runtime must be {EXPECTED_ONNXRUNTIME_VERSION}, '
            f'got {ort.__version__}'
        )
    session = ort.InferenceSession(
        str(bundle.detector.model_path),
        providers=list(bundle.providers),
    )
    if tuple(session.get_providers()) != bundle.providers:
        raise ValueError(
            'traffic ONNX active providers do not match CUDA then CPU contract'
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != 'images'
        or inputs[0].type != 'tensor(float)'
        or list(inputs[0].shape) != [1, 3, 640, 640]
    ):
        raise ValueError('traffic ONNX input metadata mismatch')
    if (
        len(outputs) != 1
        or outputs[0].name != 'output0'
        or outputs[0].type != 'tensor(float)'
        or list(outputs[0].shape) != [1, 5, 8400]
    ):
        raise ValueError('traffic ONNX output metadata mismatch')
    return TrafficLightDetector(
        session=session,
        confidence_threshold=bundle.detector.confidence_threshold,
        percentile=bundle.detector.percentile,
    )


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
