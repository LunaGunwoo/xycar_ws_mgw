"""Run front-camera policy inference with an A-button motion toggle."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Sequence

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
    PolicyPrediction,
    ToggleAction,
    ToggleDriveGate,
    is_fresh,
)
from xycar_ai_drive.policy_ipc import UnixSocketPolicyClient
from xycar_ai_drive.policy_runtime import TorchScriptPolicy

PolicyFactory = Callable[..., object]

_DDS_GUID_PREFIX_SIZE = 12
_UNKNOWN_NODE_NAME = '_NODE_NAME_UNKNOWN_'
_UNKNOWN_NODE_NAMESPACE = '_NODE_NAMESPACE_UNKNOWN_'


def _node_label(namespace: str, name: str) -> str:
    namespace = namespace.rstrip('/')
    if not namespace:
        return f'/{name}'
    return f'{namespace}/{name}'


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


class FrontCamPolicyNode(Node):
    """Infer continuously and publish motor commands only when A toggles on."""

    def __init__(
        self,
        parameter_overrides: Sequence[Parameter] | None = None,
        *,
        policy_factory: PolicyFactory = TorchScriptPolicy,
    ) -> None:
        super().__init__(
            'front_cam_policy',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()

        self.bridge = CvBridge()
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._toggle = ToggleDriveGate()
        self._latest_frame: tuple[int, np.ndarray, float] | None = None
        self._frame_sequence = 0
        self._prediction: PolicyPrediction | None = None
        self._last_joy_monotonic: float | None = None
        self._joy_valid = False
        self._competitors: tuple[str, ...] = ()
        self._has_motor_subscriber = False
        self._next_graph_check_monotonic = 0.0
        self._stop_reason: str | None = None
        self._awaiting_post_reset_prediction = False
        self._history_reset_monotonic: float | None = None
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

        self._policy = self._create_policy(policy_factory)
        self._worker = threading.Thread(
            target=self._inference_worker,
            name='front-cam-policy-inference',
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
            'Front-camera policy started with motion OFF. Release A once, '
            'then press A to toggle motion ON; press A again to stop.'
        )
        self.get_logger().info(
            f'artifact={self.artifact_dir}, camera={self.camera_topic}, '
            f'joy={self.joy_topic}, motor={self.motor_topic}, '
            f'publish_rate={self.publish_rate_hz:g} Hz, '
            f'inference_backend={self.inference_backend}, '
            f'inference_device={self.inference_device}, '
            f'A=buttons[{self.a_button_index}], '
            f'allow_motion={self.allow_motion}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('artifact_dir', '')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('motor_topic', '/xycar_motor')
        self.declare_parameter(
            'allowed_motor_relay_nodes',
            ['/ros_bridge'],
        )
        self.declare_parameter(
            'prediction_topic',
            '/front_cam_policy/prediction',
        )
        self.declare_parameter('enabled_topic', '/front_cam_policy/enabled')
        self.declare_parameter('a_button_index', 0)
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

    def _read_parameters(self) -> None:
        self.artifact_dir = str(self.get_parameter('artifact_dir').value)
        self.camera_topic = str(self.get_parameter('camera_topic').value)
        self.joy_topic = str(self.get_parameter('joy_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.allowed_motor_relay_nodes = tuple(
            str(value)
            for value in self.get_parameter(
                'allowed_motor_relay_nodes'
            ).value
        )
        self.prediction_topic = str(
            self.get_parameter('prediction_topic').value
        )
        self.enabled_topic = str(self.get_parameter('enabled_topic').value)
        self.a_button_index = int(self.get_parameter('a_button_index').value)
        self.allow_motion = bool(self.get_parameter('allow_motion').value)
        self.publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        self.joy_timeout_sec = float(
            self.get_parameter('joy_timeout_sec').value
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
        self.inference_backend = str(
            self.get_parameter('inference_backend').value
        )
        self.inference_device = str(
            self.get_parameter('inference_device').value
        )
        self.inference_socket_path = str(
            self.get_parameter('inference_socket_path').value
        )
        self.inference_rpc_timeout_sec = float(
            self.get_parameter('inference_rpc_timeout_sec').value
        )
        self.torch_num_threads = int(
            self.get_parameter('torch_num_threads').value
        )
        self.warmup_count = int(self.get_parameter('warmup_count').value)

    def _validate_parameters(self) -> None:
        for label, value in (
            ('artifact_dir', self.artifact_dir),
            ('camera_topic', self.camera_topic),
            ('joy_topic', self.joy_topic),
            ('motor_topic', self.motor_topic),
            ('prediction_topic', self.prediction_topic),
            ('enabled_topic', self.enabled_topic),
            ('inference_socket_path', self.inference_socket_path),
        ):
            if not value:
                raise ValueError(f'{label} must not be empty')
        if self.a_button_index < 0:
            raise ValueError('a_button_index must be non-negative')
        for node in self.allowed_motor_relay_nodes:
            if not node.startswith('/') or node.endswith('/'):
                raise ValueError(
                    'allowed_motor_relay_nodes entries must be fully '
                    'qualified node names'
                )
        for label, value in (
            ('publish_rate_hz', self.publish_rate_hz),
            ('joy_timeout_sec', self.joy_timeout_sec),
            ('inference_timeout_sec', self.inference_timeout_sec),
            ('inference_rpc_timeout_sec', self.inference_rpc_timeout_sec),
            ('graph_check_period_sec', self.graph_check_period_sec),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{label} must be finite and positive')
        if self.stop_publish_count < 1:
            raise ValueError('stop_publish_count must be positive')
        if self.torch_num_threads < 1:
            raise ValueError('torch_num_threads must be positive')
        if self.warmup_count < 0:
            raise ValueError('warmup_count must be non-negative')
        if self.inference_backend not in {'local', 'unix'}:
            raise ValueError('inference_backend must be local or unix')
        if self.inference_device not in {'cpu', 'cuda'}:
            raise ValueError('inference_device must be cpu or cuda')
        if (
            self.inference_backend == 'local'
            and self.inference_device != 'cpu'
        ):
            raise ValueError('local inference currently requires cpu')
        if (
            self.inference_backend == 'unix'
            and self.inference_rpc_timeout_sec > self.inference_timeout_sec
        ):
            raise ValueError(
                'inference_rpc_timeout_sec must not exceed '
                'inference_timeout_sec'
            )

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

    def _on_joy(self, message: Joy) -> None:
        now = time.monotonic()
        if len(message.buttons) <= self.a_button_index:
            with self._lock:
                self._joy_valid = False
                self._last_joy_monotonic = None
            self._force_off('Joy button array is too short for the A button')
            return

        pressed = bool(message.buttons[self.a_button_index])
        with self._lock:
            self._joy_valid = True
            self._last_joy_monotonic = now
            action = self._toggle.observe(
                pressed=pressed,
                can_enable=self._can_enable_locked(now),
            )
            if action == ToggleAction.ENABLED:
                self._policy.reset_history()
                self._prediction = None
                self._awaiting_post_reset_prediction = True
                self._history_reset_monotonic = now
                self._stop_reason = None
                self._publish_enabled(True)
            elif action == ToggleAction.DISABLED:
                self._awaiting_post_reset_prediction = False
                self._history_reset_monotonic = None
        if action == ToggleAction.ENABLED:
            self.get_logger().warning('AI motion toggled ON by A button.')
        elif action == ToggleAction.DISABLED:
            self._publish_stop()
            self._publish_enabled(False)
            self.get_logger().warning('AI motion toggled OFF by A button.')
        elif action == ToggleAction.REJECTED:
            self._publish_stop()
            self.get_logger().warning(
                'A toggle rejected because motion prerequisites are not ready; '
                'release and press A again.',
                throttle_duration_sec=1.0,
            )

    def _on_camera(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='rgb8',
            )
        except CvBridgeError as exc:
            self._force_off(f'camera image conversion failed: {exc}')
            return
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            self._force_off('camera image is not uint8 RGB')
            return
        with self._frame_condition:
            self._frame_sequence += 1
            self._latest_frame = (
                self._frame_sequence,
                np.ascontiguousarray(image.copy()),
                time.monotonic(),
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
                if self._latest_frame is None:
                    continue
                sequence, image, source_monotonic = self._latest_frame
                processed_sequence = sequence
            try:
                result = self._policy.infer(image)
            except Exception as exc:  # noqa: BLE001 - fail closed at boundary
                self._force_off(f'policy inference failed: {exc}')
                continue
            prediction = PolicyPrediction(
                command=result.command,
                source_monotonic=source_monotonic,
                completed_monotonic=time.monotonic(),
                inference_ms=float(result.inference_ms),
            )
            if not all(
                np.isfinite(value)
                for value in (
                    prediction.command.angle,
                    prediction.command.speed,
                    prediction.inference_ms,
                )
            ):
                self._force_off(
                    'policy prediction contains a non-finite value'
                )
                continue
            reset_again = False
            with self._lock:
                if self._awaiting_post_reset_prediction:
                    reset_monotonic = self._history_reset_monotonic
                    if (
                        reset_monotonic is None
                        or prediction.source_monotonic <= reset_monotonic
                    ):
                        reset_again = True
                    else:
                        self._awaiting_post_reset_prediction = False
                        self._history_reset_monotonic = None
                if not reset_again:
                    self._prediction = prediction
            if reset_again:
                self._policy.reset_history()
                continue
            message = Float32MultiArray()
            message.data = [
                float(prediction.command.angle),
                float(prediction.command.speed),
                float(prediction.inference_ms),
            ]
            self.prediction_publisher.publish(message)

    def _on_control_timer(self) -> None:
        now = time.monotonic()
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)
        with self._lock:
            reason = self._unsafe_reason_locked(now)
            if reason is None and self._toggle.enabled:
                prediction = self._prediction
                if prediction is not None:
                    self._stop_reason = None
                    # Keep the state decision and publish ordered against a
                    # worker-thread fault. A fault that waits on this lock
                    # will publish its stop command after this command.
                    self._publish(prediction.command)
                    return
        if reason is not None:
            self._force_off(reason)
            return
        self._publish_stop()

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
        return self.allow_motion and self._unsafe_reason_locked(now) is None

    def _refresh_graph(self, now: float) -> None:
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
        has_motor_subscriber = bool(subscriptions)
        with self._lock:
            self._competitors = tuple(sorted(set(competitors)))
            self._has_motor_subscriber = has_motor_subscriber
            self._next_graph_check_monotonic = (
                now + self.graph_check_period_sec
            )

    def _force_off(self, reason: str) -> None:
        with self._lock:
            was_enabled = self._toggle.fault()
            changed_reason = reason != self._stop_reason
            self._stop_reason = reason
            self._awaiting_post_reset_prediction = False
            self._history_reset_monotonic = None
        self._publish_stop()
        if was_enabled:
            self._publish_enabled(False)
        if changed_reason:
            self.get_logger().warning(f'AI motion stopped: {reason}')

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
            self._toggle.fault()
            self._frame_condition.notify_all()
        self.publish_stop_burst()
        self._publish_enabled(False)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        close_policy = getattr(self._policy, 'close', None)
        if close_policy is not None:
            close_policy()
        self.publish_stop_burst()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: FrontCamPolicyNode | None = None
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
        node = FrontCamPolicyNode()
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
