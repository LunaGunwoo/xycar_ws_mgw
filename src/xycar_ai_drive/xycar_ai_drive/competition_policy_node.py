"""ROS host node for signal shadow, shortcut-only, and combined driving."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool, Float32MultiArray, String

from xycar_ai_drive.competition_fsm import (
    CompetitionMode,
    MissionDecision,
    MissionStateMachine,
)
from xycar_ai_drive.competition_gpu_runtime import CompetitionInference
from xycar_ai_drive.competition_ipc import CompetitionIpcClient
from xycar_ai_drive.control import (
    STOP_COMMAND,
    DriveCommand,
    ToggleAction,
    ToggleDriveGate,
    is_fresh,
)


_DDS_GUID_PREFIX_SIZE = 12
_UNKNOWN_NODE_NAME = "_NODE_NAME_UNKNOWN_"
_UNKNOWN_NODE_NAMESPACE = "_NODE_NAMESPACE_UNKNOWN_"


@dataclass(frozen=True)
class _CompetitionPrediction:
    sequence: int
    requested_mode: str
    source_monotonic: float
    completed_monotonic: float
    inference: CompetitionInference


def _node_label(namespace: str, name: str) -> str:
    namespace = namespace.rstrip("/")
    return f"{namespace}/{name}" if namespace else f"/{name}"


def _endpoint_participant_prefix(endpoint) -> bytes | None:
    try:
        gid = bytes(endpoint.endpoint_gid)
    except (AttributeError, TypeError, ValueError):
        return None
    return gid[:_DDS_GUID_PREFIX_SIZE] if len(gid) >= _DDS_GUID_PREFIX_SIZE else None


def _is_unnamed_endpoint(endpoint) -> bool:
    return (
        endpoint.node_name == _UNKNOWN_NODE_NAME
        and endpoint.node_namespace == _UNKNOWN_NODE_NAMESPACE
    )


def _is_paired_unnamed_relay(endpoint, subscriptions) -> bool:
    if not _is_unnamed_endpoint(endpoint):
        return False
    participant = _endpoint_participant_prefix(endpoint)
    return participant is not None and any(
        _is_unnamed_endpoint(subscription)
        and _endpoint_participant_prefix(subscription) == participant
        for subscription in subscriptions
    )


class CompetitionPolicyNode(Node):
    """Publish one final motor command behind an A-button motion gate."""

    def __init__(
        self,
        parameter_overrides: Sequence[Parameter] | None = None,
    ) -> None:
        super().__init__(
            "competition_policy",
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()
        self.bridge = CvBridge()
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._toggle = ToggleDriveGate()
        self._client = CompetitionIpcClient(
            artifact_dir=self.artifact_dir,
            socket_path=self.inference_socket_path,
            timeout_sec=self.inference_rpc_timeout_sec,
            required_device=self.inference_device,
        )
        self.artifact = self._client.artifact
        self._fsm = MissionStateMachine(
            self.artifact.mission,
            shortcut_only=self.run_mode == "shortcut_only",
        )
        self._latest_frame: tuple[int, np.ndarray, float] | None = None
        self._frame_sequence = 0
        self._prediction: _CompetitionPrediction | None = None
        self._last_decision_sequence = 0
        self._last_decision = MissionDecision(
            STOP_COMMAND,
            CompetitionMode.DISABLED,
            "startup",
        )
        self._last_published_command = STOP_COMMAND
        self._last_joy_monotonic: float | None = None
        self._joy_valid = False
        self._competitors: tuple[str, ...] = ()
        self._has_motor_subscriber = False
        self._next_graph_check_monotonic = 0.0
        self._stop_reason: str | None = None
        self._reset_all_requested = False
        self._reset_shortcut_requested = False
        self._reset_monotonic: float | None = None
        self._awaiting_post_reset_prediction = False
        self._shutdown_started = False
        self._worker_stop = False

        self.enabled_publisher = self.create_publisher(
            Bool,
            self.enabled_topic,
            10,
        )
        self.mode_publisher = self.create_publisher(String, self.mode_topic, 10)
        self.command_publisher = self.create_publisher(
            Float32MultiArray,
            self.command_topic,
            10,
        )
        self.signal_publisher = self.create_publisher(
            Float32MultiArray,
            self.signal_topic,
            10,
        )
        self.shortcut_publisher = self.create_publisher(
            Float32MultiArray,
            self.shortcut_topic,
            10,
        )
        self.fault_publisher = self.create_publisher(String, self.fault_topic, 10)
        self.motor_publisher = None
        self.joy_subscription = None
        if self.run_mode != "signal_shadow":
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
            self._refresh_graph(time.monotonic())
            self.publish_stop_burst()
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
        self._publish_enabled(False)
        self._publish_mode()
        self._worker = threading.Thread(
            target=self._inference_worker,
            name="competition-policy-inference",
            daemon=True,
        )
        self._worker.start()
        if self.run_mode == "signal_shadow":
            self.get_logger().warning(
                "Signal shadow mode has no motor publisher and never moves the car."
            )
        else:
            self.get_logger().warning(
                "Competition policy started DRIVE OFF. Release A, then press A "
                "to enable; press A again to stop."
            )

    def _declare_parameters(self) -> None:
        self.declare_parameter("artifact_dir", "")
        self.declare_parameter("run_mode", "combined")
        self.declare_parameter("camera_topic", "/image_raw")
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter("allowed_motor_relay_nodes", ["/ros_bridge"])
        self.declare_parameter("enabled_topic", "/competition_ai/enabled")
        self.declare_parameter("mode_topic", "/competition_ai/mode")
        self.declare_parameter(
            "command_topic", "/competition_ai/active_command"
        )
        self.declare_parameter(
            "signal_topic", "/competition_ai/signal_probabilities"
        )
        self.declare_parameter(
            "shortcut_topic", "/competition_ai/shortcut_state"
        )
        self.declare_parameter("fault_topic", "/competition_ai/fault")
        self.declare_parameter("a_button_index", 0)
        self.declare_parameter("allow_motion", False)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("joy_timeout_sec", 0.25)
        self.declare_parameter("inference_timeout_sec", 0.15)
        self.declare_parameter("graph_check_period_sec", 0.5)
        self.declare_parameter("stop_publish_count", 5)
        self.declare_parameter("inference_device", "cuda")
        self.declare_parameter(
            "inference_socket_path",
            "/run/user/1000/xycar-ai/competition.sock",
        )
        self.declare_parameter("inference_rpc_timeout_sec", 0.12)

    def _read_parameters(self) -> None:
        for name in (
            "artifact_dir",
            "run_mode",
            "camera_topic",
            "joy_topic",
            "motor_topic",
            "enabled_topic",
            "mode_topic",
            "command_topic",
            "signal_topic",
            "shortcut_topic",
            "fault_topic",
            "inference_device",
            "inference_socket_path",
        ):
            setattr(self, name, str(self.get_parameter(name).value))
        self.allowed_motor_relay_nodes = tuple(
            str(value)
            for value in self.get_parameter("allowed_motor_relay_nodes").value
        )
        self.a_button_index = int(self.get_parameter("a_button_index").value)
        self.allow_motion = bool(self.get_parameter("allow_motion").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.joy_timeout_sec = float(self.get_parameter("joy_timeout_sec").value)
        self.inference_timeout_sec = float(
            self.get_parameter("inference_timeout_sec").value
        )
        self.graph_check_period_sec = float(
            self.get_parameter("graph_check_period_sec").value
        )
        self.stop_publish_count = int(
            self.get_parameter("stop_publish_count").value
        )
        self.inference_rpc_timeout_sec = float(
            self.get_parameter("inference_rpc_timeout_sec").value
        )

    def _validate_parameters(self) -> None:
        if self.run_mode not in {"signal_shadow", "shortcut_only", "combined"}:
            raise ValueError("run_mode must be signal_shadow, shortcut_only, or combined")
        for name in (
            "artifact_dir",
            "camera_topic",
            "joy_topic",
            "motor_topic",
            "enabled_topic",
            "mode_topic",
            "command_topic",
            "signal_topic",
            "shortcut_topic",
            "fault_topic",
            "inference_socket_path",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.a_button_index < 0 or self.stop_publish_count < 1:
            raise ValueError("button index/count is invalid")
        for name in (
            "publish_rate_hz",
            "joy_timeout_sec",
            "inference_timeout_sec",
            "graph_check_period_sec",
            "inference_rpc_timeout_sec",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.inference_rpc_timeout_sec > self.inference_timeout_sec:
            raise ValueError("IPC timeout must not exceed inference timeout")
        if self.inference_device not in {"cpu", "cuda"}:
            raise ValueError("inference_device must be cpu or cuda")
        for node in self.allowed_motor_relay_nodes:
            if not node.startswith("/") or node.endswith("/"):
                raise ValueError("allowed relay names must be fully qualified")

    def _on_joy(self, message: Joy) -> None:
        now = time.monotonic()
        if len(message.buttons) <= self.a_button_index:
            with self._lock:
                self._joy_valid = False
                self._last_joy_monotonic = None
            self._force_off("Joy button array is too short for A")
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
                self._last_decision = self._fsm.enable(now)
                self._prediction = None
                self._reset_all_requested = True
                self._reset_monotonic = now
                self._awaiting_post_reset_prediction = True
                self._stop_reason = None
            elif action == ToggleAction.DISABLED:
                self._last_decision = self._fsm.disable()
                self._awaiting_post_reset_prediction = False
                self._reset_monotonic = None
        if action == ToggleAction.ENABLED:
            self._publish_enabled(True)
            self.get_logger().warning("Competition DRIVE toggled ON by A.")
        elif action == ToggleAction.DISABLED:
            self._publish_stop()
            self._publish_enabled(False)
            self._publish_mode()
            self.get_logger().warning("Competition DRIVE toggled OFF by A.")
        elif action == ToggleAction.REJECTED:
            self._publish_stop()
            self.get_logger().warning(
                "A rejected because motion prerequisites are not ready; "
                "release and press A again.",
                throttle_duration_sec=1.0,
            )

    def _on_camera(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
        except CvBridgeError as exc:
            self._force_off(f"camera image conversion failed: {exc}")
            return
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            self._force_off("camera image is not uint8 RGB")
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
                assert self._latest_frame is not None
                sequence, image, source_monotonic = self._latest_frame
                processed_sequence = sequence
                reset_all = self._reset_all_requested
                reset_shortcut = self._reset_shortcut_requested
                self._reset_all_requested = False
                self._reset_shortcut_requested = False
                if self.run_mode == "signal_shadow":
                    requested_mode = "signal_only"
                elif self._toggle.enabled:
                    requested_mode = self._fsm.inference_mode
                elif self.run_mode == "shortcut_only":
                    requested_mode = "shortcut"
                else:
                    requested_mode = "normal"
                previous_command = self._last_published_command
                reset_monotonic = self._reset_monotonic
            try:
                if reset_all:
                    self._client.reset_all()
                elif reset_shortcut:
                    self._client.reset_shortcut()
                if reset_monotonic is not None and source_monotonic <= reset_monotonic:
                    continue
                result = self._client.infer(
                    image,
                    mode=requested_mode,
                    previous_command=previous_command,
                )
            except Exception as exc:  # noqa: BLE001 - fail closed boundary
                self._force_off(f"competition inference failed: {exc}")
                continue
            prediction = _CompetitionPrediction(
                sequence=sequence,
                requested_mode=requested_mode,
                source_monotonic=source_monotonic,
                completed_monotonic=time.monotonic(),
                inference=result,
            )
            with self._lock:
                if self._toggle.enabled and requested_mode != self._fsm.inference_mode:
                    continue
                if (
                    self._awaiting_post_reset_prediction
                    and self._reset_monotonic is not None
                    and source_monotonic <= self._reset_monotonic
                ):
                    continue
                self._prediction = prediction
                if self._awaiting_post_reset_prediction:
                    self._awaiting_post_reset_prediction = False
                    self._reset_monotonic = None
            self._publish_inference(result)

    def _on_control_timer(self) -> None:
        now = time.monotonic()
        if self.run_mode == "signal_shadow":
            return
        if now >= self._next_graph_check_monotonic:
            self._refresh_graph(now)
        completed_reason: str | None = None
        with self._lock:
            reason = self._unsafe_reason_locked(now)
            if reason is None and self._toggle.enabled:
                prediction = self._prediction
                if prediction is not None:
                    if prediction.sequence != self._last_decision_sequence:
                        try:
                            self._last_decision = self._fsm.update(
                                prediction.inference,
                                now_monotonic=now,
                            )
                        except Exception as exc:  # noqa: BLE001
                            reason = f"mission FSM failed: {exc}"
                        else:
                            self._last_decision_sequence = prediction.sequence
                            if self._last_decision.reset_shortcut:
                                self._reset_shortcut_requested = True
                            self._publish_mode()
                    if reason is None:
                        if self._last_decision.mode == CompetitionMode.FAULT:
                            reason = self._last_decision.reason
                        elif self._last_decision.mode == CompetitionMode.DISABLED:
                            if self._last_decision.completed:
                                completed_reason = self._last_decision.reason
                                self._toggle.fault()
                                self._awaiting_post_reset_prediction = False
                                self._reset_monotonic = None
                            else:
                                reason = self._last_decision.reason
                        else:
                            self._publish(self._last_decision.command)
                            return
        if completed_reason is not None:
            self._publish_stop()
            self._publish_enabled(False)
            self._publish_mode()
            self.get_logger().warning(
                f"Competition motion completed: {completed_reason}; "
                "release A before the next enable."
            )
            return
        if reason is not None:
            self._force_off(reason)
            return
        self._publish_stop()

    def _unsafe_reason_locked(self, now: float) -> str | None:
        if self._competitors:
            return "competing motor publisher(s): " + ", ".join(self._competitors)
        if not self._has_motor_subscriber:
            return "no motor subscriber"
        if not self._joy_valid or not is_fresh(
            now,
            self._last_joy_monotonic,
            self.joy_timeout_sec,
        ):
            return "Joy input is missing or stale"
        if self._awaiting_post_reset_prediction:
            if (
                self._reset_monotonic is None
                or now - self._reset_monotonic > self.inference_timeout_sec
            ):
                return "post-reset inference is missing or stale"
            return None
        if self._prediction is None or not is_fresh(
            now,
            self._prediction.source_monotonic,
            self.inference_timeout_sec,
        ):
            return "camera inference is missing or stale"
        return None

    def _can_enable_locked(self, now: float) -> bool:
        return self.allow_motion and self._unsafe_reason_locked(now) is None

    def _refresh_graph(self, now: float) -> None:
        if self.motor_publisher is None:
            return
        topic = self.resolve_topic_name(self.motor_topic)
        subscriptions = self.get_subscriptions_info_by_topic(topic)
        allow_unnamed_bridge = "/ros_bridge" in self.allowed_motor_relay_nodes
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
            if allow_unnamed_bridge and _is_paired_unnamed_relay(
                publisher,
                subscriptions,
            ):
                continue
            competitors.append(label)
        with self._lock:
            self._competitors = tuple(sorted(set(competitors)))
            self._has_motor_subscriber = bool(subscriptions)
            self._next_graph_check_monotonic = now + self.graph_check_period_sec

    def _force_off(self, reason: str) -> None:
        if self.run_mode == "signal_shadow":
            message = String()
            message.data = reason
            self.fault_publisher.publish(message)
            return
        with self._lock:
            was_enabled = self._toggle.fault()
            changed = reason != self._stop_reason
            self._stop_reason = reason
            self._last_decision = self._fsm.fault(reason)
            self._awaiting_post_reset_prediction = False
            self._reset_monotonic = None
        self._publish_stop()
        self._publish_mode()
        if was_enabled:
            self._publish_enabled(False)
        if changed:
            message = String()
            message.data = reason
            self.fault_publisher.publish(message)
            self.get_logger().warning(f"Competition motion stopped: {reason}")

    def _publish(self, command: DriveCommand) -> None:
        if self.motor_publisher is None:
            return
        message = Float32MultiArray()
        message.data = [float(command.angle), float(command.speed)]
        self.motor_publisher.publish(message)
        self._last_published_command = command
        diagnostic = Float32MultiArray()
        diagnostic.data = list(message.data)
        self.command_publisher.publish(diagnostic)

    def _publish_stop(self) -> None:
        self._publish(STOP_COMMAND)

    def publish_stop_burst(self) -> None:
        for _ in range(self.stop_publish_count):
            self._publish_stop()

    def _publish_enabled(self, enabled: bool) -> None:
        message = Bool()
        message.data = enabled
        self.enabled_publisher.publish(message)

    def _publish_mode(self) -> None:
        message = String()
        message.data = self._fsm.mode.value
        self.mode_publisher.publish(message)

    def _publish_inference(self, inference: CompetitionInference) -> None:
        if inference.signal is not None:
            signal_message = Float32MultiArray()
            signal_message.data = [
                inference.signal.approach,
                inference.signal.visible,
                inference.signal.readable,
                inference.signal.red,
                inference.signal.yellow,
                inference.signal.left,
                inference.signal.green,
                inference.signal.progress,
            ]
            self.signal_publisher.publish(signal_message)
        if inference.shortcut is not None:
            shortcut_message = Float32MultiArray()
            shortcut_message.data = [
                float(inference.shortcut.phase),
                inference.shortcut.handoff_probability,
            ]
            self.shortcut_publisher.publish(shortcut_message)

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        with self._frame_condition:
            self._worker_stop = True
            self._toggle.fault()
            self._frame_condition.notify_all()
        if self.run_mode != "signal_shadow":
            self.publish_stop_burst()
        self._publish_enabled(False)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._client.close()
        if self.run_mode != "signal_shadow":
            self.publish_stop_burst()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: CompetitionPolicyNode | None = None
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
        node = CompetitionPolicyNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
