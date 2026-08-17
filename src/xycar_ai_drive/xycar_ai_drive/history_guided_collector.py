# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Camera-clocked guided collection with native motor execution history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import signal
import time

import numpy as np
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from xycar_data.session_writer import AsyncSessionWriter, CameraSample
from xycar_msgs.msg import XycarMotor

from xycar_ai_drive.control import (
    STOP_COMMAND,
    DriveCommand,
    ToggleAction,
    command_class_ids,
)
from xycar_ai_drive.guided_policy_collector import (
    FusedCommand,
    GuideInput,
    GuidedPrediction,
    fuse_guided_command,
    trigger_depth,
)
from xycar_ai_drive.history_policy_node import HistoryPolicyNode, _Frame


@dataclass(frozen=True)
class _PendingGuidedExecution:
    frame: _Frame
    requested: DriveCommand
    sent_monotonic: float
    prediction: GuidedPrediction
    guide: GuideInput
    fused: FusedCommand
    history: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _PendingFinish:
    token: int
    reason: str
    complete: bool
    discarded: int
    samples: tuple[CameraSample, ...]


class HistoryGuidedCollectorNode(HistoryPolicyNode):
    """Fuse the current model output and human correction once per frame."""

    def __init__(self) -> None:
        self._executed_history_commands: deque[tuple[float, float]] = deque(
            [(0.0, 0.0)] * 4,
            maxlen=4,
        )
        super().__init__()
        self._guide = GuideInput()
        self._last_buttons: list[bool] = []
        self._last_executed_command = STOP_COMMAND
        self._session_token: int | None = None
        self._finishing_token: int | None = None
        self._pending_finish: _PendingFinish | None = None
        self._recording_tail: deque[CameraSample] = deque()
        self._recording_disabled = False
        self._writer_failure_handled = False
        self.collection_profile_metadata = _profile_metadata(
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
        self.get_logger().warning(
            'History guided collector requires Y DRIVE ON and records only '
            'commands echoed by the native motor gateway.'
        )

    def _declare_parameters(self) -> None:
        super()._declare_parameters()
        self.declare_parameter('steering_axis', 0)
        self.declare_parameter('lt_axis', 4)
        self.declare_parameter('rt_axis', 5)
        self.declare_parameter('trigger_axis_mode', 'negative')
        self.declare_parameter('invert_steering', True)
        self.declare_parameter('max_steering_angle', 100.0)
        self.declare_parameter('rt_speed_increment', 5.0)
        self.declare_parameter('lt_speed_decrement', 5.0)
        self.declare_parameter('speed_cap', 30.0)
        self.declare_parameter('correction_deadzone', 0.05)
        self.declare_parameter('record_start_button', 0)
        self.declare_parameter('record_stop_button', 1)
        self.declare_parameter('record_discard_button', 2)
        self.declare_parameter('drive_toggle_button', 3)
        self.declare_parameter('collection_profile_path', '')
        self.declare_parameter(
            'recording_root_dir',
            '/home/xytron/xycar_data/history_guided',
        )
        self.declare_parameter('tail_discard_frames', 10)
        self.declare_parameter('recording_image_format', 'jpeg')
        self.declare_parameter('recording_jpeg_quality', 95)
        self.declare_parameter('recording_png_compression', 3)
        self.declare_parameter('recording_queue_size', 128)
        self.declare_parameter('recording_min_free_space_mb', 1024)
        self.declare_parameter('curriculum_generation', 0)

    def _read_parameters(self) -> None:
        super()._read_parameters()
        for name in (
            'steering_axis',
            'lt_axis',
            'rt_axis',
            'record_start_button',
            'record_stop_button',
            'record_discard_button',
            'drive_toggle_button',
            'tail_discard_frames',
            'recording_jpeg_quality',
            'recording_png_compression',
            'recording_queue_size',
            'recording_min_free_space_mb',
            'curriculum_generation',
        ):
            setattr(self, name, int(self.get_parameter(name).value))
        for name in (
            'max_steering_angle',
            'rt_speed_increment',
            'lt_speed_decrement',
            'speed_cap',
            'correction_deadzone',
        ):
            setattr(self, name, float(self.get_parameter(name).value))
        for name in (
            'trigger_axis_mode',
            'collection_profile_path',
            'recording_root_dir',
            'recording_image_format',
        ):
            setattr(self, name, str(self.get_parameter(name).value))
        self.invert_steering = bool(
            self.get_parameter('invert_steering').value
        )
        self.recording_image_format = self.recording_image_format.lower()
        self.a_button_index = self.drive_toggle_button

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        indices = (
            self.steering_axis,
            self.lt_axis,
            self.rt_axis,
            self.record_start_button,
            self.record_stop_button,
            self.record_discard_button,
            self.drive_toggle_button,
        )
        if any(index < 0 for index in indices):
            raise ValueError('guided axis and button indices must be non-negative')
        if len(
            {
                self.record_start_button,
                self.record_stop_button,
                self.record_discard_button,
                self.drive_toggle_button,
            }
        ) != 4:
            raise ValueError('guided button indices must be distinct')
        if self.trigger_axis_mode not in {'negative', 'positive', 'signed'}:
            raise ValueError('unsupported trigger axis mode')
        if not 0.0 < self.max_steering_angle <= 100.0:
            raise ValueError('max_steering_angle must be in (0,100]')
        if self.speed_cap != 30.0:
            raise ValueError('history guided speed_cap must be the fixed ceiling 30')
        if self.rt_speed_increment < 0.0 or self.lt_speed_decrement < 0.0:
            raise ValueError('guided speed corrections must be non-negative')
        if not 0.0 <= self.correction_deadzone < 1.0:
            raise ValueError('correction_deadzone must be in [0,1)')
        if self.curriculum_generation < 0 or self.tail_discard_frames < 0:
            raise ValueError('generation and tail discard must be non-negative')
        if self.recording_image_format not in {'jpeg', 'png'}:
            raise ValueError('recording image format must be jpeg or png')
        if self.recording_queue_size < 1 or self.recording_min_free_space_mb < 0:
            raise ValueError('recording queue or free-space setting is invalid')
        path = Path(self.collection_profile_path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('collection_profile_path must be an existing absolute file')

    def _on_joy(self, message: Joy) -> None:
        now = time.monotonic()
        required_axis = max(self.steering_axis, self.lt_axis, self.rt_axis)
        required_button = max(
            self.record_start_button,
            self.record_stop_button,
            self.record_discard_button,
            self.drive_toggle_button,
        )
        if len(message.axes) <= required_axis or len(message.buttons) <= required_button:
            self._force_off('Joy axis or button array is too short')
            return
        try:
            steering = float(message.axes[self.steering_axis])
            if not math.isfinite(steering):
                raise ValueError('steering axis is not finite')
            guide = GuideInput(
                steering_axis=max(-1.0, min(1.0, steering)),
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
        buttons = [bool(value) for value in message.buttons]
        with self._condition:
            previous = self._last_buttons

            def rising(index: int) -> bool:
                return buttons[index] and (
                    len(previous) <= index or not previous[index]
                )

            self._last_buttons = buttons
            self._guide = guide
            self._joy_valid = True
            self._last_joy_monotonic = now
            action = self._toggle.observe(
                pressed=buttons[self.drive_toggle_button],
                can_enable=self._can_enable_locked(now),
            )
            start_pressed = rising(self.record_start_button)
            stop_pressed = rising(self.record_stop_button)
            discard_pressed = rising(self.record_discard_button)
            if action == ToggleAction.ENABLED:
                self._reset_history_locked()
                self._stop_reason = None
            elif action == ToggleAction.DISABLED:
                self._reset_history_locked()
        if action == ToggleAction.ENABLED:
            self._publish_enabled(True)
            self.get_logger().warning('History guided motion toggled ON by Y.')
        elif action == ToggleAction.DISABLED:
            self._publish_stop()
            self._publish_enabled(False)
            self._finish_session('y_drive_off', discard_tail=True, complete=True)
        elif action == ToggleAction.REJECTED:
            self._force_off('Y toggle rejected because prerequisites are not ready')
        if start_pressed:
            self._start_session()
        if stop_pressed:
            self._finish_session('b_button', discard_tail=False, complete=True)
        if discard_pressed:
            self._finish_session('x_button', discard_tail=True, complete=True)

    def _can_enable_locked(self, now: float) -> bool:
        neutral = (
            abs(self._guide.steering_axis) <= self.correction_deadzone
            and self._guide.lt_depth <= self.correction_deadzone
            and self._guide.rt_depth <= self.correction_deadzone
        )
        return self.allow_motion and neutral and self._unsafe_reason_locked(now) is None

    def _reset_history_locked(self) -> None:
        super()._reset_history_locked()
        self._executed_history_commands = deque(
            [(0.0, 0.0)] * 4,
            maxlen=4,
        )

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
                history_ids = tuple(self._history)
                history_commands = tuple(self._executed_history_commands)
            try:
                result = self._policy.infer(frame.image_rgb, history_ids)
            except Exception as exc:  # noqa: BLE001
                self._force_off(f'guided policy inference failed: {exc}')
                continue
            completed = time.monotonic()
            model = result.command
            if not all(
                math.isfinite(value)
                for value in (model.angle, model.speed, result.inference_ms)
            ):
                self._force_off('guided policy inference returned NaN or Inf')
                continue
            prediction = GuidedPrediction(
                sequence=frame.sequence,
                command=model,
                source_monotonic=frame.received_monotonic,
                completed_monotonic=completed,
                inference_ms=float(result.inference_ms),
                image_bgr=np.ascontiguousarray(frame.image_rgb[:, :, ::-1]),
                stamp_sec=frame.stamp_sec,
                stamp_nanosec=frame.stamp_nanosec,
                received_wall_time_ns=frame.received_wall_time_ns,
            )
            prediction_message = Float32MultiArray()
            prediction_message.data = [
                float(model.angle),
                float(model.speed),
                float(result.inference_ms),
            ]
            self.prediction_publisher.publish(prediction_message)
            with self._condition:
                self._last_inference_source_monotonic = frame.received_monotonic
                self._inference_times.append(completed)
                self._inference_latencies_ms.append(float(result.inference_ms))
                self._image_to_command_ms.append(
                    (completed - frame.received_monotonic) * 1000.0
                )
                reason = self._unsafe_reason_locked(completed)
                guide = self._guide
                if reason is None and self._toggle.enabled:
                    fused = fuse_guided_command(
                        model,
                        guide,
                        max_steering_angle=self.max_steering_angle,
                        invert_steering=self.invert_steering,
                        rt_speed_increment=self.rt_speed_increment,
                        lt_speed_decrement=self.lt_speed_decrement,
                        speed_cap=self.speed_cap,
                        correction_deadzone=self.correction_deadzone,
                    )
                else:
                    fused = FusedCommand(STOP_COMMAND, 0.0, 0.0, False)
                self._send_guided_locked(
                    frame,
                    prediction,
                    guide,
                    fused,
                    history_commands,
                    now=completed,
                )

    def _send_guided_locked(
        self,
        frame: _Frame,
        prediction: GuidedPrediction,
        guide: GuideInput,
        fused: FusedCommand,
        history: tuple[tuple[float, float], ...],
        *,
        now: float,
    ) -> None:
        message = XycarMotor()
        message.header.stamp.sec = frame.stamp_sec
        message.header.stamp.nanosec = frame.stamp_nanosec
        message.header.frame_id = 'history_guided_camera'
        message.angle = float(fused.executed.angle)
        message.speed = float(fused.executed.speed)
        self._pending = _PendingGuidedExecution(
            frame=frame,
            requested=fused.executed,
            sent_monotonic=now,
            prediction=prediction,
            guide=guide,
            fused=fused,
            history=history,
        )
        self.motor_publisher.publish(message)
        self._command_times.append(now)

    def _on_executed(self, message: XycarMotor) -> None:
        stamp = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        with self._condition:
            if stamp in self._ignored_execution_stamps:
                self._ignored_execution_stamps.remove(stamp)
                return
            pending = self._pending
            if not isinstance(pending, _PendingGuidedExecution):
                if message.header.frame_id == 'native_motor_watchdog':
                    self._reset_history_locked()
                    self._toggle.fault()
                    self._publish_enabled(False)
                    return
                if stamp == self._last_execution_stamp:
                    self._duplicate_executions += 1
                    mismatch = 'duplicate motor execution echo'
                else:
                    self._out_of_order_executions += 1
                    mismatch = 'unexpected motor execution echo'
            else:
                mismatch = None
            expected = (
                (pending.frame.stamp_sec, pending.frame.stamp_nanosec)
                if isinstance(pending, _PendingGuidedExecution)
                else None
            )
            if expected is not None and stamp != expected:
                self._out_of_order_executions += 1
                self._ignored_execution_stamps.add(expected)
                self._pending = None
                self._condition.notify_all()
                mismatch = f'motor execution stamp mismatch: {stamp} != {expected}'
            elif isinstance(pending, _PendingGuidedExecution):
                actual = DriveCommand(float(message.angle), float(message.speed))
                if not all(math.isfinite(value) for value in (actual.angle, actual.speed)):
                    self._pending = None
                    self._condition.notify_all()
                    mismatch = 'motor execution contains NaN or Inf'
                else:
                    now = time.monotonic()
                    if self._toggle.enabled:
                        self._history.append(command_class_ids(actual))
                        self._executed_history_commands.append(
                            (actual.angle, actual.speed)
                        )
                    else:
                        self._reset_history_locked()
                    self._last_executed_command = actual
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
                    mismatch = None
        if mismatch is not None:
            self._force_off(mismatch)
            return
        if self._session_token is not None and actual.speed > 0.0:
            self._record_prediction(pending, actual)

    def _start_session(self) -> None:
        if self._recording_disabled or self._session_token is not None:
            return
        if not self._toggle.enabled or self._last_executed_command.speed <= 0.0:
            self.get_logger().warning(
                'A ignored: enable Y and wait for positive executed speed first.'
            )
            return
        assert self.artifact is not None
        metadata = {
            'dataset_kind': 'camera_first_teleop_behavior_cloning',
            'camera_is_primary': True,
            'lidar_is_optional': False,
            'control_mode': 'history_guided_policy',
            'sample_clock': 'camera_frame',
            'motor_transport': 'ros2_native',
            'curriculum': {
                'generation': self.curriculum_generation,
                'parent_artifact_id': self.artifact.artifact_id,
                'parent_artifact_sha256s_digest': _sha256_file(
                    Path(self.artifact_dir) / 'SHA256SUMS'
                ),
            },
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
            'guided_control': {
                'max_steering_angle': self.max_steering_angle,
                'invert_steering': self.invert_steering,
                'rt_speed_increment': self.rt_speed_increment,
                'lt_speed_decrement': self.lt_speed_decrement,
                'speed_cap': self.speed_cap,
                'correction_deadzone': self.correction_deadzone,
            },
            'collection_profile': dict(self.collection_profile_metadata),
            'runtime_safety': {
                'watchdog_rate_hz': self.watchdog_rate_hz,
                'joy_timeout_sec': self.joy_timeout_sec,
                'inference_timeout_sec': self.inference_timeout_sec,
                'execution_timeout_sec': self.execution_timeout_sec,
            },
            'recording': {
                'root_dir': self.recording_root_dir,
                'tail_discard_frames': self.tail_discard_frames,
                'image_format': self.recording_image_format,
                'jpeg_quality': self.recording_jpeg_quality,
                'queue_size': self.recording_queue_size,
            },
        }
        token = self.writer.start_session(metadata)
        if token is None:
            self._recording_failure('could not start history guided session')
            return
        self._session_token = token
        self._recording_tail.clear()
        self.get_logger().info('History guided recording started.')

    def _record_prediction(
        self,
        pending: _PendingGuidedExecution,
        actual: DriveCommand,
    ) -> None:
        prediction = pending.prediction
        fused = pending.fused
        sample = CameraSample(
            image=prediction.image_bgr.copy(),
            camera_sequence=prediction.sequence,
            camera_stamp_sec=prediction.stamp_sec,
            camera_stamp_nanosec=prediction.stamp_nanosec,
            camera_received_monotonic=prediction.source_monotonic,
            camera_received_wall_time_ns=prediction.received_wall_time_ns,
            angle=actual.angle,
            speed=actual.speed,
            input_key='history_guided_policy',
            lidar=None,
            lidar_skew_sec=None,
            model_angle=prediction.command.angle,
            model_speed=prediction.command.speed,
            steering_axis=pending.guide.steering_axis,
            steering_residual=actual.angle - prediction.command.angle,
            lt_depth=pending.guide.lt_depth,
            rt_depth=pending.guide.rt_depth,
            speed_delta=actual.speed - prediction.command.speed,
            human_correction=fused.human_correction,
            inference_ms=prediction.inference_ms,
            history_commands=pending.history,
            motor_executed_received_wall_time_ns=time.time_ns(),
        )
        self._recording_tail.append(sample)
        self._flush_recording_prefix()
        if len(self._recording_tail) > self.recording_queue_size + self.tail_discard_frames:
            self._recording_failure('history guided recording backlog exceeded')

    def _flush_recording_prefix(self) -> None:
        if self._session_token is None:
            return
        while len(self._recording_tail) > self.tail_discard_frames:
            if not self.writer.submit(self._session_token, self._recording_tail[0]):
                return
            self._recording_tail.popleft()

    def _finish_session(self, reason: str, *, discard_tail: bool, complete: bool) -> None:
        token = self._session_token
        if token is None:
            return
        buffered = tuple(self._recording_tail)
        self._recording_tail.clear()
        discarded = min(self.tail_discard_frames, len(buffered)) if discard_tail else 0
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
            extra_metadata={'emergency_discard_count': pending.discarded},
            final_samples=pending.samples,
        ):
            self._pending_finish = None

    def _poll_writer_results(self) -> None:
        for result in self.writer.poll_results():
            if result.token == self._finishing_token:
                self._finishing_token = None
                self.get_logger().info(
                    f'History guided session completed={result.completed} '
                    f'path={result.path} samples={result.sample_count}'
                )

    def _recording_failure(self, reason: str) -> None:
        if not self._recording_disabled:
            self._recording_disabled = True
            self._finish_session(reason, discard_tail=False, complete=False)
        self._force_off(reason)

    def _force_off(self, reason: str) -> None:
        super()._force_off(reason)
        if hasattr(self, '_session_token') and self._session_token is not None:
            self._finish_session(reason, discard_tail=False, complete=False)

    def _on_safety_timer(self) -> None:
        if hasattr(self, 'writer'):
            if self.writer.failure is not None and not self._writer_failure_handled:
                self._writer_failure_handled = True
                self._recording_failure(self.writer.failure)
            self._poll_writer_results()
            self._retry_finish()
            self._flush_recording_prefix()
        super()._on_safety_timer()

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._finish_session('shutdown', discard_tail=False, complete=False)
        super().shutdown()
        deadline = time.monotonic() + 1.0
        while self._pending_finish is not None and time.monotonic() < deadline:
            self._retry_finish()
            time.sleep(0.01)
        self.writer.shutdown()
        self._poll_writer_results()


def _profile_metadata(path_value: str) -> dict[str, object]:
    path = Path(path_value)
    return {'path': path_value, 'sha256': _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(args=None) -> None:
    import rclpy
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: HistoryGuidedCollectorNode | None = None
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
        node = HistoryGuidedCollectorNode()
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
