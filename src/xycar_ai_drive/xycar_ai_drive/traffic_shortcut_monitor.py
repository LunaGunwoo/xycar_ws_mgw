"""Passive Tk dashboard for traffic classification and drive commands."""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from tkinter import ttk

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PilImage
from PIL import ImageTk
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String

from xycar_ai_drive.traffic_shortcut_artifact import (
    TrafficShortcutBundle,
    load_traffic_shortcut_bundle,
)
from xycar_ai_drive.traffic_shortcut_diagnostics import (
    SignalDebugContractError,
    SignalDebugSnapshot,
    decode_signal_debug,
)


@dataclass(frozen=True)
class DriveVectorSample:
    """One received model prediction or emitted motor command."""

    angle: float
    speed: float
    received_monotonic: float
    inference_ms: float | None = None
    is_shortcut: bool | None = None


@dataclass(frozen=True)
class MonitorFrame:
    """One BGR camera frame retained for exact timestamp matching."""

    stamp_key: tuple[int, int] | None
    image: np.ndarray
    received_monotonic: float


@dataclass(frozen=True)
class TrafficShortcutMonitorSnapshot:
    signal: SignalDebugSnapshot | None
    signal_received_monotonic: float | None
    matched_frame: MonitorFrame | None
    latest_frame: MonitorFrame | None
    prediction: DriveVectorSample | None
    actual: DriveVectorSample | None
    enabled: bool
    enabled_received_monotonic: float | None
    camera_error: str | None
    signal_error: str | None
    control_error: str | None


class TrafficShortcutMonitorNode(Node):
    """Read-only subscriber node; it never publishes or runs inference."""

    def __init__(
        self,
        parameter_overrides: Sequence[Parameter] | None = None,
        *,
        bundle_loader=load_traffic_shortcut_bundle,
    ) -> None:
        super().__init__(
            'traffic_shortcut_monitor',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()
        self.bundle: TrafficShortcutBundle = bundle_loader(self.bundle_dir)
        self._validate_bundle()
        self.class_labels = tuple(self.bundle.detector.classifier_classes)
        self.speed_scale = float(self.bundle.base_speed_cap)
        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._frames: OrderedDict[tuple[int, int], MonitorFrame] = (
            OrderedDict()
        )
        self._latest_frame: MonitorFrame | None = None
        self._signal: SignalDebugSnapshot | None = None
        self._signal_received_monotonic: float | None = None
        self._prediction: DriveVectorSample | None = None
        self._actual: DriveVectorSample | None = None
        self._enabled = False
        self._enabled_received_monotonic: float | None = None
        self._camera_error: str | None = None
        self._signal_error: str | None = None
        self._control_error: str | None = None

        self.camera_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self._on_camera,
            qos_profile_sensor_data,
        )
        self.signal_subscription = self.create_subscription(
            String,
            self.signal_debug_topic,
            self._on_signal_debug,
            10,
        )
        self.prediction_subscription = self.create_subscription(
            Float32MultiArray,
            self.prediction_topic,
            self._on_prediction,
            10,
        )
        self.motor_subscription = self.create_subscription(
            Float32MultiArray,
            self.motor_topic,
            self._on_motor,
            10,
        )
        self.enabled_subscription = self.create_subscription(
            Bool,
            self.enabled_topic,
            self._on_enabled,
            10,
        )
        self.get_logger().warning(
            'Passive traffic shortcut monitor started. Closing this window '
            'stops the integrated mission launch.'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('bundle_dir', '')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter(
            'signal_debug_topic',
            '/traffic_shortcut/signal_debug',
        )
        self.declare_parameter(
            'prediction_topic',
            '/traffic_shortcut/prediction',
        )
        self.declare_parameter('motor_topic', '/xycar_motor')
        self.declare_parameter('enabled_topic', '/traffic_shortcut/enabled')
        self.declare_parameter('camera_stale_sec', 1.0)
        self.declare_parameter('signal_stale_sec', 1.0)
        self.declare_parameter('prediction_stale_sec', 0.5)
        self.declare_parameter('motor_stale_sec', 0.5)
        self.declare_parameter('frame_buffer_size', 30)

    def _read_parameters(self) -> None:
        self.bundle_dir = str(self.get_parameter('bundle_dir').value).strip()
        self.camera_topic = str(
            self.get_parameter('camera_topic').value
        ).strip()
        self.signal_debug_topic = str(
            self.get_parameter('signal_debug_topic').value
        ).strip()
        self.prediction_topic = str(
            self.get_parameter('prediction_topic').value
        ).strip()
        self.motor_topic = str(
            self.get_parameter('motor_topic').value
        ).strip()
        self.enabled_topic = str(
            self.get_parameter('enabled_topic').value
        ).strip()
        self.camera_stale_sec = float(
            self.get_parameter('camera_stale_sec').value
        )
        self.signal_stale_sec = float(
            self.get_parameter('signal_stale_sec').value
        )
        self.prediction_stale_sec = float(
            self.get_parameter('prediction_stale_sec').value
        )
        self.motor_stale_sec = float(
            self.get_parameter('motor_stale_sec').value
        )
        self.frame_buffer_size = int(
            self.get_parameter('frame_buffer_size').value
        )

    def _validate_parameters(self) -> None:
        for label, value in (
            ('bundle_dir', self.bundle_dir),
            ('camera_topic', self.camera_topic),
            ('signal_debug_topic', self.signal_debug_topic),
            ('prediction_topic', self.prediction_topic),
            ('motor_topic', self.motor_topic),
            ('enabled_topic', self.enabled_topic),
        ):
            if not value:
                raise ValueError(f'{label} must not be empty')
        for label, value in (
            ('camera_stale_sec', self.camera_stale_sec),
            ('signal_stale_sec', self.signal_stale_sec),
            ('prediction_stale_sec', self.prediction_stale_sec),
            ('motor_stale_sec', self.motor_stale_sec),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{label} must be finite and positive')
        if self.frame_buffer_size < 1:
            raise ValueError('frame_buffer_size must be positive')

    def _validate_bundle(self) -> None:
        if self.bundle.detector.mode != 'yolo_cnn_classifier':
            raise ValueError(
                'traffic shortcut monitor requires YOLO/CNN bundle'
            )
        if len(self.bundle.detector.classifier_classes) != 3:
            raise ValueError(
                'traffic shortcut monitor requires three CNN classes'
            )
        if not math.isfinite(self.bundle.base_speed_cap) or (
            self.bundle.base_speed_cap <= 0.0
        ):
            raise ValueError('traffic shortcut Base speed cap is invalid')

    def _on_camera(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8',
            )
            if (
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[2] != 3
            ):
                raise ValueError('converted camera frame is not uint8 BGR')
            frame = np.ascontiguousarray(frame.copy())
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            with self._lock:
                self._camera_error = f'camera conversion failed: {exc}'
            return
        stamp = message.header.stamp
        stamp_key = frame_stamp_key(int(stamp.sec), int(stamp.nanosec))
        received = time.monotonic()
        retained = MonitorFrame(
            stamp_key=stamp_key,
            image=frame,
            received_monotonic=received,
        )
        with self._lock:
            self._latest_frame = retained
            self._camera_error = None
            if stamp_key is not None:
                self._frames[stamp_key] = retained
                self._frames.move_to_end(stamp_key)
                while len(self._frames) > self.frame_buffer_size:
                    self._frames.popitem(last=False)

    def _on_signal_debug(self, message: String) -> None:
        try:
            signal = decode_signal_debug(message.data)
            if signal.bundle_id != self.bundle.artifact_id:
                raise SignalDebugContractError(
                    'signal-debug bundle does not match monitor bundle'
                )
            if signal.class_labels != self.class_labels:
                raise SignalDebugContractError(
                    'signal-debug CNN classes do not match monitor bundle'
                )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            with self._lock:
                self._signal_error = f'signal diagnostics rejected: {exc}'
            return
        with self._lock:
            self._signal = signal
            self._signal_received_monotonic = time.monotonic()
            self._signal_error = None

    def _on_prediction(self, message: Float32MultiArray) -> None:
        try:
            values = _finite_message_values(message, expected_length=4)
            angle, speed, inference_ms, shortcut_flag = values
            _validate_drive_values(angle, speed)
            if inference_ms < 0.0 or shortcut_flag not in {0.0, 1.0}:
                raise ValueError('prediction metadata is invalid')
            sample = DriveVectorSample(
                angle=angle,
                speed=speed,
                inference_ms=inference_ms,
                is_shortcut=bool(shortcut_flag),
                received_monotonic=time.monotonic(),
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            with self._lock:
                self._control_error = f'prediction rejected: {exc}'
            return
        with self._lock:
            self._prediction = sample
            self._control_error = None

    def _on_motor(self, message: Float32MultiArray) -> None:
        try:
            angle, speed = _finite_message_values(
                message,
                expected_length=2,
            )
            _validate_drive_values(angle, speed)
            sample = DriveVectorSample(
                angle=angle,
                speed=speed,
                received_monotonic=time.monotonic(),
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            with self._lock:
                self._control_error = f'motor command rejected: {exc}'
            return
        with self._lock:
            self._actual = sample
            self._control_error = None

    def _on_enabled(self, message: Bool) -> None:
        with self._lock:
            self._enabled = bool(message.data)
            self._enabled_received_monotonic = time.monotonic()

    def snapshot(self) -> TrafficShortcutMonitorSnapshot:
        with self._lock:
            signal = self._signal
            matched_frame = (
                None
                if signal is None or signal.stamp_key is None
                else self._frames.get(signal.stamp_key)
            )
            return TrafficShortcutMonitorSnapshot(
                signal=signal,
                signal_received_monotonic=(
                    self._signal_received_monotonic
                ),
                matched_frame=matched_frame,
                latest_frame=self._latest_frame,
                prediction=self._prediction,
                actual=self._actual,
                enabled=self._enabled,
                enabled_received_monotonic=(
                    self._enabled_received_monotonic
                ),
                camera_error=self._camera_error,
                signal_error=self._signal_error,
                control_error=self._control_error,
            )


class TrafficShortcutMonitorApplication:
    """One-window visualization of signal and drive diagnostics."""

    def __init__(self, node: TrafficShortcutMonitorNode) -> None:
        self.node = node
        self.closed = False
        self.root = tk.Tk()
        self.root.title('Xycar traffic shortcut live monitor (PASSIVE)')
        self.root.geometry('1500x900')
        self.root.minsize(1220, 760)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.report_callback_exception = self._on_gui_exception
        self.root.bind('<Escape>', lambda _event: self.close())
        self.root.bind('q', lambda _event: self.close())
        self._frame_photo: ImageTk.PhotoImage | None = None
        self._crop_photo: ImageTk.PhotoImage | None = None
        self._render_signature: tuple[object, ...] | None = None
        self._probability_vars = {
            label: tk.DoubleVar(value=0.0)
            for label in node.class_labels
        }
        self._probability_text = {
            label: tk.StringVar(value='0.0%')
            for label in node.class_labels
        }
        self._signal_class_text = tk.StringVar(value='WAITING')
        self._signal_status_text = tk.StringVar(
            value='Waiting for policy diagnostics'
        )
        self._signal_detail_text = tk.StringVar(value='')
        self._drive_status_text = tk.StringVar(value='DRIVE OFF')
        self._prediction_text = tk.StringVar(value='Prediction: waiting')
        self._actual_text = tk.StringVar(value='Motor output: waiting')
        self._warning_text = tk.StringVar(value='')
        self._build_layout()
        self.root.after(20, self._schedule_update)

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        preview = ttk.Frame(outer)
        preview.grid(row=0, column=0, sticky='nsew')
        preview.rowconfigure(1, weight=4)
        preview.rowconfigure(3, weight=1)
        preview.columnconfigure(0, weight=1)
        ttk.Label(
            preview,
            text='Camera frame used by the policy signal result',
        ).grid(row=0, column=0, sticky='w')
        self.frame_label = ttk.Label(preview, anchor=tk.CENTER)
        self.frame_label.grid(
            row=1,
            column=0,
            sticky='nsew',
            pady=(4, 10),
        )
        ttk.Label(preview, text='Padded CNN classifier crop').grid(
            row=2,
            column=0,
            sticky='w',
        )
        self.crop_label = ttk.Label(preview, anchor=tk.CENTER)
        self.crop_label.grid(
            row=3,
            column=0,
            sticky='nsew',
            pady=(4, 0),
        )

        diagnostics = ttk.Frame(outer, padding=(16, 0, 0, 0))
        diagnostics.grid(row=0, column=1, sticky='nsew')
        diagnostics.columnconfigure(0, weight=1)
        ttk.Label(
            diagnostics,
            text='Traffic-light prediction',
            font=('TkDefaultFont', 15, 'bold'),
        ).grid(row=0, column=0, sticky='w')
        ttk.Label(
            diagnostics,
            text=f'Bundle: {self.node.bundle.artifact_id}',
            wraplength=500,
        ).grid(row=1, column=0, sticky='w', pady=(4, 8))
        self.signal_class_label = tk.Label(
            diagnostics,
            textvariable=self._signal_class_text,
            font=('TkDefaultFont', 28, 'bold'),
            fg='#555555',
        )
        self.signal_class_label.grid(row=2, column=0, sticky='ew')

        probabilities = ttk.Frame(diagnostics)
        probabilities.grid(row=3, column=0, sticky='ew', pady=(4, 6))
        probabilities.columnconfigure(1, weight=1)
        for row_index, label in enumerate(self.node.class_labels):
            ttk.Label(probabilities, text=label, width=11).grid(
                row=row_index,
                column=0,
                sticky='w',
                pady=2,
            )
            ttk.Progressbar(
                probabilities,
                maximum=1.0,
                variable=self._probability_vars[label],
            ).grid(
                row=row_index,
                column=1,
                sticky='ew',
                padx=(4, 8),
                pady=2,
            )
            ttk.Label(
                probabilities,
                textvariable=self._probability_text[label],
                width=8,
                anchor=tk.E,
            ).grid(row=row_index, column=2, sticky='e')

        ttk.Label(
            diagnostics,
            textvariable=self._signal_status_text,
            wraplength=500,
            justify=tk.LEFT,
            font=('TkDefaultFont', 11, 'bold'),
        ).grid(row=4, column=0, sticky='ew')
        ttk.Label(
            diagnostics,
            textvariable=self._signal_detail_text,
            wraplength=500,
            justify=tk.LEFT,
        ).grid(row=5, column=0, sticky='ew', pady=(4, 8))

        ttk.Separator(diagnostics).grid(
            row=6,
            column=0,
            sticky='ew',
            pady=6,
        )
        ttk.Label(
            diagnostics,
            text='Drive command vectors',
            font=('TkDefaultFont', 15, 'bold'),
        ).grid(row=7, column=0, sticky='w')
        ttk.Label(
            diagnostics,
            textvariable=self._drive_status_text,
            font=('TkDefaultFont', 12, 'bold'),
        ).grid(row=8, column=0, sticky='w', pady=(2, 0))
        self.vector_canvas = tk.Canvas(
            diagnostics,
            width=470,
            height=270,
            background='#f8f8f8',
            highlightthickness=1,
            highlightbackground='#bbbbbb',
        )
        self.vector_canvas.grid(row=9, column=0, sticky='ew', pady=(4, 4))
        self._draw_vector_axes()
        ttk.Label(
            diagnostics,
            textvariable=self._prediction_text,
            foreground='#1f5fbf',
        ).grid(row=10, column=0, sticky='w')
        ttk.Label(
            diagnostics,
            textvariable=self._actual_text,
            foreground='#14843c',
        ).grid(row=11, column=0, sticky='w')
        warning = tk.Label(
            diagnostics,
            textvariable=self._warning_text,
            fg='#cc0000',
            justify=tk.LEFT,
            wraplength=500,
        )
        warning.grid(row=12, column=0, sticky='ew', pady=(4, 2))
        ttk.Button(
            diagnostics,
            text='Quit and stop mission',
            command=self.close,
        ).grid(row=13, column=0, sticky='ew', pady=(2, 4))
        ttk.Label(
            diagnostics,
            text=(
                'Read-only monitor: no model inference and no ROS '
                'publishers. Blue dashed = model prediction; green solid '
                '= command emitted on /xycar_motor. Q/Esc/window close '
                'stops the integrated launch.'
            ),
            wraplength=500,
            justify=tk.LEFT,
        ).grid(row=14, column=0, sticky='sw')

    def _draw_vector_axes(self) -> None:
        canvas = self.vector_canvas
        origin = (235.0, 235.0)
        maximum_length = 185.0
        canvas.create_line(
            origin[0],
            origin[1],
            origin[0],
            origin[1] - maximum_length,
            fill='#cccccc',
        )
        for angle in (-100.0, 100.0):
            endpoint = drive_vector_endpoint(
                angle,
                self.node.speed_scale,
                origin=origin,
                maximum_length=maximum_length,
                speed_scale=self.node.speed_scale,
            )
            canvas.create_line(
                origin[0],
                origin[1],
                endpoint[0],
                endpoint[1],
                fill='#dddddd',
            )
        canvas.create_text(35, 45, text='LEFT  -100', fill='#666666')
        canvas.create_text(235, 30, text='FRONT  0', fill='#666666')
        canvas.create_text(430, 45, text='RIGHT  +100', fill='#666666')
        canvas.create_text(
            235,
            258,
            text=(
                f'arrow length: |speed| / {self.node.speed_scale:g} '
                '(visual scale)'
            ),
            fill='#666666',
        )

    def _schedule_update(self) -> None:
        if self.closed:
            return
        if not rclpy.ok():
            self.close()
            return
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self._refresh()
        self.root.after(20, self._schedule_update)

    def _refresh(self) -> None:
        snapshot = self.node.snapshot()
        now = time.monotonic()
        camera_age = _sample_age(snapshot.latest_frame, now)
        signal_age = _age(snapshot.signal_received_monotonic, now)
        prediction_age = _drive_age(snapshot.prediction, now)
        actual_age = _drive_age(snapshot.actual, now)
        exact_match = (
            snapshot.signal is not None
            and snapshot.matched_frame is not None
        )
        display_frame = (
            snapshot.matched_frame
            if exact_match
            else snapshot.latest_frame
        )
        signature = (
            None
            if display_frame is None
            else display_frame.stamp_key,
            snapshot.signal,
            exact_match,
            snapshot.camera_error,
        )
        if signature != self._render_signature:
            frame_panel = monitor_frame_panel(
                None if display_frame is None else display_frame.image,
                snapshot.signal,
                exact_match=exact_match,
            )
            crop_panel = monitor_crop_panel(
                None if display_frame is None else display_frame.image,
                snapshot.signal,
                exact_match=exact_match,
            )
            self._frame_photo = _photo_for_panel(frame_panel, (880, 570))
            self.frame_label.configure(image=self._frame_photo)
            self._crop_photo = _photo_for_panel(crop_panel, (880, 170))
            self.crop_label.configure(image=self._crop_photo)
            self._render_signature = signature

        signal = snapshot.signal
        probability_map = (
            {}
            if signal is None or signal.probabilities is None
            else dict(zip(signal.class_labels, signal.probabilities, strict=True))
        )
        for label in self.node.class_labels:
            probability = probability_map.get(label, 0.0)
            self._probability_vars[label].set(probability)
            self._probability_text[label].set(f'{probability * 100.0:.1f}%')

        if signal is None:
            raw_class = 'WAITING'
            self._signal_status_text.set('Waiting for policy signal diagnostics')
            self._signal_detail_text.set(
                'The policy publishes diagnostics only while its mission '
                'gate is active and signal classification runs.'
            )
        else:
            raw_class = signal.raw_class
            match_text = (
                'exact frame matched'
                if exact_match
                else 'FRAME MATCH UNAVAILABLE — bbox not overlaid'
            )
            gate = 'PASS' if signal.width_gate_accepted else 'REJECT'
            self._signal_status_text.set(
                f'{signal.source} | width gate={gate} | {match_text}'
            )
            self._signal_detail_text.set(
                f'raw={signal.raw_class} | final={signal.final_action} | '
                f'phase={signal.phase} | mission={signal.mission_state}\n'
                f'candidate={signal.candidate} | vote='
                f'{signal.candidate_reads}/{signal.required_reads} | '
                f'vote update={"YES" if signal.vote_updated else "NO"}\n'
                f'policy frame={signal.frame_sequence} | traffic inference='
                f'{signal.detector_inference_ms:.1f}ms | signal age='
                f'{_age_text(signal_age)}'
            )
        self._signal_class_text.set(raw_class)
        self.signal_class_label.configure(fg=_signal_color(raw_class))

        self._drive_status_text.set(
            'DRIVE ENABLED' if snapshot.enabled else 'DRIVE OFF / STOPPED'
        )
        self._update_vectors(
            prediction=snapshot.prediction,
            actual=snapshot.actual,
            prediction_fresh=(
                prediction_age is not None
                and prediction_age <= self.node.prediction_stale_sec
            ),
            actual_fresh=(
                actual_age is not None
                and actual_age <= self.node.motor_stale_sec
            ),
            prediction_age=prediction_age,
            actual_age=actual_age,
        )

        warnings = []
        if camera_age is None or camera_age > self.node.camera_stale_sec:
            warnings.append(f'CAMERA STALE ({_age_text(camera_age)})')
        if signal_age is None or signal_age > self.node.signal_stale_sec:
            warnings.append(f'SIGNAL STALE ({_age_text(signal_age)})')
        if prediction_age is None or (
            prediction_age > self.node.prediction_stale_sec
        ):
            warnings.append(f'PREDICTION STALE ({_age_text(prediction_age)})')
        if actual_age is None or actual_age > self.node.motor_stale_sec:
            warnings.append(f'MOTOR TOPIC STALE ({_age_text(actual_age)})')
        for error in (
            snapshot.camera_error,
            snapshot.signal_error,
            snapshot.control_error,
        ):
            if error:
                warnings.append(error)
        self._warning_text.set(' | '.join(warnings))

    def _update_vectors(
        self,
        *,
        prediction: DriveVectorSample | None,
        actual: DriveVectorSample | None,
        prediction_fresh: bool,
        actual_fresh: bool,
        prediction_age: float | None,
        actual_age: float | None,
    ) -> None:
        canvas = self.vector_canvas
        canvas.delete('drive-vector')
        origin = (235.0, 235.0)
        maximum_length = 185.0
        if prediction is not None:
            self._draw_drive_vector(
                prediction,
                color='#1f5fbf' if prediction_fresh else '#8a96a8',
                dash=(7, 4),
                origin=origin,
                maximum_length=maximum_length,
            )
            mode = 'SHORTCUT' if prediction.is_shortcut else 'BASE'
            self._prediction_text.set(
                f'Prediction: angle={prediction.angle:+.1f}, '
                f'speed={prediction.speed:.1f}, mode={mode}, '
                f'inference={prediction.inference_ms:.1f}ms, '
                f'age={_age_text(prediction_age)}'
            )
        else:
            self._prediction_text.set('Prediction: waiting')
        if actual is not None:
            self._draw_drive_vector(
                actual,
                color='#14843c' if actual_fresh else '#8a968e',
                dash=None,
                origin=origin,
                maximum_length=maximum_length,
            )
            self._actual_text.set(
                f'Motor output: angle={actual.angle:+.1f}, '
                f'speed={actual.speed:.1f}, age={_age_text(actual_age)}'
            )
        else:
            self._actual_text.set('Motor output: waiting')

    def _draw_drive_vector(
        self,
        sample: DriveVectorSample,
        *,
        color: str,
        dash: tuple[int, int] | None,
        origin: tuple[float, float],
        maximum_length: float,
    ) -> None:
        endpoint = drive_vector_endpoint(
            sample.angle,
            sample.speed,
            origin=origin,
            maximum_length=maximum_length,
            speed_scale=self.node.speed_scale,
        )
        if math.isclose(sample.speed, 0.0, abs_tol=1e-9):
            self.vector_canvas.create_oval(
                origin[0] - 5,
                origin[1] - 5,
                origin[0] + 5,
                origin[1] + 5,
                outline=color,
                width=3,
                tags='drive-vector',
            )
            return
        self.vector_canvas.create_line(
            origin[0],
            origin[1],
            endpoint[0],
            endpoint[1],
            fill=color,
            width=4,
            arrow=tk.LAST,
            arrowshape=(14, 18, 7),
            dash=dash,
            tags='drive-vector',
        )

    def _on_gui_exception(self, exception_type, exception, traceback) -> None:
        try:
            self.node.get_logger().error(
                f'traffic shortcut GUI failed: {exception_type.__name__}: '
                f'{exception}'
            )
        finally:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.root.destroy()


def frame_stamp_key(sec: int, nanosec: int) -> tuple[int, int] | None:
    """Return an exact match key, rejecting an unspecified zero stamp."""
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError('camera timestamp is out of range')
    if sec == 0 and nanosec == 0:
        return None
    return sec, nanosec


def drive_vector_endpoint(
    angle: float,
    speed: float,
    *,
    origin: tuple[float, float],
    maximum_length: float,
    speed_scale: float,
) -> tuple[float, float]:
    """Map normalized steering and speed to a screen-space arrow endpoint."""
    values = (*origin, angle, speed, maximum_length, speed_scale)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('drive vector values must be finite')
    if maximum_length <= 0.0 or speed_scale <= 0.0:
        raise ValueError('drive vector scales must be positive')
    visual_angle = math.radians(
        max(-100.0, min(100.0, float(angle))) * 0.6
    )
    length = min(abs(float(speed)) / speed_scale, 1.0) * maximum_length
    direction = -1.0 if speed < 0.0 else 1.0
    return (
        origin[0] + direction * math.sin(visual_angle) * length,
        origin[1] - direction * math.cos(visual_angle) * length,
    )


def monitor_frame_panel(
    frame: np.ndarray | None,
    signal: SignalDebugSnapshot | None,
    *,
    exact_match: bool,
) -> np.ndarray:
    """Render bbox only when diagnostics match the exact camera stamp."""
    if frame is None:
        return _placeholder_panel('WAITING FOR CAMERA', width=640, height=420)
    panel = frame.copy()
    if signal is None:
        _put_overlay_text(panel, 'WAITING FOR SIGNAL', (20, 38), (0, 165, 255))
        return panel
    if not exact_match:
        _put_overlay_text(
            panel,
            'FRAME MATCH UNAVAILABLE',
            (20, 38),
            (0, 0, 255),
        )
        return panel
    if signal.bbox is None:
        _put_overlay_text(panel, 'YOLO: NO BOX', (20, 38), (0, 165, 255))
        return panel
    box = signal.bbox
    height, width = panel.shape[:2]
    x1 = min(max(math.floor(box.x), 0), width - 1)
    y1 = min(max(math.floor(box.y), 0), height - 1)
    x2 = min(max(math.ceil(box.x + box.width), 0), width - 1)
    y2 = min(max(math.ceil(box.y + box.height), 0), height - 1)
    color = (0, 190, 0) if signal.width_gate_accepted else (0, 120, 255)
    if x2 > x1 and y2 > y1:
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
    probability = _raw_probability(signal)
    label = (
        f'{signal.raw_class} {probability * 100.0:.1f}% '
        f'w={box.width:.1f} conf={box.confidence:.2f} '
        f'{signal.source}'
    )
    _put_overlay_text(panel, label, (x1, max(24, y1 - 8)), color)
    return panel


def monitor_crop_panel(
    frame: np.ndarray | None,
    signal: SignalDebugSnapshot | None,
    *,
    exact_match: bool,
) -> np.ndarray:
    """Extract the CNN crop only from the exact frame that produced it."""
    if (
        frame is None
        or signal is None
        or signal.crop is None
        or not exact_match
    ):
        text = (
            'FRAME MATCH UNAVAILABLE'
            if signal is not None and not exact_match
            else 'NO CNN CROP'
        )
        return _placeholder_panel(text, width=640, height=120)
    bounds = signal.crop
    height, width = frame.shape[:2]
    x1 = min(max(bounds.x1, 0), width)
    y1 = min(max(bounds.y1, 0), height)
    x2 = min(max(bounds.x2, 0), width)
    y2 = min(max(bounds.y2, 0), height)
    if x2 <= x1 or y2 <= y1:
        return _placeholder_panel('INVALID CNN CROP', width=640, height=120)
    return frame[y1:y2, x1:x2].copy()


def _finite_message_values(
    message: Float32MultiArray,
    *,
    expected_length: int,
) -> tuple[float, ...]:
    if len(message.data) != expected_length:
        raise ValueError(
            f'expected {expected_length} values, got {len(message.data)}'
        )
    values = tuple(float(value) for value in message.data)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('message contains NaN or Inf')
    return values


def _validate_drive_values(angle: float, speed: float) -> None:
    if not -100.0 <= angle <= 100.0:
        raise ValueError('angle is outside normalized -100..100')
    if speed < 0.0:
        raise ValueError('traffic shortcut speed must not be negative')


def _raw_probability(signal: SignalDebugSnapshot) -> float:
    if signal.probabilities is None:
        return 0.0
    mapping = dict(
        zip(signal.class_labels, signal.probabilities, strict=True)
    )
    if signal.raw_class in mapping:
        return mapping[signal.raw_class]
    return max(signal.probabilities, default=0.0)


def _sample_age(frame: MonitorFrame | None, now: float) -> float | None:
    if frame is None:
        return None
    return max(0.0, now - frame.received_monotonic)


def _drive_age(
    sample: DriveVectorSample | None,
    now: float,
) -> float | None:
    if sample is None:
        return None
    return max(0.0, now - sample.received_monotonic)


def _age(received: float | None, now: float) -> float | None:
    if received is None:
        return None
    return max(0.0, now - received)


def _age_text(age: float | None) -> str:
    return 'n/a' if age is None else f'{age:.2f}s'


def _signal_color(raw_class: str) -> str:
    return {
        'STOP': '#cc0000',
        'LEFT': '#008b45',
        'STRAIGHT': '#008b45',
        'UNKNOWN': '#555555',
        'WAITING': '#555555',
    }.get(raw_class, '#555555')


def _placeholder_panel(text: str, *, width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    _put_overlay_text(
        panel,
        text,
        (20, max(38, height // 2)),
        (180, 180, 180),
    )
    return panel


def _put_overlay_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _photo_for_panel(
    bgr_image: np.ndarray,
    bounds: tuple[int, int],
) -> ImageTk.PhotoImage:
    rgb_image = np.ascontiguousarray(bgr_image[:, :, ::-1])
    image = PilImage.fromarray(rgb_image)
    resampling = getattr(PilImage, 'Resampling', PilImage)
    image.thumbnail(bounds, resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: TrafficShortcutMonitorNode | None = None
    try:
        node = TrafficShortcutMonitorNode()
        application = TrafficShortcutMonitorApplication(node)
        application.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
