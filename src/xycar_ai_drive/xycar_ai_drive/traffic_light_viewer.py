"""Passive Tk viewer for the deployed traffic-light classifier contract."""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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

from xycar_ai_drive.traffic_light_detector import (
    InitialStopSignalLatch,
    InitialStopSignalLatchSnapshot,
    InitialWaitSignalLatch,
    LampAction,
    SignalClass,
    SignalInspection,
    TrafficClassifierCadence,
    TrafficClassifierDetector,
    TrafficSignalLatch,
    TrafficSignalLatchSnapshot,
    should_update_signal_vote,
)
from xycar_ai_drive.traffic_light_runtime import create_onnx_detector
from xycar_ai_drive.traffic_shortcut_artifact import (
    TrafficShortcutBundle,
    load_traffic_shortcut_bundle,
)

BundleLoader = Callable[[str], TrafficShortcutBundle]
DetectorFactory = Callable[[TrafficShortcutBundle], object]


@dataclass(frozen=True)
class TrafficLightViewerResult:
    """One inference result bound to the exact camera frame it inspected."""

    frame: np.ndarray
    frame_sequence: int
    source_monotonic: float
    completed_monotonic: float
    inference_ms: float
    inspection: SignalInspection | None
    width_gate_accepted: bool
    final_action: LampAction
    latch_snapshot: (
        TrafficSignalLatchSnapshot | InitialStopSignalLatchSnapshot
    )
    vote_updated: bool = True
    inference_kind: str = 'YOLO+CNN'
    error: str | None = None


class TrafficLightViewerNode(Node):
    """Subscribe and infer without creating Joy or publisher endpoints."""

    def __init__(
        self,
        parameter_overrides: Sequence[Parameter] | None = None,
        *,
        bundle_loader: BundleLoader = load_traffic_shortcut_bundle,
        detector_factory: DetectorFactory = create_onnx_detector,
    ) -> None:
        super().__init__(
            'traffic_light_viewer',
            parameter_overrides=parameter_overrides,
        )
        self.declare_parameter('bundle_dir', '')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('camera_stale_sec', 1.0)
        self.bundle_dir = str(self.get_parameter('bundle_dir').value).strip()
        self.camera_topic = str(
            self.get_parameter('camera_topic').value
        ).strip()
        self.camera_stale_sec = float(
            self.get_parameter('camera_stale_sec').value
        )
        if not self.bundle_dir:
            raise ValueError('bundle_dir must not be empty')
        if not self.camera_topic:
            raise ValueError('camera_topic must not be empty')
        if (
            not math.isfinite(self.camera_stale_sec)
            or self.camera_stale_sec <= 0.0
        ):
            raise ValueError('camera_stale_sec must be finite and positive')

        self.bundle = bundle_loader(self.bundle_dir)
        self._validate_bundle_contract()
        self.class_labels = tuple(
            SignalClass(value)
            for value in self.bundle.detector.classifier_classes
        )
        detector = detector_factory(self.bundle)
        if not isinstance(detector, TrafficClassifierDetector):
            raise ValueError(
                'traffic viewer requires the CNN classifier detector'
            )
        self._detector = detector
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
        if self.bundle.schema_version == 15:
            self._latch = InitialWaitSignalLatch(
                bbox_width_min=self.bundle.detector.bbox_width_min,
                bbox_width_max=self.bundle.detector.bbox_width_max,
                consecutive_reads=(
                    self.bundle.detector.red_consecutive_reads
                ),
            )
            self._latch.reset(
                initial_stop_armed=True,
                wait_for_signal=True,
            )
        elif self.bundle.schema_version == 14:
            clear_reads = self.bundle.initial_stop_clear_consecutive_reads
            if clear_reads is None:
                raise ValueError('schema v14 initial STOP clear reads missing')
            self._latch = InitialStopSignalLatch(
                bbox_width_min=self.bundle.detector.bbox_width_min,
                bbox_width_max=self.bundle.detector.bbox_width_max,
                stop_consecutive_reads=(
                    self.bundle.detector.red_consecutive_reads
                ),
                clear_consecutive_reads=clear_reads,
                left_consecutive_reads=(
                    self.bundle.detector.left_consecutive_reads
                ),
                straight_consecutive_reads=(
                    self.bundle.detector.straight_consecutive_reads
                ),
            )
            self._latch.reset(
                initial_stop_armed=True,
                wait_for_signal=True,
            )
        else:
            self._latch = TrafficSignalLatch(
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
        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._pending_frame: tuple[int, np.ndarray, float, int] | None = None
        self._result: TrafficLightViewerResult | None = None
        self._last_camera_monotonic: float | None = None
        self._camera_error: str | None = None
        self._frame_sequence = 0
        self._latch_generation = 0
        self._last_action = LampAction.UNKNOWN
        self._worker_stop = False
        self._shutdown_started = False
        self._worker = threading.Thread(
            target=self._inference_worker,
            name='traffic-light-viewer-inference',
            daemon=True,
        )
        self._worker.start()
        self.camera_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self._on_camera,
            qos_profile_sensor_data,
        )
        self.get_logger().warning(
            'Passive traffic-light viewer started with no ROS publishers, '
            'Joy subscription, policy server, or motor endpoint.'
        )
        clear_contract = (
            f',clear:{self.bundle.initial_stop_clear_consecutive_reads}'
            if self.bundle.schema_version in {14, 15}
            else ''
        )
        self.get_logger().info(
            f'bundle={self.bundle.artifact_id}, camera={self.camera_topic}, '
            f'classes={"/".join(value.value for value in self.class_labels)}, '
            f'search_every={self.bundle.detector.inference_every_n_frames}, '
            f'classify_every='
            f'{self.bundle.detector.classification_every_n_frames_after_detection}, '
            f'width={self.bundle.detector.bbox_width_min}..'
            f'{self.bundle.detector.bbox_width_max}, '
            f'votes=stop:{self.bundle.detector.red_consecutive_reads},'
            f'left:{self.bundle.detector.left_consecutive_reads},'
            f'straight:{self.bundle.detector.straight_consecutive_reads}'
            f'{clear_contract}'
        )

    def _validate_bundle_contract(self) -> None:
        detector = self.bundle.detector
        if self.bundle.schema_version not in {
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        }:
            raise ValueError(
                'traffic viewer supports classifier bundle schema 4..15'
            )
        if detector.mode != 'yolo_cnn_classifier':
            raise ValueError(
                'traffic viewer requires yolo_cnn_classifier mode'
            )
        expected_width = (
            (40, 225)
            if self.bundle.schema_version in {
                6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            }
            else (45, 200)
        )
        expected_votes = (
            (5, 5, 5)
            if self.bundle.schema_version == 15
            else (15, 15, 15)
            if self.bundle.schema_version in {13, 14}
            else (10, 30, 30)
            if self.bundle.schema_version == 12
            else (30, 30, 30)
            if self.bundle.schema_version in {10, 11}
            else (10, 15, 15)
            if self.bundle.schema_version == 9
            else (3, 15, 15)
            if self.bundle.schema_version in {7, 8}
            else (2, 2, 2)
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
            raise ValueError(
                'traffic viewer bundle width/every3/vote contract mismatch'
            )
        expected_classification_every = (
            1
            if self.bundle.schema_version in {
                8, 9, 10, 11, 12, 13, 14, 15
            }
            else 3
        )
        expected_reuse_detected_bbox = self.bundle.schema_version in {
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        }
        if (
            detector.classification_every_n_frames_after_detection
            != expected_classification_every
            or detector.reuse_detected_bbox_between_yolo_frames
            is not expected_reuse_detected_bbox
        ):
            raise ValueError('traffic viewer classifier cadence mismatch')

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
                raise ValueError('converted frame is not uint8 BGR')
            frame = np.ascontiguousarray(frame)
        except Exception as exc:  # noqa: BLE001 - ROS conversion boundary
            with self._lock:
                self._camera_error = f'camera conversion failed: {exc}'
            self.get_logger().warning(
                self._camera_error,
                throttle_duration_sec=1.0,
            )
            return

        now = time.monotonic()
        with self._frame_condition:
            self._frame_sequence += 1
            sequence = self._frame_sequence
            self._last_camera_monotonic = now
            self._camera_error = None
            self._pending_frame = (
                sequence,
                frame.copy(),
                now,
                self._latch_generation,
            )
            self._frame_condition.notify()

    def _inference_worker(self) -> None:
        processed_sequence = 0
        while True:
            with self._frame_condition:
                while not self._worker_stop and (
                    self._pending_frame is None
                    or self._pending_frame[0] <= processed_sequence
                ):
                    self._frame_condition.wait()
                if self._worker_stop:
                    return
                assert self._pending_frame is not None
                sequence, frame, source_monotonic, generation = (
                    self._pending_frame
                )
                processed_sequence = sequence
                inference_plan = self._signal_cadence.plan(
                    frame_sequence=sequence,
                )

            if (
                not inference_plan.run_detector
                and inference_plan.classification_box is None
            ):
                continue

            started = time.perf_counter()
            try:
                if inference_plan.run_detector:
                    inspection = self._detector.inspect_signal(frame)
                    inference_kind = 'YOLO+CNN'
                else:
                    assert inference_plan.classification_box is not None
                    inspection = self._detector.classify_signal_box(
                        frame,
                        inference_plan.classification_box,
                    )
                    inference_kind = 'CNN cached bbox'
                inference_ms = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    if generation != self._latch_generation:
                        continue
                    if inference_plan.run_detector:
                        self._signal_cadence.observe_detection(
                            frame_sequence=sequence,
                            box=(
                                None
                                if inspection is None
                                else inspection.bbox
                            ),
                        )
                    else:
                        self._signal_cadence.observe_classification(
                            frame_sequence=sequence,
                        )
                    reading = (
                        None if inspection is None else inspection.reading
                    )
                    vote_updated = should_update_signal_vote(
                        reading_observed=True,
                        detector_ran=inference_plan.run_detector,
                        fresh_yolo_only=(
                            self.bundle.control_vote_on_fresh_yolo_only
                        ),
                    )
                    if vote_updated:
                        final_action = self._latch.observe(reading)
                        self._last_action = final_action
                    else:
                        final_action = self._last_action
                    width_gate_accepted = bool(
                        reading is not None
                        and self.bundle.detector.bbox_width_min
                        <= reading.bbox_width
                        <= self.bundle.detector.bbox_width_max
                    )
                    self._result = TrafficLightViewerResult(
                        frame=frame,
                        frame_sequence=sequence,
                        source_monotonic=source_monotonic,
                        completed_monotonic=time.monotonic(),
                        inference_ms=inference_ms,
                        inspection=inspection,
                        width_gate_accepted=width_gate_accepted,
                        final_action=final_action,
                        latch_snapshot=self._latch.snapshot,
                        vote_updated=vote_updated,
                        inference_kind=inference_kind,
                    )
            except Exception as exc:  # noqa: BLE001 - inference boundary
                inference_ms = (time.perf_counter() - started) * 1000.0
                error = f'traffic inference failed: {exc}'
                with self._lock:
                    if generation != self._latch_generation:
                        continue
                    self._result = TrafficLightViewerResult(
                        frame=frame,
                        frame_sequence=sequence,
                        source_monotonic=source_monotonic,
                        completed_monotonic=time.monotonic(),
                        inference_ms=inference_ms,
                        inspection=None,
                        width_gate_accepted=False,
                        final_action=LampAction.UNKNOWN,
                        latch_snapshot=self._latch.snapshot,
                        vote_updated=False,
                        inference_kind=(
                            'YOLO+CNN'
                            if inference_plan.run_detector
                            else 'CNN cached bbox'
                        ),
                        error=error,
                    )
                self.get_logger().error(
                    error,
                    throttle_duration_sec=1.0,
                )

    def snapshot(
        self,
    ) -> tuple[
        TrafficLightViewerResult | None,
        float | None,
        str | None,
    ]:
        with self._lock:
            return (
                self._result,
                self._last_camera_monotonic,
                self._camera_error,
            )

    def reset_vote(self) -> None:
        with self._frame_condition:
            self._latch_generation += 1
            if isinstance(
                self._latch,
                (InitialStopSignalLatch, InitialWaitSignalLatch),
            ):
                self._latch.reset(
                    initial_stop_armed=True,
                    wait_for_signal=True,
                )
            else:
                self._latch.reset()
            self._last_action = LampAction.UNKNOWN
            self._signal_cadence.reset(
                frame_sequence=self._frame_sequence,
            )
            if self._result is not None:
                self._result = replace(
                    self._result,
                    final_action=LampAction.UNKNOWN,
                    latch_snapshot=self._latch.snapshot,
                    vote_updated=False,
                )
        self.get_logger().info('Traffic signal vote and initial-stop phase reset')

    def shutdown(self) -> None:
        with self._frame_condition:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self._worker_stop = True
            self._frame_condition.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)


class TrafficLightViewerApplication:
    """Tk presentation of raw detector and deployed latch state."""

    def __init__(self, node: TrafficLightViewerNode) -> None:
        self.node = node
        self.closed = False
        self.root = tk.Tk()
        self.root.title('Xycar traffic-light prediction viewer (PASSIVE)')
        self.root.geometry('1380x820')
        self.root.minsize(1120, 700)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self._frame_photo: ImageTk.PhotoImage | None = None
        self._crop_photo: ImageTk.PhotoImage | None = None
        self._render_signature: tuple[object, ...] | None = None
        self.class_labels = node.class_labels
        self._probability_vars = {
            signal_class: tk.DoubleVar(value=0.0)
            for signal_class in self.class_labels
        }
        self._probability_text = {
            signal_class: tk.StringVar(value='0.0%')
            for signal_class in self.class_labels
        }
        self._status_text = tk.StringVar(value='Waiting for camera frames')
        self._detail_text = tk.StringVar(value='')
        self._action_text = tk.StringVar(value='UNKNOWN')
        self._build_layout()
        self.root.bind('<Escape>', lambda _event: self.close())
        self.root.bind('q', lambda _event: self.close())
        self.root.after(20, self._schedule_update)

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        preview = ttk.Frame(outer)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.Frame(outer, padding=(16, 0, 0, 0), width=430)
        controls.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(
            preview,
            text='Latest sampled frame (exact inference input)',
        ).pack(anchor=tk.W)
        self.frame_label = ttk.Label(preview, anchor=tk.CENTER)
        self.frame_label.pack(fill=tk.BOTH, expand=True, pady=(4, 12))
        ttk.Label(preview, text='Padded classifier crop').pack(anchor=tk.W)
        self.crop_label = ttk.Label(preview, anchor=tk.CENTER)
        self.crop_label.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        ttk.Label(
            controls,
            text='Traffic-light prediction',
            font=('TkDefaultFont', 15, 'bold'),
        ).pack(anchor=tk.W)
        ttk.Label(
            controls,
            text=f'Bundle: {self.node.bundle.artifact_id}',
            wraplength=410,
        ).pack(anchor=tk.W, pady=(4, 12))
        self.action_label = tk.Label(
            controls,
            textvariable=self._action_text,
            font=('TkDefaultFont', 28, 'bold'),
            fg='#555555',
        )
        self.action_label.pack(fill=tk.X, pady=(0, 10))

        for signal_class in self.class_labels:
            row = ttk.Frame(controls)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=signal_class.value, width=16).pack(
                side=tk.LEFT
            )
            ttk.Progressbar(
                row,
                maximum=1.0,
                variable=self._probability_vars[signal_class],
                length=180,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(
                row,
                textvariable=self._probability_text[signal_class],
                width=8,
                anchor=tk.E,
            ).pack(side=tk.RIGHT)

        ttk.Separator(controls).pack(fill=tk.X, pady=12)
        ttk.Label(
            controls,
            textvariable=self._status_text,
            wraplength=410,
            justify=tk.LEFT,
            font=('TkDefaultFont', 11, 'bold'),
        ).pack(anchor=tk.W, fill=tk.X)
        ttk.Label(
            controls,
            textvariable=self._detail_text,
            wraplength=410,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(8, 12))
        ttk.Button(
            controls,
            text='Reset vote / initial-stop phase',
            command=self._reset_vote,
        ).pack(fill=tk.X, pady=4)
        ttk.Button(controls, text='Quit', command=self.close).pack(
            fill=tk.X,
            pady=(0, 8),
        )
        ttk.Label(
            controls,
            text=(
                'Passive viewer: no Joy, policy server, ROS publisher, '
                'motor bridge, recording, or image save.\nKeys: Q/Esc quit'
            ),
            wraplength=410,
            justify=tk.LEFT,
        ).pack(side=tk.BOTTOM, anchor=tk.W)

    def _schedule_update(self) -> None:
        if self.closed:
            return
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
        self._refresh()
        self.root.after(20, self._schedule_update)

    def _refresh(self) -> None:
        result, last_camera, camera_error = self.node.snapshot()
        now = time.monotonic()
        camera_age = (
            None
            if last_camera is None
            else max(0.0, now - last_camera)
        )
        if result is None:
            status = camera_error or (
                f'Waiting for sampled frames on {self.node.camera_topic}'
            )
            if camera_age is not None:
                status += f' | latest camera frame {camera_age:.2f}s old'
            self._status_text.set(status)
            self._detail_text.set(
                'YOLO searches every third camera frame. After detection, '
                'CNN classification runs on each processed camera frame.'
            )
            return

        snapshot = result.latch_snapshot
        signature = (
            result.frame_sequence,
            result.error,
            snapshot,
            result.final_action,
        )
        if signature != self._render_signature:
            overlay = draw_signal_overlay(result)
            self._frame_photo = _photo_for_panel(overlay, (830, 530))
            self.frame_label.configure(image=self._frame_photo)
            crop = _inspection_crop(result)
            self._crop_photo = _photo_for_panel(crop, (830, 180))
            self.crop_label.configure(image=self._crop_photo)
            self._render_signature = signature

        inspection = result.inspection
        probabilities = (
            (0.0,) * len(self.class_labels)
            if inspection is None
            else inspection.probabilities
        )
        for signal_class, probability in zip(
            self.class_labels,
            probabilities,
            strict=True,
        ):
            self._probability_vars[signal_class].set(probability)
            self._probability_text[signal_class].set(
                f'{probability * 100.0:.1f}%'
            )

        sample_age = max(0.0, now - result.source_monotonic)
        stale = (
            camera_age is None
            or camera_age > self.node.camera_stale_sec
        )
        raw_class = (
            SignalClass.UNKNOWN
            if inspection is None
            else inspection.reading.signal_class
        )
        if result.error is not None:
            state = f'ERROR: {result.error}'
        elif inspection is None:
            state = 'NO DETECTION'
        elif not result.width_gate_accepted:
            state = 'WIDTH GATE REJECTED'
        else:
            state = 'DETECTED / WIDTH GATE ACCEPTED'
        if camera_error is not None:
            state = f'CAMERA ERROR: {camera_error} | {state}'
        if stale:
            state = f'CAMERA STALE | {state}'
        self._status_text.set(state)

        if inspection is None:
            bbox_text = 'bbox: none | YOLO confidence: n/a'
        else:
            box = inspection.bbox
            bbox_text = (
                f'bbox: x={box.x}, y={box.y}, w={box.width}, '
                f'h={box.height} | YOLO confidence={box.confidence:.3f}'
            )
        camera_age_text = (
            'n/a' if camera_age is None else f'{camera_age:.2f}s'
        )
        self._detail_text.set(
            f'raw class: {raw_class.value}\n'
            + (
                f'phase: {snapshot.phase.value}\n'
                if isinstance(snapshot, InitialStopSignalLatchSnapshot)
                else ''
            )
            + f'candidate: {snapshot.candidate.value} | '
            f'vote: {snapshot.candidate_reads}/{snapshot.required_reads}\n'
            f'stop latch: {"ON" if snapshot.stop_latched else "OFF"}\n'
            + (
                'schema v15: fresh YOLO+CNN only vote | all classes 5 | '
                'cached CNN diagnostics only | post-clear STOP ignored\n'
                if self.node.bundle.schema_version == 15
                else
                'schema v14: armed STOP 15 | non-STOP clear 3 | '
                'navigation LEFT 15 | post-clear STOP ignored\n'
                if self.node.bundle.schema_version == 14
                else ''
            )
            + f'{bbox_text}\n'
            f'width gate: {self.node.bundle.detector.bbox_width_min}..'
            f'{self.node.bundle.detector.bbox_width_max}px\n'
            f'frame: {result.frame_sequence} (every '
            f'{self.node.bundle.detector.inference_every_n_frames} search, '
            f'{self.node.bundle.detector.classification_every_n_frames_after_detection} '
            f'classify) | mode: {result.inference_kind} | '
            f'vote update: {"YES" if result.vote_updated else "NO"} | '
            f'inference: {result.inference_ms:.1f}ms\n'
            f'sampled frame age: {sample_age:.2f}s | '
            f'latest camera age: {camera_age_text}'
        )
        action = result.final_action
        self._action_text.set(action.value)
        self.action_label.configure(fg=_action_color(action, result.error))

    def _reset_vote(self) -> None:
        self.node.reset_vote()
        self._render_signature = None
        self._refresh()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.node.shutdown()
        self.root.destroy()


def draw_signal_overlay(result: TrafficLightViewerResult) -> np.ndarray:
    """Draw a result only on the frame that produced it."""
    frame = result.frame.copy()
    if result.error is not None:
        _put_overlay_text(frame, 'INFERENCE ERROR', (20, 40), (0, 0, 255))
        return frame
    inspection = result.inspection
    if inspection is None:
        _put_overlay_text(frame, 'NO DETECTION', (20, 40), (0, 165, 255))
        return frame

    box = inspection.bbox
    color = (0, 190, 0) if result.width_gate_accepted else (0, 120, 255)
    height, width = frame.shape[:2]
    x1 = min(max(math.floor(box.x), 0), width - 1)
    y1 = min(max(math.floor(box.y), 0), height - 1)
    x2 = min(max(math.ceil(box.x + box.width), 0), width - 1)
    y2 = min(max(math.ceil(box.y + box.height), 0), height - 1)
    if x2 > x1 and y2 > y1:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = (
        f'{inspection.reading.signal_class.value} '
        f'{inspection.reading.probability * 100.0:.1f}% '
        f'w={box.width} conf={box.confidence:.2f}'
    )
    _put_overlay_text(frame, label, (x1, max(24, y1 - 8)), color)
    return frame


def _inspection_crop(result: TrafficLightViewerResult) -> np.ndarray:
    if result.inspection is None or result.error is not None:
        panel = np.zeros((120, 420, 3), dtype=np.uint8)
        text = 'NO CROP' if result.error is None else 'INFERENCE ERROR'
        _put_overlay_text(panel, text, (20, 68), (180, 180, 180))
        return panel
    bounds = result.inspection.crop_bounds
    return result.frame[bounds.y1:bounds.y2, bounds.x1:bounds.x2]


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


def _action_color(action: LampAction, error: str | None) -> str:
    if error is not None:
        return '#cc0000'
    return {
        LampAction.RED: '#cc0000',
        LampAction.LEFT: '#008b45',
        LampAction.STRAIGHT: '#008b45',
        LampAction.UNKNOWN: '#555555',
    }[action]


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: TrafficLightViewerNode | None = None
    try:
        node = TrafficLightViewerNode()
        application = TrafficLightViewerApplication(node)
        application.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
