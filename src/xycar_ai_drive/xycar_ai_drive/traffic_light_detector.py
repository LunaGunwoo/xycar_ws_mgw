"""YOLO traffic-light decoding and the fail-closed lamp latch contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import cv2
import numpy as np


class TrafficLightError(RuntimeError):
    """Raised when detector input or output violates the deployed contract."""


class LampAction(str, Enum):
    UNKNOWN = 'UNKNOWN'
    RED = 'RED'
    LEFT = 'LEFT'
    STRAIGHT = 'STRAIGHT'


class SignalClass(str, Enum):
    """Raw CNN traffic-light classes before mission action mapping."""

    UNKNOWN = 'UNKNOWN'
    RED = 'red'
    YELLOW = 'yellow'
    LEFT_GREEN = 'left_green'
    STRAIGHT_GREEN = 'straight_green'


@dataclass(frozen=True)
class DetectionBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True)
class LampReading:
    scores: tuple[float, float, float, float]
    bbox_width: int
    confidence: float


@dataclass(frozen=True)
class SignalReading:
    signal_class: SignalClass
    probability: float
    bbox_width: int
    confidence: float


@dataclass(frozen=True)
class ImageBounds:
    """Inclusive-exclusive pixel bounds for a frame crop."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class SignalInspection:
    """Detailed classifier result used by passive diagnostic viewers."""

    reading: SignalReading
    bbox: DetectionBox
    crop_bounds: ImageBounds
    probabilities: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrafficSignalLatchSnapshot:
    """Read-only state of the deployed two-read signal vote."""

    candidate: SignalClass
    candidate_reads: int
    required_reads: int
    stop_latched: bool


class _OnnxSession(Protocol):
    def run(self, output_names, inputs): ...


def decode_detection_box(
    output: np.ndarray,
    *,
    frame_height: int,
    frame_width: int,
    confidence_threshold: float = 0.25,
) -> DetectionBox | None:
    """Decode the highest-confidence `[1,5,8400]` YOLO prediction."""
    if (
        not isinstance(output, np.ndarray)
        or output.shape != (1, 5, 8400)
        or output.dtype.kind not in {'f'}
    ):
        raise TrafficLightError('traffic ONNX output must be [1,5,8400] float')
    if not np.isfinite(output).all():
        raise TrafficLightError('traffic ONNX output contains NaN or Inf')
    if frame_height < 1 or frame_width < 1:
        raise TrafficLightError('camera frame dimensions must be positive')
    if not math.isfinite(confidence_threshold) or not 0.0 < confidence_threshold < 1.0:
        raise TrafficLightError('confidence threshold must be in (0,1)')
    predictions = output[0].T
    confidence = predictions[:, 4]
    kept = confidence > confidence_threshold
    if not bool(np.any(kept)):
        return None
    candidates = predictions[kept]
    candidate_confidence = confidence[kept]
    index = int(np.argmax(candidate_confidence))
    center_x, center_y, width, height = (
        float(value) for value in candidates[index, :4]
    )
    scale_x = frame_width / 640.0
    scale_y = frame_height / 640.0
    return DetectionBox(
        x=int((center_x - width / 2.0) * scale_x),
        y=int((center_y - height / 2.0) * scale_y),
        width=int(width * scale_x),
        height=int(height * scale_y),
        confidence=float(candidate_confidence[index]),
    )


def lamp_scores(
    bgr_frame: np.ndarray,
    box: DetectionBox,
    *,
    percentile: float = 80.0,
) -> tuple[float, float, float, float]:
    """Measure four horizontal lamps using the team's HSV-V percentile."""
    _validate_bgr_frame(bgr_frame)
    if box.width < 1 or box.height < 1:
        raise TrafficLightError('traffic detection bbox is empty')
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise TrafficLightError('lamp percentile must be in [0,100]')
    value = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    pitch = box.width / 4.0
    center_y = box.y + box.height // 2
    radius = max(3, int(min(pitch, box.height) * 0.3))
    scores: list[float] = []
    for index in range(4):
        center_x = int(box.x + pitch * (index + 0.5))
        patch = value[
            max(0, center_y - radius) : center_y + radius,
            max(0, center_x - radius) : center_x + radius,
        ]
        scores.append(
            float(np.percentile(patch, percentile)) if patch.size else 0.0
        )
    if not all(math.isfinite(score) for score in scores):
        raise TrafficLightError('traffic lamp scores contain NaN or Inf')
    return scores[0], scores[1], scores[2], scores[3]


class TrafficLightDetector:
    """Run the exact team ONNX preprocessing and lamp score extraction."""

    def __init__(
        self,
        *,
        session: _OnnxSession,
        confidence_threshold: float = 0.25,
        percentile: float = 80.0,
    ) -> None:
        self._session = session
        self._confidence_threshold = float(confidence_threshold)
        self._percentile = float(percentile)

    def read_lamp(self, bgr_frame: np.ndarray) -> LampReading | None:
        _validate_bgr_frame(bgr_frame)
        height, width = bgr_frame.shape[:2]
        resized = cv2.resize(bgr_frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(np.transpose(rgb, (2, 0, 1)), 0)
        try:
            outputs = self._session.run(None, {'images': tensor})
        except Exception as exc:  # noqa: BLE001 - runtime boundary
            raise TrafficLightError(f'traffic ONNX inference failed: {exc}') from exc
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise TrafficLightError('traffic ONNX must return exactly one output')
        box = decode_detection_box(
            outputs[0],
            frame_height=height,
            frame_width=width,
            confidence_threshold=self._confidence_threshold,
        )
        if box is None:
            return None
        return LampReading(
            scores=lamp_scores(
                bgr_frame,
                box,
                percentile=self._percentile,
            ),
            bbox_width=box.width,
            confidence=box.confidence,
        )


class TrafficClassifierDetector:
    """Run team YOLO box detection followed by the CNN signal classifier."""

    _CLASS_ORDER = (
        SignalClass.RED,
        SignalClass.YELLOW,
        SignalClass.LEFT_GREEN,
        SignalClass.STRAIGHT_GREEN,
    )
    _IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        *,
        yolo_session: _OnnxSession,
        classifier_session: _OnnxSession,
        confidence_threshold: float = 0.25,
        crop_padding: float = 0.15,
    ) -> None:
        if not math.isfinite(crop_padding) or crop_padding < 0.0:
            raise ValueError('classifier crop padding must be finite and non-negative')
        self._yolo_session = yolo_session
        self._classifier_session = classifier_session
        self._confidence_threshold = float(confidence_threshold)
        self._crop_padding = float(crop_padding)

    def read_signal(self, bgr_frame: np.ndarray) -> SignalReading | None:
        inspection = self.inspect_signal(bgr_frame)
        return None if inspection is None else inspection.reading

    def inspect_signal(self, bgr_frame: np.ndarray) -> SignalInspection | None:
        """Return the exact runtime reading plus GUI diagnostic details."""
        _validate_bgr_frame(bgr_frame)
        height, width = bgr_frame.shape[:2]
        resized = cv2.resize(bgr_frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        yolo_tensor = np.expand_dims(np.transpose(rgb, (2, 0, 1)), 0)
        try:
            outputs = self._yolo_session.run(None, {'images': yolo_tensor})
        except Exception as exc:  # noqa: BLE001 - runtime boundary
            raise TrafficLightError(f'traffic ONNX inference failed: {exc}') from exc
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise TrafficLightError('traffic ONNX must return exactly one output')
        box = decode_detection_box(
            outputs[0],
            frame_height=height,
            frame_width=width,
            confidence_threshold=self._confidence_threshold,
        )
        if box is None:
            return None
        crop_bounds = self._crop_bounds(bgr_frame, box)
        crop = bgr_frame[
            crop_bounds.y1:crop_bounds.y2,
            crop_bounds.x1:crop_bounds.x2,
        ]
        classifier_tensor = self._classifier_tensor(crop)
        try:
            outputs = self._classifier_session.run(
                None,
                {'image': classifier_tensor},
            )
        except Exception as exc:  # noqa: BLE001 - runtime boundary
            raise TrafficLightError(
                f'traffic classifier ONNX inference failed: {exc}'
            ) from exc
        signal_class, probabilities = _decode_classifier_logits(outputs)
        reading = SignalReading(
            signal_class=signal_class,
            probability=max(probabilities),
            bbox_width=box.width,
            confidence=box.confidence,
        )
        return SignalInspection(
            reading=reading,
            bbox=box,
            crop_bounds=crop_bounds,
            probabilities=probabilities,
        )

    def _crop_bounds(
        self,
        frame: np.ndarray,
        box: DetectionBox,
    ) -> ImageBounds:
        pad_x = int(box.width * self._crop_padding)
        pad_y = int(box.height * self._crop_padding)
        x1 = max(0, box.x - pad_x)
        y1 = max(0, box.y - pad_y)
        x2 = min(frame.shape[1], box.x + box.width + pad_x)
        y2 = min(frame.shape[0], box.y + box.height + pad_y)
        if x2 <= x1 or y2 <= y1:
            raise TrafficLightError('traffic classifier crop is empty')
        return ImageBounds(x1=x1, y1=y1, x2=x2, y2=y2)

    def _classifier_tensor(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (96, 48), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._IMAGE_MEAN) / self._IMAGE_STD
        tensor = np.expand_dims(np.transpose(rgb, (2, 0, 1)), 0).astype(
            np.float32
        )
        if tensor.shape != (1, 3, 48, 96) or not np.isfinite(tensor).all():
            raise TrafficLightError('traffic classifier input is invalid')
        return tensor


def _decode_classifier_logits(
    outputs,
) -> tuple[SignalClass, tuple[float, float, float, float]]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise TrafficLightError('traffic classifier must return exactly one output')
    logits = outputs[0]
    if (
        not isinstance(logits, np.ndarray)
        or logits.shape != (1, 4)
        or logits.dtype.kind not in {'f'}
    ):
        raise TrafficLightError('traffic classifier output must be [1,4] float')
    if not np.isfinite(logits).all():
        raise TrafficLightError('traffic classifier output contains NaN or Inf')
    values = logits[0]
    shifted = values - np.max(values)
    exponential = np.exp(shifted)
    probabilities = exponential / np.sum(exponential)
    if not np.isfinite(probabilities).all():
        raise TrafficLightError('traffic classifier probabilities are invalid')
    index = int(np.argmax(probabilities))
    decoded = tuple(float(value) for value in probabilities)
    return TrafficClassifierDetector._CLASS_ORDER[index], decoded


class TrafficLampLatch:
    """Vote on lamp actions while retaining a confirmed red stop."""

    def __init__(
        self,
        *,
        bbox_width_min: int = 45,
        bbox_width_max: int = 200,
        red_consecutive_reads: int = 5,
        left_consecutive_reads: int = 5,
        straight_consecutive_reads: int = 5,
    ) -> None:
        if bbox_width_min < 1 or bbox_width_max < bbox_width_min:
            raise ValueError('invalid traffic bbox width gate')
        if any(
            value < 1
            for value in (
                red_consecutive_reads,
                left_consecutive_reads,
                straight_consecutive_reads,
            )
        ):
            raise ValueError('signal consecutive reads must be positive')
        self._bbox_width_min = bbox_width_min
        self._bbox_width_max = bbox_width_max
        self._consecutive_reads = {
            LampAction.RED: red_consecutive_reads,
            LampAction.LEFT: left_consecutive_reads,
            LampAction.STRAIGHT: straight_consecutive_reads,
        }
        self._candidate: LampAction | None = None
        self._candidate_reads = 0
        self._red_latched = False

    @property
    def red_latched(self) -> bool:
        return self._red_latched

    @property
    def red_reads(self) -> int:
        return (
            self._candidate_reads
            if self._candidate == LampAction.RED
            else 0
        )

    def reset(self) -> None:
        self._candidate = None
        self._candidate_reads = 0
        self._red_latched = False

    def observe(self, reading: LampReading | None) -> LampAction:
        raw = self._classify(reading)
        if raw == LampAction.UNKNOWN:
            self._candidate = None
            self._candidate_reads = 0
            return (
                LampAction.RED
                if self._red_latched
                else LampAction.UNKNOWN
            )

        if raw == self._candidate:
            self._candidate_reads += 1
        else:
            self._candidate = raw
            self._candidate_reads = 1
        required = self._consecutive_reads[raw]
        self._candidate_reads = min(self._candidate_reads, required)
        confirmed = self._candidate_reads >= required

        if self._red_latched:
            if raw == LampAction.RED or not confirmed:
                return LampAction.RED
            self._red_latched = False
            return raw
        if not confirmed:
            return LampAction.UNKNOWN
        if raw == LampAction.RED:
            self._red_latched = True
        return raw

    def _classify(self, reading: LampReading | None) -> LampAction:
        if reading is None or not (
            self._bbox_width_min
            <= reading.bbox_width
            <= self._bbox_width_max
        ):
            return LampAction.UNKNOWN
        scores = reading.scores
        if len(scores) != 4 or not all(math.isfinite(value) for value in scores):
            raise TrafficLightError('lamp reading must contain four finite scores')
        threshold = (min(scores) + max(scores)) / 2.0
        lit = tuple(value > threshold for value in scores)
        if lit[0]:
            return LampAction.RED
        if lit[2] and lit[3]:
            return LampAction.LEFT
        if lit[3] and not lit[0]:
            return LampAction.STRAIGHT
        return LampAction.UNKNOWN


class TrafficSignalLatch:
    """Vote raw CNN classes while retaining confirmed red/yellow stops."""

    def __init__(
        self,
        *,
        bbox_width_min: int = 45,
        bbox_width_max: int = 200,
        consecutive_reads: int = 2,
    ) -> None:
        if bbox_width_min < 1 or bbox_width_max < bbox_width_min:
            raise ValueError('invalid traffic bbox width gate')
        if consecutive_reads < 1:
            raise ValueError('signal consecutive reads must be positive')
        self._bbox_width_min = bbox_width_min
        self._bbox_width_max = bbox_width_max
        self._consecutive_reads = consecutive_reads
        self._candidate = SignalClass.UNKNOWN
        self._candidate_reads = 0
        self._stop_latched = False

    @property
    def stop_latched(self) -> bool:
        return self._stop_latched

    @property
    def snapshot(self) -> TrafficSignalLatchSnapshot:
        return TrafficSignalLatchSnapshot(
            candidate=self._candidate,
            candidate_reads=self._candidate_reads,
            required_reads=self._consecutive_reads,
            stop_latched=self._stop_latched,
        )

    def reset(self) -> None:
        self._candidate = SignalClass.UNKNOWN
        self._candidate_reads = 0
        self._stop_latched = False

    def release_stop_latch(self) -> None:
        """Clear a confirmed stop after the mission's YOLO-loss fallback."""
        self.reset()

    def observe(self, reading: SignalReading | None) -> LampAction:
        raw = self._raw_class(reading)
        if raw == SignalClass.UNKNOWN:
            self._candidate = SignalClass.UNKNOWN
            self._candidate_reads = 0
            return LampAction.RED if self._stop_latched else LampAction.UNKNOWN
        if raw == self._candidate:
            self._candidate_reads += 1
        else:
            self._candidate = raw
            self._candidate_reads = 1
        self._candidate_reads = min(
            self._candidate_reads,
            self._consecutive_reads,
        )
        confirmed = self._candidate_reads >= self._consecutive_reads
        action = _signal_class_action(raw)
        if self._stop_latched:
            if action == LampAction.RED or not confirmed:
                return LampAction.RED
            self._stop_latched = False
            return action
        if not confirmed:
            return LampAction.UNKNOWN
        if action == LampAction.RED:
            self._stop_latched = True
        return action

    def _raw_class(self, reading: SignalReading | None) -> SignalClass:
        if reading is None or not (
            self._bbox_width_min <= reading.bbox_width <= self._bbox_width_max
        ):
            return SignalClass.UNKNOWN
        if (
            reading.signal_class == SignalClass.UNKNOWN
            or not math.isfinite(reading.probability)
            or not math.isfinite(reading.confidence)
        ):
            raise TrafficLightError('traffic classifier reading is invalid')
        return reading.signal_class


def _signal_class_action(signal_class: SignalClass) -> LampAction:
    if signal_class in {SignalClass.RED, SignalClass.YELLOW}:
        return LampAction.RED
    if signal_class == SignalClass.LEFT_GREEN:
        return LampAction.LEFT
    if signal_class == SignalClass.STRAIGHT_GREEN:
        return LampAction.STRAIGHT
    return LampAction.UNKNOWN


def _validate_bgr_frame(frame: np.ndarray) -> None:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] < 1
        or frame.shape[1] < 1
    ):
        raise TrafficLightError('traffic detector frame must be uint8 BGR')
