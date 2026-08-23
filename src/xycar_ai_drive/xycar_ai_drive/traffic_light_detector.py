"""YOLO traffic-light decoding and the fail-closed lamp latch contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import cv2
import numpy as np
from PIL import Image as PilImage

_PIL_BILINEAR = getattr(PilImage, 'Resampling', PilImage).BILINEAR


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
    STOP = 'STOP'
    STRAIGHT = 'STRAIGHT'
    LEFT = 'LEFT'


@dataclass(frozen=True)
class DetectionBox:
    x: float
    y: float
    width: float
    height: float
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
    bbox_width: float
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
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class TrafficClassifierInferencePlan:
    """Select one YOLO refresh or cached-box CNN classification."""

    run_detector: bool
    classification_box: DetectionBox | None
    detector_frame_span: int


class TrafficClassifierCadence:
    """Search every N frames, then classify a cached bbox every frame."""

    def __init__(
        self,
        *,
        detector_every_n_frames: int,
        classification_every_n_frames_after_detection: int,
        reuse_detected_bbox_between_yolo_frames: bool,
    ) -> None:
        if (
            detector_every_n_frames < 1
            or classification_every_n_frames_after_detection < 1
            or classification_every_n_frames_after_detection
            > detector_every_n_frames
        ):
            raise ValueError('invalid traffic classifier cadence')
        self._detector_every_n_frames = detector_every_n_frames
        self._classification_every_n_frames = (
            classification_every_n_frames_after_detection
        )
        self._reuse_detected_bbox = bool(
            reuse_detected_bbox_between_yolo_frames
        )
        self.reset()

    def reset(self, *, frame_sequence: int = 0) -> None:
        if frame_sequence < 0:
            raise ValueError('frame sequence must not be negative')
        self._last_detector_sequence = frame_sequence
        self._last_classification_sequence = frame_sequence
        self._tracked_box: DetectionBox | None = None

    def plan(self, *, frame_sequence: int) -> TrafficClassifierInferencePlan:
        if frame_sequence <= self._last_detector_sequence:
            return TrafficClassifierInferencePlan(False, None, 0)
        detector_frame_span = frame_sequence - self._last_detector_sequence
        if detector_frame_span >= self._detector_every_n_frames:
            return TrafficClassifierInferencePlan(
                run_detector=True,
                classification_box=None,
                detector_frame_span=detector_frame_span,
            )
        if (
            self._reuse_detected_bbox
            and self._tracked_box is not None
            and frame_sequence - self._last_classification_sequence
            >= self._classification_every_n_frames
        ):
            return TrafficClassifierInferencePlan(
                run_detector=False,
                classification_box=self._tracked_box,
                detector_frame_span=0,
            )
        return TrafficClassifierInferencePlan(False, None, 0)

    def observe_detection(
        self,
        *,
        frame_sequence: int,
        box: DetectionBox | None,
    ) -> None:
        if frame_sequence <= self._last_detector_sequence:
            raise ValueError('detector frame sequence must increase')
        self._last_detector_sequence = frame_sequence
        self._last_classification_sequence = frame_sequence
        self._tracked_box = box if self._reuse_detected_bbox else None

    def observe_classification(self, *, frame_sequence: int) -> None:
        if (
            self._tracked_box is None
            or frame_sequence <= self._last_classification_sequence
        ):
            raise ValueError('invalid cached-box classification sequence')
        self._last_classification_sequence = frame_sequence


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometry needed to map a centered 640 letterbox back to a frame."""

    scale: float
    pad_x: int
    pad_y: int


@dataclass(frozen=True)
class TrafficSignalLatchSnapshot:
    """Read-only state of the deployed action-specific signal vote."""

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
    letterbox_transform: LetterboxTransform | None = None,
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
    if letterbox_transform is None:
        scale_x = frame_width / 640.0
        scale_y = frame_height / 640.0
        x1 = (center_x - width / 2.0) * scale_x
        y1 = (center_y - height / 2.0) * scale_y
        decoded_width = width * scale_x
        decoded_height = height * scale_y
    else:
        scale = letterbox_transform.scale
        if not math.isfinite(scale) or scale <= 0.0:
            raise TrafficLightError('letterbox scale must be finite and positive')
        x1_float = (center_x - width / 2.0 - letterbox_transform.pad_x) / scale
        y1_float = (center_y - height / 2.0 - letterbox_transform.pad_y) / scale
        x2_float = (center_x + width / 2.0 - letterbox_transform.pad_x) / scale
        y2_float = (center_y + height / 2.0 - letterbox_transform.pad_y) / scale
        x1_float = min(max(x1_float, 0.0), float(frame_width))
        y1_float = min(max(y1_float, 0.0), float(frame_height))
        x2_float = min(max(x2_float, 0.0), float(frame_width))
        y2_float = min(max(y2_float, 0.0), float(frame_height))
        x1 = x1_float
        y1 = y1_float
        decoded_width = x2_float - x1_float
        decoded_height = y2_float - y1_float
        return DetectionBox(
            x=x1,
            y=y1,
            width=decoded_width,
            height=decoded_height,
            confidence=float(candidate_confidence[index]),
        )
    return DetectionBox(
        x=int(x1),
        y=int(y1),
        width=int(decoded_width),
        height=int(decoded_height),
        confidence=float(candidate_confidence[index]),
    )


def letterbox_640_bgr_frame(
    bgr_frame: np.ndarray,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Apply Ultralytics-compatible centered 640 letterbox preprocessing."""
    _validate_bgr_frame(bgr_frame)
    frame_height, frame_width = bgr_frame.shape[:2]
    scale = min(640.0 / frame_height, 640.0 / frame_width)
    resized_width = int(round(frame_width * scale))
    resized_height = int(round(frame_height * scale))
    resized = cv2.resize(
        bgr_frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    padding_width = 640 - resized_width
    padding_height = 640 - resized_height
    left = int(round(padding_width / 2.0 - 0.1))
    right = int(round(padding_width / 2.0 + 0.1))
    top = int(round(padding_height / 2.0 - 0.1))
    bottom = int(round(padding_height / 2.0 + 0.1))
    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if letterboxed.shape != (640, 640, 3):
        raise TrafficLightError('traffic letterbox must produce 640 square')
    return letterboxed, LetterboxTransform(scale=scale, pad_x=left, pad_y=top)


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

    _LEGACY_CLASS_ORDER = (
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
        detector_preprocessing: str = (
            'resize_640_bgr_to_rgb_float32_nchw_div255'
        ),
        classifier_input_height: int = 48,
        classifier_input_width: int = 96,
        classifier_classes: tuple[str, ...] = (
            'red',
            'yellow',
            'left_green',
            'straight_green',
        ),
        classifier_probability_threshold: float | None = None,
        classifier_interpolation: str = 'area',
    ) -> None:
        if not math.isfinite(crop_padding) or crop_padding < 0.0:
            raise ValueError('classifier crop padding must be finite and non-negative')
        self._yolo_session = yolo_session
        self._classifier_session = classifier_session
        self._confidence_threshold = float(confidence_threshold)
        self._crop_padding = float(crop_padding)
        if detector_preprocessing not in {
            'resize_640_bgr_to_rgb_float32_nchw_div255',
            'letterbox_640_center_pad114_bgr_to_rgb_float32_nchw_div255',
        }:
            raise ValueError('unsupported traffic detector preprocessing')
        if classifier_input_height < 1 or classifier_input_width < 1:
            raise ValueError('classifier input dimensions must be positive')
        try:
            class_order = tuple(SignalClass(value) for value in classifier_classes)
        except ValueError as exc:
            raise ValueError('unsupported traffic classifier class') from exc
        if (
            not class_order
            or SignalClass.UNKNOWN in class_order
            or len(set(class_order)) != len(class_order)
        ):
            raise ValueError('traffic classifier classes must be unique')
        if classifier_probability_threshold is not None and (
            not math.isfinite(classifier_probability_threshold)
            or not 0.0 < classifier_probability_threshold < 1.0
        ):
            raise ValueError('classifier probability threshold must be in (0,1)')
        if classifier_interpolation not in {
            'area',
            'pillow_bilinear_antialias',
        }:
            raise ValueError('unsupported traffic classifier interpolation')
        self._detector_preprocessing = detector_preprocessing
        self._classifier_input_height = classifier_input_height
        self._classifier_input_width = classifier_input_width
        self._classifier_class_order = class_order
        self._classifier_probability_threshold = (
            classifier_probability_threshold
        )
        self._classifier_interpolation = classifier_interpolation

    def read_signal(self, bgr_frame: np.ndarray) -> SignalReading | None:
        inspection = self.inspect_signal(bgr_frame)
        return None if inspection is None else inspection.reading

    def inspect_signal(self, bgr_frame: np.ndarray) -> SignalInspection | None:
        """Return the exact runtime reading plus GUI diagnostic details."""
        _validate_bgr_frame(bgr_frame)
        height, width = bgr_frame.shape[:2]
        if (
            self._detector_preprocessing
            == 'letterbox_640_center_pad114_bgr_to_rgb_float32_nchw_div255'
        ):
            resized, letterbox_transform = letterbox_640_bgr_frame(bgr_frame)
        else:
            resized = cv2.resize(bgr_frame, (640, 640))
            letterbox_transform = None
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
            letterbox_transform=letterbox_transform,
        )
        if box is None:
            return None
        return self.classify_signal_box(bgr_frame, box)

    def classify_signal_box(
        self,
        bgr_frame: np.ndarray,
        box: DetectionBox,
    ) -> SignalInspection:
        """Classify the current frame using a recently detected YOLO bbox."""
        _validate_bgr_frame(bgr_frame)
        if not all(
            math.isfinite(value)
            for value in (
                box.x,
                box.y,
                box.width,
                box.height,
                box.confidence,
            )
        ):
            raise TrafficLightError('traffic detection bbox is invalid')
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
        signal_class, probabilities = _decode_classifier_logits(
            outputs,
            class_order=self._classifier_class_order,
            minimum_probability=self._classifier_probability_threshold,
        )
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
        if (
            self._detector_preprocessing
            == 'letterbox_640_center_pad114_bgr_to_rgb_float32_nchw_div255'
        ):
            pad_x = box.width * self._crop_padding
            pad_y = box.height * self._crop_padding
            x1 = max(0, math.floor(box.x - pad_x))
            y1 = max(0, math.floor(box.y - pad_y))
            x2 = min(
                frame.shape[1],
                math.ceil(box.x + box.width + pad_x),
            )
            y2 = min(
                frame.shape[0],
                math.ceil(box.y + box.height + pad_y),
            )
        else:
            pad_x = int(box.width * self._crop_padding)
            pad_y = int(box.height * self._crop_padding)
            x1 = max(0, int(box.x) - pad_x)
            y1 = max(0, int(box.y) - pad_y)
            x2 = min(frame.shape[1], int(box.x + box.width) + pad_x)
            y2 = min(frame.shape[0], int(box.y + box.height) + pad_y)
        if x2 <= x1 or y2 <= y1:
            raise TrafficLightError('traffic classifier crop is empty')
        return ImageBounds(x1=x1, y1=y1, x2=x2, y2=y2)

    def _classifier_tensor(self, crop: np.ndarray) -> np.ndarray:
        if self._classifier_interpolation == 'area':
            resized = cv2.resize(
                crop,
                (self._classifier_input_width, self._classifier_input_height),
                interpolation=cv2.INTER_AREA,
            )
            rgb_uint8 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized_image = PilImage.fromarray(rgb_crop).resize(
                (self._classifier_input_width, self._classifier_input_height),
                resample=_PIL_BILINEAR,
            )
            rgb_uint8 = np.asarray(resized_image)
        rgb = rgb_uint8.astype(np.float32) / 255.0
        rgb = (rgb - self._IMAGE_MEAN) / self._IMAGE_STD
        tensor = np.expand_dims(np.transpose(rgb, (2, 0, 1)), 0).astype(
            np.float32
        )
        if tensor.shape != (
            1,
            3,
            self._classifier_input_height,
            self._classifier_input_width,
        ) or not np.isfinite(tensor).all():
            raise TrafficLightError('traffic classifier input is invalid')
        return tensor


def _decode_classifier_logits(
    outputs,
    *,
    class_order: tuple[SignalClass, ...] = TrafficClassifierDetector._LEGACY_CLASS_ORDER,
    minimum_probability: float | None = None,
) -> tuple[SignalClass, tuple[float, ...]]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise TrafficLightError('traffic classifier must return exactly one output')
    logits = outputs[0]
    if (
        not isinstance(logits, np.ndarray)
        or logits.shape != (1, len(class_order))
        or logits.dtype.kind not in {'f'}
    ):
        raise TrafficLightError(
            'traffic classifier output must be '
            f'[1,{len(class_order)}] float'
        )
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
    signal_class = class_order[index]
    if (
        minimum_probability is not None
        and decoded[index] < minimum_probability
    ):
        signal_class = SignalClass.UNKNOWN
    return signal_class, decoded


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
        red_consecutive_reads: int | None = None,
        left_consecutive_reads: int | None = None,
        straight_consecutive_reads: int | None = None,
    ) -> None:
        if bbox_width_min < 1 or bbox_width_max < bbox_width_min:
            raise ValueError('invalid traffic bbox width gate')
        common_reads = int(consecutive_reads)
        red_reads = (
            common_reads
            if red_consecutive_reads is None
            else int(red_consecutive_reads)
        )
        left_reads = (
            common_reads
            if left_consecutive_reads is None
            else int(left_consecutive_reads)
        )
        straight_reads = (
            common_reads
            if straight_consecutive_reads is None
            else int(straight_consecutive_reads)
        )
        if any(value < 1 for value in (red_reads, left_reads, straight_reads)):
            raise ValueError('signal consecutive reads must be positive')
        self._bbox_width_min = bbox_width_min
        self._bbox_width_max = bbox_width_max
        self._consecutive_reads = {
            LampAction.RED: red_reads,
            LampAction.LEFT: left_reads,
            LampAction.STRAIGHT: straight_reads,
        }
        self._candidate = SignalClass.UNKNOWN
        self._candidate_reads = 0
        self._stop_latched = False

    @property
    def stop_latched(self) -> bool:
        return self._stop_latched

    @property
    def snapshot(self) -> TrafficSignalLatchSnapshot:
        required_reads = self._required_reads(self._candidate)
        return TrafficSignalLatchSnapshot(
            candidate=self._candidate,
            candidate_reads=self._candidate_reads,
            required_reads=required_reads,
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
        required_reads = self._required_reads(raw)
        self._candidate_reads = min(
            self._candidate_reads,
            required_reads,
        )
        confirmed = self._candidate_reads >= required_reads
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
            not math.isfinite(reading.probability)
            or not math.isfinite(reading.confidence)
        ):
            raise TrafficLightError('traffic classifier reading is invalid')
        if reading.signal_class == SignalClass.UNKNOWN:
            return SignalClass.UNKNOWN
        return reading.signal_class

    def _required_reads(self, signal_class: SignalClass) -> int:
        action = _signal_class_action(signal_class)
        return self._consecutive_reads.get(action, 0)


def _signal_class_action(signal_class: SignalClass) -> LampAction:
    if signal_class in {
        SignalClass.RED,
        SignalClass.YELLOW,
        SignalClass.STOP,
    }:
        return LampAction.RED
    if signal_class in {SignalClass.LEFT_GREEN, SignalClass.LEFT}:
        return LampAction.LEFT
    if signal_class in {SignalClass.STRAIGHT_GREEN, SignalClass.STRAIGHT}:
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
