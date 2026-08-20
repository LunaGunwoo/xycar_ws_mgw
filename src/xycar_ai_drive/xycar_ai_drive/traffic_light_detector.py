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


class TrafficLampLatch:
    """Apply width gating, relative brightness and the three-read red latch."""

    def __init__(
        self,
        *,
        bbox_width_min: int = 45,
        bbox_width_max: int = 200,
        red_consecutive_reads: int = 3,
    ) -> None:
        if bbox_width_min < 1 or bbox_width_max < bbox_width_min:
            raise ValueError('invalid traffic bbox width gate')
        if red_consecutive_reads < 1:
            raise ValueError('red_consecutive_reads must be positive')
        self._bbox_width_min = bbox_width_min
        self._bbox_width_max = bbox_width_max
        self._red_consecutive_reads = red_consecutive_reads
        self._red_reads = 0
        self._red_latched = False

    @property
    def red_latched(self) -> bool:
        return self._red_latched

    @property
    def red_reads(self) -> int:
        return self._red_reads

    def reset(self) -> None:
        self._red_reads = 0
        self._red_latched = False

    def observe(self, reading: LampReading | None) -> LampAction:
        raw = self._classify(reading)
        if raw == LampAction.RED:
            self._red_reads += 1
            if self._red_reads >= self._red_consecutive_reads:
                self._red_latched = True
            return LampAction.RED if self._red_latched else LampAction.UNKNOWN
        if raw in {LampAction.LEFT, LampAction.STRAIGHT}:
            self._red_reads = 0
            self._red_latched = False
            return raw
        self._red_reads = 0
        return LampAction.RED if self._red_latched else LampAction.UNKNOWN

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
