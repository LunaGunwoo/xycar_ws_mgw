"""Versioned, image-free diagnostics for the traffic-shortcut monitor."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

SIGNAL_DEBUG_SCHEMA_VERSION = 2
SIGNAL_DEBUG_SOURCES = frozenset({'YOLO_CNN', 'CACHED_CNN', 'YOLO_NO_BOX'})
SHORTCUT_STATUSES = frozenset(
    {
        'DISABLED',
        'READY',
        'TRANSITION_STOP',
        'ACTIVE',
        'COMPLETED_THIS_ACTIVATION',
        'FAULT',
    }
)


class SignalDebugContractError(ValueError):
    """Raised when a signal-debug payload violates the monitor contract."""


@dataclass(frozen=True)
class SignalDebugBbox:
    x: float
    y: float
    width: float
    height: float
    confidence: float


@dataclass(frozen=True)
class SignalDebugCrop:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class SignalDebugSnapshot:
    """One classifier result tied to the exact ROS camera timestamp."""

    schema_version: int
    bundle_id: str
    frame_sequence: int
    stamp_sec: int
    stamp_nanosec: int
    source: str
    vote_updated: bool
    raw_class: str
    final_action: str
    class_labels: tuple[str, ...]
    probabilities: tuple[float, ...] | None
    bbox: SignalDebugBbox | None
    crop: SignalDebugCrop | None
    width_gate_accepted: bool
    candidate: str
    candidate_reads: int
    required_reads: int
    phase: str
    mission_state: str
    shortcut_status: str
    detector_inference_ms: float

    @property
    def stamp_key(self) -> tuple[int, int] | None:
        if self.stamp_sec == 0 and self.stamp_nanosec == 0:
            return None
        return self.stamp_sec, self.stamp_nanosec


def encode_signal_debug(snapshot: SignalDebugSnapshot) -> str:
    """Validate and serialize one compact JSON payload."""
    validated = _validated_snapshot(asdict(snapshot))
    return json.dumps(
        _snapshot_to_dict(validated),
        allow_nan=False,
        separators=(',', ':'),
        sort_keys=True,
    )


def decode_signal_debug(payload: str) -> SignalDebugSnapshot:
    """Parse and strictly validate one signal-debug JSON payload."""
    if not isinstance(payload, str) or not payload:
        raise SignalDebugContractError('signal-debug payload must be text')
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SignalDebugContractError(
            f'signal-debug payload is not valid JSON: {exc}'
        ) from exc
    return _validated_snapshot(value)


def _snapshot_to_dict(snapshot: SignalDebugSnapshot) -> dict[str, object]:
    value = asdict(snapshot)
    value['class_labels'] = list(snapshot.class_labels)
    value['probabilities'] = (
        None
        if snapshot.probabilities is None
        else list(snapshot.probabilities)
    )
    return value


def _validated_snapshot(value: object) -> SignalDebugSnapshot:
    if not isinstance(value, dict):
        raise SignalDebugContractError('signal-debug root must be an object')
    required = {
        'schema_version',
        'bundle_id',
        'frame_sequence',
        'stamp_sec',
        'stamp_nanosec',
        'source',
        'vote_updated',
        'raw_class',
        'final_action',
        'class_labels',
        'probabilities',
        'bbox',
        'crop',
        'width_gate_accepted',
        'candidate',
        'candidate_reads',
        'required_reads',
        'phase',
        'mission_state',
        'shortcut_status',
        'detector_inference_ms',
    }
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise SignalDebugContractError(
            f'signal-debug fields mismatch: missing={missing}, extra={extra}'
        )

    schema_version = _integer(value['schema_version'], 'schema_version')
    if schema_version != SIGNAL_DEBUG_SCHEMA_VERSION:
        raise SignalDebugContractError(
            f'unsupported signal-debug schema: {schema_version}'
        )
    bundle_id = _text(value['bundle_id'], 'bundle_id')
    frame_sequence = _integer(value['frame_sequence'], 'frame_sequence')
    stamp_sec = _integer(value['stamp_sec'], 'stamp_sec')
    stamp_nanosec = _integer(value['stamp_nanosec'], 'stamp_nanosec')
    if frame_sequence < 1:
        raise SignalDebugContractError('frame_sequence must be positive')
    if stamp_sec < 0 or not 0 <= stamp_nanosec < 1_000_000_000:
        raise SignalDebugContractError('camera timestamp is out of range')

    source = _text(value['source'], 'source')
    if source not in SIGNAL_DEBUG_SOURCES:
        raise SignalDebugContractError(f'unsupported signal source: {source}')
    vote_updated = _boolean(value['vote_updated'], 'vote_updated')
    raw_class = _text(value['raw_class'], 'raw_class')
    final_action = _text(value['final_action'], 'final_action')
    candidate = _text(value['candidate'], 'candidate')
    phase = _text(value['phase'], 'phase')
    mission_state = _text(value['mission_state'], 'mission_state')
    shortcut_status = _text(value['shortcut_status'], 'shortcut_status')
    if shortcut_status not in SHORTCUT_STATUSES:
        raise SignalDebugContractError(
            f'unsupported shortcut status: {shortcut_status}'
        )

    labels_value = value['class_labels']
    if not isinstance(labels_value, (list, tuple)) or len(labels_value) != 3:
        raise SignalDebugContractError('class_labels must contain 3 values')
    class_labels = tuple(
        _text(label, f'class_labels[{index}]')
        for index, label in enumerate(labels_value)
    )
    if len(set(class_labels)) != len(class_labels):
        raise SignalDebugContractError('class_labels must be unique')

    probabilities_value = value['probabilities']
    probabilities: tuple[float, ...] | None
    if probabilities_value is None:
        probabilities = None
    elif isinstance(probabilities_value, (list, tuple)) and len(
        probabilities_value
    ) == len(class_labels):
        probabilities = tuple(
            _finite_float(
                probability,
                f'probabilities[{index}]',
                minimum=0.0,
                maximum=1.0,
            )
            for index, probability in enumerate(probabilities_value)
        )
    else:
        raise SignalDebugContractError(
            'probabilities must be null or match class_labels'
        )

    bbox = _bbox(value['bbox'])
    crop = _crop(value['crop'])
    width_gate_accepted = _boolean(
        value['width_gate_accepted'],
        'width_gate_accepted',
    )
    if source == 'YOLO_NO_BOX':
        if any(item is not None for item in (probabilities, bbox, crop)):
            raise SignalDebugContractError(
                'YOLO_NO_BOX must not contain classification geometry'
            )
        if width_gate_accepted:
            raise SignalDebugContractError(
                'YOLO_NO_BOX cannot pass the width gate'
            )
    elif any(item is None for item in (probabilities, bbox, crop)):
        raise SignalDebugContractError(
            f'{source} requires probabilities, bbox and crop'
        )
    candidate_reads = _integer(value['candidate_reads'], 'candidate_reads')
    required_reads = _integer(value['required_reads'], 'required_reads')
    if candidate_reads < 0 or required_reads < 0:
        raise SignalDebugContractError('vote counts must not be negative')
    if required_reads and candidate_reads > required_reads:
        raise SignalDebugContractError(
            'candidate_reads must not exceed required_reads'
        )
    detector_inference_ms = _finite_float(
        value['detector_inference_ms'],
        'detector_inference_ms',
        minimum=0.0,
    )

    return SignalDebugSnapshot(
        schema_version=schema_version,
        bundle_id=bundle_id,
        frame_sequence=frame_sequence,
        stamp_sec=stamp_sec,
        stamp_nanosec=stamp_nanosec,
        source=source,
        vote_updated=vote_updated,
        raw_class=raw_class,
        final_action=final_action,
        class_labels=class_labels,
        probabilities=probabilities,
        bbox=bbox,
        crop=crop,
        width_gate_accepted=width_gate_accepted,
        candidate=candidate,
        candidate_reads=candidate_reads,
        required_reads=required_reads,
        phase=phase,
        mission_state=mission_state,
        shortcut_status=shortcut_status,
        detector_inference_ms=detector_inference_ms,
    )


def _bbox(value: object) -> SignalDebugBbox | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        'x',
        'y',
        'width',
        'height',
        'confidence',
    }:
        raise SignalDebugContractError('bbox fields mismatch')
    bbox = SignalDebugBbox(
        x=_finite_float(value['x'], 'bbox.x'),
        y=_finite_float(value['y'], 'bbox.y'),
        width=_finite_float(value['width'], 'bbox.width', minimum=0.0),
        height=_finite_float(value['height'], 'bbox.height', minimum=0.0),
        confidence=_finite_float(
            value['confidence'],
            'bbox.confidence',
            minimum=0.0,
            maximum=1.0,
        ),
    )
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        raise SignalDebugContractError('bbox dimensions must be positive')
    return bbox


def _crop(value: object) -> SignalDebugCrop | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'x1', 'y1', 'x2', 'y2'}:
        raise SignalDebugContractError('crop fields mismatch')
    crop = SignalDebugCrop(
        x1=_integer(value['x1'], 'crop.x1'),
        y1=_integer(value['y1'], 'crop.y1'),
        x2=_integer(value['x2'], 'crop.x2'),
        y2=_integer(value['y2'], 'crop.y2'),
    )
    if crop.x1 < 0 or crop.y1 < 0 or crop.x2 <= crop.x1 or crop.y2 <= crop.y1:
        raise SignalDebugContractError('crop bounds are invalid')
    return crop


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SignalDebugContractError(f'{name} must be non-empty text')
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise SignalDebugContractError(f'{name} must be boolean')
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SignalDebugContractError(f'{name} must be an integer')
    return value


def _finite_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignalDebugContractError(f'{name} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise SignalDebugContractError(f'{name} must be finite')
    if minimum is not None and result < minimum:
        raise SignalDebugContractError(f'{name} is below its minimum')
    if maximum is not None and result > maximum:
        raise SignalDebugContractError(f'{name} exceeds its maximum')
    return result
