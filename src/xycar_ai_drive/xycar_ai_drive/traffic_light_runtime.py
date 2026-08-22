"""Shared ONNX Runtime construction for deployed traffic-light detection."""

from __future__ import annotations

import numpy as np

from xycar_ai_drive.traffic_light_detector import (
    TrafficClassifierDetector,
    TrafficLightDetector,
)
from xycar_ai_drive.traffic_shortcut_artifact import (
    EXPECTED_NUMPY_VERSION,
    EXPECTED_ONNXRUNTIME_VERSION,
    TrafficShortcutBundle,
)


def create_onnx_detector(
    bundle: TrafficShortcutBundle,
) -> TrafficLightDetector | TrafficClassifierDetector:
    """Build detector sessions using the bundle's exact host contract."""
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise ValueError(
            f'host NumPy must be {EXPECTED_NUMPY_VERSION}, '
            f'got {np.__version__}'
        )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError(
            'onnxruntime is required for traffic detection'
        ) from exc
    if ort.__version__ != EXPECTED_ONNXRUNTIME_VERSION:
        raise ValueError(
            f'host ONNX Runtime must be {EXPECTED_ONNXRUNTIME_VERSION}, '
            f'got {ort.__version__}'
        )

    session = ort.InferenceSession(
        str(bundle.detector.model_path),
        providers=list(bundle.providers),
    )
    _validate_provider_order(session, bundle, label='traffic ONNX')
    _validate_model_metadata(
        session,
        input_name='images',
        input_shape=[1, 3, 640, 640],
        output_name='output0',
        output_shape=[1, 5, 8400],
        label='traffic ONNX',
    )
    if bundle.detector.mode == 'hsv_lamp':
        return TrafficLightDetector(
            session=session,
            confidence_threshold=bundle.detector.confidence_threshold,
            percentile=bundle.detector.percentile,
        )

    classifier_path = bundle.detector.classifier_model_path
    crop_padding = bundle.detector.classifier_crop_padding
    classifier_height = bundle.detector.classifier_input_height
    classifier_width = bundle.detector.classifier_input_width
    classifier_classes = bundle.detector.classifier_classes
    classifier_interpolation = bundle.detector.classifier_interpolation
    if (
        classifier_path is None
        or crop_padding is None
        or classifier_height is None
        or classifier_width is None
        or not classifier_classes
        or classifier_interpolation is None
    ):
        raise ValueError('classifier bundle is missing classifier contract')
    classifier = ort.InferenceSession(
        str(classifier_path),
        providers=list(bundle.providers),
    )
    _validate_provider_order(
        classifier,
        bundle,
        label='traffic classifier ONNX',
    )
    _validate_model_metadata(
        classifier,
        input_name='image',
        input_shape=[1, 3, classifier_height, classifier_width],
        output_name='logits',
        output_shape=[1, len(classifier_classes)],
        label='traffic classifier ONNX',
    )
    return TrafficClassifierDetector(
        yolo_session=session,
        classifier_session=classifier,
        confidence_threshold=bundle.detector.confidence_threshold,
        crop_padding=crop_padding,
        detector_preprocessing=bundle.detector.detector_preprocessing,
        classifier_input_height=classifier_height,
        classifier_input_width=classifier_width,
        classifier_classes=classifier_classes,
        classifier_probability_threshold=(
            bundle.detector.classifier_probability_threshold
        ),
        classifier_interpolation=classifier_interpolation,
    )


def _validate_provider_order(session, bundle, *, label: str) -> None:
    if tuple(session.get_providers()) != bundle.providers:
        raise ValueError(
            f'{label} active providers do not match CUDA then CPU contract'
        )


def _validate_model_metadata(
    session,
    *,
    input_name: str,
    input_shape: list[int],
    output_name: str,
    output_shape: list[int],
    label: str,
) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != input_name
        or inputs[0].type != 'tensor(float)'
        or list(inputs[0].shape) != input_shape
    ):
        raise ValueError(f'{label} input metadata mismatch')
    if (
        len(outputs) != 1
        or outputs[0].name != output_name
        or outputs[0].type != 'tensor(float)'
        or list(outputs[0].shape) != output_shape
    ):
        raise ValueError(f'{label} output metadata mismatch')
