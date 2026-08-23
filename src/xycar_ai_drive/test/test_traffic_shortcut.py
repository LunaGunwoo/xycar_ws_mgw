# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from collections import deque
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from xycar_ai_drive.artifact import ArtifactContractError
from xycar_ai_drive.traffic_light_detector import (
    DetectionBox,
    ImageBounds,
    LetterboxTransform,
    LampAction,
    LampReading,
    SignalClass,
    SignalInspection,
    SignalReading,
    TrafficClassifierCadence,
    TrafficClassifierDetector,
    TrafficLampLatch,
    TrafficLightDetector,
    TrafficLightError,
    TrafficSignalLatch,
    TrafficSignalLatchSnapshot,
    decode_detection_box,
    lamp_scores,
    letterbox_640_bgr_frame,
)
from xycar_ai_drive.traffic_light_viewer import (
    TrafficLightViewerResult,
    draw_signal_overlay,
)
from xycar_ai_drive.traffic_shortcut_fsm import (
    MissionState,
    PolicyChoice,
    TrafficShortcutFsm,
)
from xycar_ai_drive.traffic_shortcut_artifact import (
    EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID,
    EXPECTED_EXPANDED_SIGNAL_VOTE_BUNDLE_ID,
    EXPECTED_CLASSIFIER_BUNDLE_ID,
    EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID,
    EXPECTED_SHORTCUT_ARTIFACT_ID,
    EXPECTED_SIGNAL_VOTE_BUNDLE_ID,
    _expected_shortcut_artifact_id,
    _load_signal_vote_contract,
)
from xycar_ai_drive.traffic_shortcut_policy_node import (
    MissionDecision,
    TrafficShortcutPolicyNode,
    YoloMissingReleaseCounter,
)
from xycar_ai_drive.control import DriveCommand


class _FakeOnnxSession:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    def run(self, output_names, inputs):
        assert output_names is None
        self.inputs.append(inputs)
        return [self.output]


class _FakeClassifierSession(_FakeOnnxSession):
    pass


def _output_with_candidates(*candidates):
    output = np.zeros((1, 5, 8400), dtype=np.float32)
    for index, values in enumerate(candidates):
        output[0, :, index] = values
    return output


def _reading(scores, width=100):
    return LampReading(
        scores=tuple(float(value) for value in scores),
        bbox_width=width,
        confidence=0.9,
    )


def _signal(signal_class, width=100):
    return SignalReading(
        signal_class=signal_class,
        probability=0.9,
        bbox_width=width,
        confidence=0.9,
    )


def test_yolo_decode_selects_max_confidence_and_scales_bbox():
    output = _output_with_candidates(
        (100.0, 100.0, 40.0, 20.0, 0.4),
        (320.0, 160.0, 100.0, 40.0, 0.9),
    )
    box = decode_detection_box(
        output,
        frame_height=480,
        frame_width=640,
    )
    assert box == DetectionBox(
        x=270,
        y=105,
        width=100,
        height=30,
        confidence=pytest.approx(0.9),
    )
    assert decode_detection_box(
        np.zeros((1, 5, 8400), dtype=np.float32),
        frame_height=480,
        frame_width=640,
    ) is None


def test_yolo_letterbox_preserves_aspect_ratio_and_restores_bbox():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    letterboxed, transform = letterbox_640_bgr_frame(frame)

    assert letterboxed.shape == (640, 640, 3)
    assert transform == LetterboxTransform(scale=1.0, pad_x=0, pad_y=80)
    assert np.all(letterboxed[:80] == 114)
    output = _output_with_candidates(
        (140.0, 212.0, 80.0, 24.0, 0.9),
        (300.0, 300.0, 30.0, 10.0, 0.4),
    )
    assert decode_detection_box(
        output,
        frame_height=480,
        frame_width=640,
        letterbox_transform=transform,
    ) == DetectionBox(
        x=100,
        y=120,
        width=80,
        height=24,
        confidence=pytest.approx(0.9),
    )


def test_detector_preprocess_and_lamp_percentile_contract():
    output = _output_with_candidates((320.0, 320.0, 320.0, 80.0, 0.9))
    session = _FakeOnnxSession(output)
    detector = TrafficLightDetector(session=session)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[45:55, 25:35] = (0, 0, 250)
    frame[45:55, 75:85] = (0, 0, 20)
    frame[45:55, 125:135] = (0, 0, 180)
    frame[45:55, 175:185] = (0, 0, 220)
    reading = detector.read_lamp(frame)

    tensor = session.inputs[0]['images']
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    assert reading is not None
    assert reading.bbox_width == 100
    assert len(reading.scores) == 4

    direct = lamp_scores(
        frame,
        DetectionBox(0, 40, 200, 20, 0.9),
        percentile=80.0,
    )
    assert direct[0] > direct[1]
    assert direct[3] > direct[2]


def test_detector_rejects_wrong_shape_and_nonfinite_output():
    with pytest.raises(TrafficLightError, match=r'\[1,5,8400\]'):
        decode_detection_box(
            np.zeros((1, 6, 8400), dtype=np.float32),
            frame_height=480,
            frame_width=640,
        )


def test_classifier_detector_uses_yolo_box_padding_and_cnn_argmax():
    yolo = _FakeOnnxSession(
        _output_with_candidates((320.0, 240.0, 100.0, 40.0, 0.9))
    )
    classifier = _FakeClassifierSession(
        np.array([[0.0, 1.0, 4.0, 2.0]], dtype=np.float32)
    )
    detector = TrafficClassifierDetector(
        yolo_session=yolo,
        classifier_session=classifier,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[215:265, 255:385] = (10, 20, 30)
    inspection = detector.inspect_signal(frame)
    reading = detector.read_signal(frame)

    assert inspection is not None
    assert reading is not None
    assert reading == inspection.reading
    assert reading.signal_class == SignalClass.LEFT_GREEN
    assert reading.bbox_width == 100
    assert reading.probability > 0.8
    assert inspection.bbox == DetectionBox(
        x=270,
        y=165,
        width=100,
        height=30,
        confidence=pytest.approx(0.9),
    )
    assert inspection.crop_bounds == ImageBounds(
        x1=255,
        y1=161,
        x2=385,
        y2=199,
    )
    assert len(inspection.probabilities) == 4
    assert sum(inspection.probabilities) == pytest.approx(1.0)
    assert inspection.probabilities[2] == pytest.approx(
        reading.probability
    )
    tensor = classifier.inputs[0]['image']
    assert tensor.shape == (1, 3, 48, 96)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


@pytest.mark.parametrize(
    'logits, expected',
    [
        ([4, 0, 0, 0], SignalClass.RED),
        ([0, 4, 0, 0], SignalClass.YELLOW),
        ([0, 0, 4, 0], SignalClass.LEFT_GREEN),
        ([0, 0, 0, 4], SignalClass.STRAIGHT_GREEN),
    ],
)
def test_classifier_detector_decodes_all_signal_classes(logits, expected):
    yolo = _FakeOnnxSession(
        _output_with_candidates((320.0, 240.0, 100.0, 40.0, 0.9))
    )
    classifier = _FakeClassifierSession(
        np.array([logits], dtype=np.float32)
    )
    detector = TrafficClassifierDetector(
        yolo_session=yolo,
        classifier_session=classifier,
    )
    reading = detector.read_signal(np.zeros((480, 640, 3), dtype=np.uint8))
    assert reading is not None and reading.signal_class == expected


def test_human_bbox_classifier_uses_letterbox_416_input_and_threshold():
    yolo = _FakeOnnxSession(
        _output_with_candidates((320.0, 260.0, 100.0, 30.0, 0.9))
    )
    classifier = _FakeClassifierSession(
        np.array([[0.0, 0.0, 4.0]], dtype=np.float32)
    )
    detector = TrafficClassifierDetector(
        yolo_session=yolo,
        classifier_session=classifier,
        detector_preprocessing=(
            'letterbox_640_center_pad114_bgr_to_rgb_float32_nchw_div255'
        ),
        classifier_input_height=128,
        classifier_input_width=416,
        classifier_classes=('STOP', 'STRAIGHT', 'LEFT'),
        classifier_probability_threshold=0.5,
        classifier_interpolation='pillow_bilinear_antialias',
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    inspection = detector.inspect_signal(frame)

    assert inspection is not None
    assert inspection.reading.signal_class == SignalClass.LEFT
    assert inspection.bbox == DetectionBox(
        x=270,
        y=165,
        width=100,
        height=30,
        confidence=pytest.approx(0.9),
    )
    assert inspection.crop_bounds == ImageBounds(255, 160, 385, 200)
    assert len(inspection.probabilities) == 3
    tensor = classifier.inputs[0]['image']
    assert tensor.shape == (1, 3, 128, 416)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()

    yolo_calls = len(yolo.inputs)
    cached = detector.classify_signal_box(frame, inspection.bbox)
    assert cached.reading == inspection.reading
    assert len(yolo.inputs) == yolo_calls
    assert len(classifier.inputs) == 2

    classifier.output = np.zeros((1, 3), dtype=np.float32)
    unknown = detector.read_signal(frame)
    assert unknown is not None
    assert unknown.signal_class == SignalClass.UNKNOWN
    latch = TrafficSignalLatch(bbox_width_min=40, bbox_width_max=225)
    assert latch.observe(unknown) == LampAction.UNKNOWN


def test_classifier_cadence_searches_every_three_then_classifies_every_frame():
    cadence = TrafficClassifierCadence(
        detector_every_n_frames=3,
        classification_every_n_frames_after_detection=1,
        reuse_detected_bbox_between_yolo_frames=True,
    )
    box = DetectionBox(10.0, 20.0, 100.0, 40.0, 0.9)

    assert not cadence.plan(frame_sequence=1).run_detector
    assert not cadence.plan(frame_sequence=2).run_detector
    search = cadence.plan(frame_sequence=3)
    assert search.run_detector
    assert search.detector_frame_span == 3
    cadence.observe_detection(frame_sequence=3, box=box)

    classify_4 = cadence.plan(frame_sequence=4)
    assert not classify_4.run_detector
    assert classify_4.classification_box == box
    cadence.observe_classification(frame_sequence=4)
    classify_5 = cadence.plan(frame_sequence=5)
    assert classify_5.classification_box == box
    cadence.observe_classification(frame_sequence=5)

    refresh = cadence.plan(frame_sequence=6)
    assert refresh.run_detector
    cadence.observe_detection(frame_sequence=6, box=None)
    assert cadence.plan(frame_sequence=7).classification_box is None
    assert cadence.plan(frame_sequence=8).classification_box is None
    assert cadence.plan(frame_sequence=9).run_detector


def test_action_classifier_stop_maps_to_latched_red():
    latch = TrafficSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        consecutive_reads=2,
    )
    stop = _signal(SignalClass.STOP, width=100)
    straight = _signal(SignalClass.STRAIGHT, width=100)

    assert latch.observe(stop) == LampAction.UNKNOWN
    assert latch.observe(stop) == LampAction.RED
    assert latch.observe(_signal(SignalClass.UNKNOWN, width=100)) == LampAction.RED
    assert latch.observe(straight) == LampAction.RED
    assert latch.observe(straight) == LampAction.STRAIGHT


def test_action_classifier_stop3_go15_applies_before_and_after_stop():
    latch = TrafficSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        red_consecutive_reads=3,
        left_consecutive_reads=15,
        straight_consecutive_reads=15,
    )
    stop = _signal(SignalClass.STOP)
    left = _signal(SignalClass.LEFT)
    straight = _signal(SignalClass.STRAIGHT)
    fsm = TrafficShortcutFsm()
    fsm.enable()

    for step in range(2):
        assert latch.observe(stop) == LampAction.UNKNOWN
        assert latch.snapshot.required_reads == 3
        assert latch.snapshot.candidate_reads == step + 1
        assert fsm.on_frame(
            LampAction.UNKNOWN,
            now_monotonic=float(step),
        ).policy == PolicyChoice.BASE
    assert latch.observe(stop) == LampAction.RED
    assert fsm.on_frame(
        LampAction.RED,
        now_monotonic=2.0,
    ).state == MissionState.RED_STOP

    for step in range(14):
        assert latch.observe(left) == LampAction.RED
        assert latch.snapshot.required_reads == 15
        assert fsm.on_frame(
            LampAction.RED,
            now_monotonic=3.0 + step,
        ).publish_stop
    confirmed_left = latch.observe(left)
    assert confirmed_left == LampAction.LEFT
    left_plan = fsm.on_frame(confirmed_left, now_monotonic=17.0)
    assert left_plan.state == MissionState.SWITCH_TO_SHORTCUT
    assert left_plan.publish_stop

    latch.reset()
    fsm.enable()
    for _ in range(14):
        assert latch.observe(left) == LampAction.UNKNOWN
    assert latch.observe(left) == LampAction.LEFT
    assert fsm.on_frame(
        LampAction.LEFT,
        now_monotonic=20.0,
    ).state == MissionState.SWITCH_TO_SHORTCUT

    latch.reset()
    fsm.enable()
    for _ in range(3):
        stopped = latch.observe(stop)
    assert stopped == LampAction.RED
    assert fsm.on_frame(stopped, now_monotonic=30.0).state == MissionState.RED_STOP
    for _ in range(14):
        assert latch.observe(straight) == LampAction.RED
    assert latch.observe(straight) == LampAction.STRAIGHT
    resumed = fsm.on_frame(LampAction.STRAIGHT, now_monotonic=31.0)
    assert resumed.state == MissionState.BASE
    assert resumed.policy == PolicyChoice.BASE


def test_action_specific_signal_vote_resets_on_unknown_and_class_change():
    latch = TrafficSignalLatch(
        red_consecutive_reads=3,
        left_consecutive_reads=15,
        straight_consecutive_reads=15,
    )
    left = _signal(SignalClass.LEFT)
    straight = _signal(SignalClass.STRAIGHT)

    for _ in range(14):
        assert latch.observe(left) == LampAction.UNKNOWN
    assert latch.snapshot.candidate_reads == 14
    assert latch.snapshot.required_reads == 15
    assert latch.observe(straight) == LampAction.UNKNOWN
    assert latch.snapshot.candidate == SignalClass.STRAIGHT
    assert latch.snapshot.candidate_reads == 1
    assert latch.snapshot.required_reads == 15
    assert latch.observe(None) == LampAction.UNKNOWN
    assert latch.snapshot.candidate == SignalClass.UNKNOWN
    assert latch.snapshot.candidate_reads == 0
    assert latch.snapshot.required_reads == 0


def test_classifier_detector_rejects_bad_logits_and_empty_crop():
    yolo = _FakeOnnxSession(
        _output_with_candidates((320.0, 240.0, 100.0, 40.0, 0.9))
    )
    bad = _FakeClassifierSession(np.zeros((1, 5), dtype=np.float32))
    detector = TrafficClassifierDetector(
        yolo_session=yolo,
        classifier_session=bad,
    )
    with pytest.raises(TrafficLightError, match=r'\[1,4\]'):
        detector.read_signal(np.zeros((480, 640, 3), dtype=np.uint8))
    empty_yolo = _FakeOnnxSession(
        _output_with_candidates((2000.0, 2000.0, 20.0, 20.0, 0.9))
    )
    detector = TrafficClassifierDetector(
        yolo_session=empty_yolo,
        classifier_session=_FakeClassifierSession(
            np.zeros((1, 4), dtype=np.float32)
        ),
    )
    with pytest.raises(TrafficLightError, match='crop is empty'):
        detector.read_signal(np.zeros((100, 100, 3), dtype=np.uint8))
    output = np.zeros((1, 5, 8400), dtype=np.float32)
    output[0, 0, 0] = np.nan
    with pytest.raises(TrafficLightError, match='NaN or Inf'):
        decode_detection_box(
            output,
            frame_height=480,
            frame_width=640,
        )


def test_viewer_overlay_accepts_fractional_letterbox_coordinates():
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    reading = SignalReading(
        signal_class=SignalClass.LEFT_GREEN,
        probability=0.8,
        bbox_width=50,
        confidence=0.9,
    )
    inspection = SignalInspection(
        reading=reading,
        bbox=DetectionBox(10.25, 10.5, 50.5, 20.25, 0.9),
        crop_bounds=ImageBounds(5, 5, 65, 35),
        probabilities=(0.05, 0.05, 0.8, 0.1),
    )
    result = TrafficLightViewerResult(
        frame=frame,
        frame_sequence=3,
        source_monotonic=1.0,
        completed_monotonic=1.01,
        inference_ms=10.0,
        inspection=inspection,
        width_gate_accepted=True,
        final_action=LampAction.UNKNOWN,
        latch_snapshot=TrafficSignalLatchSnapshot(
            candidate=SignalClass.LEFT_GREEN,
            candidate_reads=1,
            required_reads=2,
            stop_latched=False,
        ),
    )

    overlay = draw_signal_overlay(result)

    assert overlay.shape == frame.shape
    assert not np.shares_memory(overlay, frame)
    assert np.count_nonzero(overlay) > 0
    assert np.count_nonzero(frame) == 0


def test_lamp_width_gate_five_votes_priority_and_latch_clearing():
    latch = TrafficLampLatch()
    red = _reading((255, 10, 220, 230), width=45)
    left = _reading((10, 10, 220, 230))
    green = _reading((10, 10, 10, 230))
    unknown = _reading((10, 10, 10, 10))

    assert latch.observe(_reading((255, 10, 10, 10), width=44)) == LampAction.UNKNOWN
    for _ in range(4):
        assert latch.observe(red) == LampAction.UNKNOWN
    assert latch.observe(red) == LampAction.RED
    assert latch.red_latched
    assert latch.observe(None) == LampAction.RED
    assert latch.observe(unknown) == LampAction.RED
    for _ in range(4):
        assert latch.observe(left) == LampAction.RED
    assert latch.observe(left) == LampAction.LEFT
    assert not latch.red_latched

    latch.reset()
    for _ in range(2):
        assert latch.observe(left) == LampAction.UNKNOWN
    assert latch.observe(unknown) == LampAction.UNKNOWN
    for _ in range(4):
        assert latch.observe(left) == LampAction.UNKNOWN
    assert latch.observe(left) == LampAction.LEFT

    latch.reset()
    for _ in range(2):
        assert latch.observe(green) == LampAction.UNKNOWN
    assert latch.observe(left) == LampAction.UNKNOWN
    for _ in range(4):
        assert latch.observe(green) == LampAction.UNKNOWN
    assert latch.observe(green) == LampAction.STRAIGHT


def test_legacy_lamp_votes_keep_red_three_and_other_actions_immediate():
    latch = TrafficLampLatch(
        red_consecutive_reads=3,
        left_consecutive_reads=1,
        straight_consecutive_reads=1,
    )
    red = _reading((255, 10, 220, 230), width=45)
    left = _reading((10, 10, 220, 230))

    assert latch.observe(red) == LampAction.UNKNOWN
    assert latch.observe(red) == LampAction.UNKNOWN
    assert latch.observe(red) == LampAction.RED
    assert latch.observe(left) == LampAction.LEFT
    assert not latch.red_latched


def test_classifier_latch_requires_same_raw_class_and_retains_stop():
    latch = TrafficSignalLatch()
    assert (
        latch.observe(_signal(SignalClass.RED, width=44))
        == LampAction.UNKNOWN
    )
    assert latch.snapshot.candidate == SignalClass.UNKNOWN
    assert latch.snapshot.candidate_reads == 0
    assert latch.observe(_signal(SignalClass.RED)) == LampAction.UNKNOWN
    assert latch.snapshot.candidate == SignalClass.RED
    assert latch.snapshot.candidate_reads == 1
    assert latch.snapshot.required_reads == 2
    assert latch.observe(_signal(SignalClass.YELLOW)) == LampAction.UNKNOWN
    assert latch.observe(_signal(SignalClass.YELLOW)) == LampAction.RED
    assert latch.stop_latched
    assert latch.snapshot.stop_latched
    assert latch.observe(None) == LampAction.RED
    latch.release_stop_latch()
    assert not latch.stop_latched
    assert latch.observe(None) == LampAction.UNKNOWN
    assert latch.observe(_signal(SignalClass.YELLOW)) == LampAction.UNKNOWN
    assert latch.observe(_signal(SignalClass.YELLOW)) == LampAction.RED
    assert latch.observe(_signal(SignalClass.LEFT_GREEN)) == LampAction.RED
    assert latch.observe(_signal(SignalClass.LEFT_GREEN)) == LampAction.LEFT
    assert not latch.stop_latched
    assert (
        latch.observe(_signal(SignalClass.STRAIGHT_GREEN))
        == LampAction.UNKNOWN
    )
    assert (
        latch.observe(_signal(SignalClass.STRAIGHT_GREEN))
        == LampAction.STRAIGHT
    )
    latch.reset()
    assert latch.snapshot.candidate == SignalClass.UNKNOWN
    assert latch.snapshot.candidate_reads == 0
    assert not latch.snapshot.stop_latched


def test_bundle_signal_vote_contract_preserves_legacy_and_requires_five():
    legacy = {
        'red_latch': {
            'consecutive_red_reads': 3,
            'unknown_behavior': 'retain_latch',
            'clear_actions': ['LEFT', 'STRAIGHT'],
        }
    }
    signal_vote = {
        'actions': ['RED', 'LEFT', 'STRAIGHT'],
        'consecutive_reads': 5,
        'unknown_behavior': 'reset_candidate',
        'different_action_behavior': 'restart_candidate_at_one',
        'red_latch_behavior': 'retain_until_confirmed_clear_action',
        'red_clear_actions': ['LEFT', 'STRAIGHT'],
    }
    classifier_vote = {
        'raw_classes': ['red', 'yellow', 'left_green', 'straight_green'],
        'consecutive_reads': 2,
        'unknown_behavior': 'reset_candidate',
        'different_raw_class_behavior': 'restart_candidate_at_one',
        'stop_classes': ['red', 'yellow'],
        'stop_latch_behavior': 'retain_until_confirmed_green_class',
        'stop_clear_classes': ['left_green', 'straight_green'],
    }
    human_bbox_classifier_vote = {
        'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
        'consecutive_reads': 2,
        'unknown_behavior': 'reset_candidate',
        'different_raw_class_behavior': 'restart_candidate_at_one',
        'stop_classes': ['STOP'],
        'stop_latch_behavior': 'retain_until_confirmed_go_action',
        'stop_clear_classes': ['LEFT', 'STRAIGHT'],
    }
    stabilized_human_bbox_classifier_vote = {
        'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
        'consecutive_reads_by_raw_class': {
            'STOP': 3,
            'STRAIGHT': 15,
            'LEFT': 15,
        },
        'unknown_behavior': 'reset_candidate',
        'different_raw_class_behavior': 'restart_candidate_at_one',
        'stop_classes': ['STOP'],
        'stop_latch_behavior': 'retain_until_confirmed_go_action',
        'stop_clear_classes': ['LEFT', 'STRAIGHT'],
    }

    assert _load_signal_vote_contract(
        legacy,
        schema_version=2,
        artifact_id='legacy',
    ) == (3, 1, 1)
    assert _load_signal_vote_contract(
        {'signal_vote': signal_vote},
        schema_version=3,
        artifact_id=EXPECTED_SIGNAL_VOTE_BUNDLE_ID,
    ) == (5, 5, 5)
    assert _load_signal_vote_contract(
        {'signal_vote': signal_vote},
        schema_version=3,
        artifact_id=EXPECTED_EXPANDED_SIGNAL_VOTE_BUNDLE_ID,
    ) == (5, 5, 5)
    assert _expected_shortcut_artifact_id(
        schema_version=2,
        artifact_id='legacy',
    ) == EXPECTED_SHORTCUT_ARTIFACT_ID
    assert _expected_shortcut_artifact_id(
        schema_version=3,
        artifact_id=EXPECTED_EXPANDED_SIGNAL_VOTE_BUNDLE_ID,
    ) == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    assert _load_signal_vote_contract(
        {'signal_vote': classifier_vote},
        schema_version=4,
        artifact_id=EXPECTED_CLASSIFIER_BUNDLE_ID,
    ) == (2, 2, 2)
    assert _load_signal_vote_contract(
        {'signal_vote': classifier_vote},
        schema_version=5,
        artifact_id=EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID,
    ) == (2, 2, 2)
    assert _expected_shortcut_artifact_id(
        schema_version=5,
        artifact_id=EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID,
    ) == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    assert _load_signal_vote_contract(
        {'signal_vote': human_bbox_classifier_vote},
        schema_version=6,
        artifact_id=EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == (2, 2, 2)
    assert _expected_shortcut_artifact_id(
        schema_version=6,
        artifact_id=EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    assert _load_signal_vote_contract(
        {'signal_vote': stabilized_human_bbox_classifier_vote},
        schema_version=7,
        artifact_id=EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == (3, 15, 15)
    assert _expected_shortcut_artifact_id(
        schema_version=7,
        artifact_id=EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    assert _load_signal_vote_contract(
        {'signal_vote': stabilized_human_bbox_classifier_vote},
        schema_version=8,
        artifact_id=EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == (3, 15, 15)
    assert _expected_shortcut_artifact_id(
        schema_version=8,
        artifact_id=EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    with pytest.raises(ArtifactContractError, match='signal vote'):
        _load_signal_vote_contract(
            {
                'signal_vote': {
                    **signal_vote,
                    'consecutive_reads': 4,
                }
            },
            schema_version=3,
            artifact_id=EXPECTED_SIGNAL_VOTE_BUNDLE_ID,
        )
    with pytest.raises(ArtifactContractError, match='not approved'):
        _load_signal_vote_contract(
            {'signal_vote': signal_vote},
            schema_version=3,
            artifact_id='unapproved-schema3-bundle',
        )


def test_yolo_missing_release_requires_ten_scheduled_misses():
    counter = YoloMissingReleaseCounter(
        release_frames=30,
        inference_every_n_frames=3,
    )
    fsm = TrafficShortcutFsm()
    fsm.enable()
    assert fsm.on_frame(
        LampAction.RED,
        now_monotonic=0.0,
    ).state == MissionState.RED_STOP

    for step in range(9):
        assert not counter.observe(
            red_stop_active=True,
            detector_observed=True,
            yolo_box_found=False,
        )
        assert counter.missing_frames == (step + 1) * 3
        assert fsm.on_frame(
            LampAction.UNKNOWN,
            now_monotonic=0.1 + step,
        ).publish_stop

    assert counter.observe(
        red_stop_active=True,
        detector_observed=True,
        yolo_box_found=True,
    ) is False
    assert counter.missing_frames == 0

    for frame_span in (4, 4, 4, 4, 4, 4, 4):
        assert not counter.observe(
            red_stop_active=True,
            detector_observed=True,
            yolo_box_found=False,
            detector_frame_span=frame_span,
        )
    assert counter.missing_frames == 28
    assert counter.observe(
        red_stop_active=True,
        detector_observed=True,
        yolo_box_found=False,
        detector_frame_span=3,
    )
    assert counter.missing_frames == 0

    for _ in range(9):
        assert not counter.observe(
            red_stop_active=True,
            detector_observed=True,
            yolo_box_found=False,
        )
    assert counter.observe(
        red_stop_active=True,
        detector_observed=True,
        yolo_box_found=False,
    )
    assert counter.missing_frames == 0
    resumed = fsm.on_frame(LampAction.STRAIGHT, now_monotonic=20.0)
    assert resumed.state == MissionState.BASE
    assert resumed.policy == PolicyChoice.BASE

    assert not counter.observe(
        red_stop_active=False,
        detector_observed=True,
        yolo_box_found=False,
    )
    assert counter.missing_frames == 0


def test_fsm_base_transition_exact_eight_seconds_and_one_shot():
    fsm = TrafficShortcutFsm(shortcut_duration_sec=8.0)
    assert fsm.state == MissionState.OFF
    fsm.enable()
    base = fsm.on_frame(LampAction.UNKNOWN, now_monotonic=1.0)
    assert base.policy == PolicyChoice.BASE

    switch = fsm.on_frame(LampAction.LEFT, now_monotonic=2.0)
    assert switch.publish_stop
    assert switch.state == MissionState.SWITCH_TO_SHORTCUT
    shortcut = fsm.on_frame(LampAction.LEFT, now_monotonic=2.1)
    assert shortcut.policy == PolicyChoice.SHORTCUT
    fsm.on_shortcut_command_published(now_monotonic=3.0)
    assert fsm.on_control_tick(now_monotonic=10.999999) is None

    completed = fsm.on_control_tick(now_monotonic=11.0)
    assert completed is not None and completed.publish_stop
    assert completed.state == MissionState.SWITCH_TO_BASE
    assert fsm.shortcut_completed
    back_to_base = fsm.on_frame(LampAction.LEFT, now_monotonic=11.1)
    assert back_to_base.policy == PolicyChoice.BASE
    ignored_left = fsm.on_frame(LampAction.LEFT, now_monotonic=11.2)
    assert ignored_left.policy == PolicyChoice.BASE


def test_fsm_shadow_handoff_has_no_exit_stop_and_completes_on_promotion():
    fsm = TrafficShortcutFsm(
        shortcut_duration_sec=8.0,
        seamless_base_handoff=True,
    )
    fsm.enable()
    assert fsm.on_frame(
        LampAction.LEFT,
        now_monotonic=1.0,
    ).publish_stop
    assert fsm.on_frame(
        LampAction.LEFT,
        now_monotonic=1.1,
    ).policy == PolicyChoice.SHORTCUT
    fsm.on_shortcut_command_published(now_monotonic=2.0)

    handoff = fsm.on_control_tick(now_monotonic=10.0)

    assert handoff is not None
    assert not handoff.publish_stop
    assert handoff.promote_base_shadow
    assert handoff.state == MissionState.SWITCH_TO_BASE
    assert not fsm.shortcut_completed
    fsm.on_base_shadow_promoted()
    assert fsm.state == MissionState.BASE
    assert fsm.shortcut_completed


def test_fsm_red_cancels_without_consuming_and_allows_retry():
    fsm = TrafficShortcutFsm()
    fsm.enable()
    fsm.on_frame(LampAction.LEFT, now_monotonic=0.0)
    fsm.on_frame(LampAction.LEFT, now_monotonic=0.1)
    fsm.on_shortcut_command_published(now_monotonic=0.2)

    red = fsm.on_frame(LampAction.RED, now_monotonic=1.0)
    assert red.publish_stop
    assert red.state == MissionState.RED_STOP
    assert not fsm.shortcut_completed
    assert fsm.on_frame(
        LampAction.UNKNOWN,
        now_monotonic=1.1,
    ).publish_stop

    retry = fsm.on_frame(LampAction.LEFT, now_monotonic=1.2)
    assert retry.state == MissionState.SWITCH_TO_SHORTCUT
    assert retry.publish_stop
    assert fsm.on_frame(
        LampAction.LEFT,
        now_monotonic=1.3,
    ).policy == PolicyChoice.SHORTCUT


def test_fsm_off_fault_and_red_priority_paths_stop():
    fsm = TrafficShortcutFsm()
    assert fsm.on_frame(LampAction.LEFT, now_monotonic=0.0).publish_stop
    fsm.enable()
    assert fsm.on_frame(LampAction.RED, now_monotonic=0.1).state == MissionState.RED_STOP
    assert fsm.on_frame(LampAction.STRAIGHT, now_monotonic=0.2).policy == PolicyChoice.BASE
    fsm.fault()
    assert fsm.on_frame(LampAction.STRAIGHT, now_monotonic=0.3).publish_stop
    fsm.disable()
    assert fsm.state == MissionState.OFF


def _safe_node_state():
    return SimpleNamespace(
        _competitors=(),
        _has_motor_subscriber=True,
        require_gamepad_hold=True,
        allow_motion=True,
        _joy_valid=True,
        _last_joy_monotonic=0.95,
        joy_timeout_sec=0.25,
        _last_camera_monotonic=0.95,
        camera_timeout_sec=0.25,
        _awaiting_post_reset_decision=False,
        _history_reset_monotonic=None,
        inference_timeout_sec=0.25,
        _drive_gate=SimpleNamespace(enabled=True),
        _transition_stop_pending=False,
        _decision=MissionDecision(
            command=DriveCommand(1.0, 20.0),
            policy=PolicyChoice.BASE,
            state=MissionState.BASE,
            source_monotonic=0.95,
            completed_monotonic=0.96,
            inference_ms=2.0,
            frame_sequence=1,
        ),
    )


@pytest.mark.parametrize(
    ('mutation', 'expected'),
    [
        ({'_competitors': ('/other',)}, 'competing motor publisher'),
        ({'_has_motor_subscriber': False}, 'no motor subscriber'),
        ({'_joy_valid': False}, 'Joy input'),
        ({'_last_joy_monotonic': 0.0}, 'Joy input'),
        ({'_last_camera_monotonic': 0.0}, 'camera input'),
        ({'_decision': None}, 'selected policy inference'),
    ],
)
def test_integrated_node_all_stale_and_graph_paths_fail_closed(
    mutation,
    expected,
):
    node = _safe_node_state()
    for name, value in mutation.items():
        setattr(node, name, value)
    reason = TrafficShortcutPolicyNode._unsafe_reason_locked(node, 1.0)
    assert expected in reason


def test_integrated_node_post_reset_timeout_and_actual_history_only():
    node = _safe_node_state()
    node._awaiting_post_reset_decision = True
    node._history_reset_monotonic = 0.0
    assert 'post-reset' in TrafficShortcutPolicyNode._unsafe_reason_locked(
        node,
        1.0,
    )

    published = []
    history_node = SimpleNamespace(
        _history=deque([(50, 75)] * 4, maxlen=4),
        _last_executed_decision_sequence=0,
        _publish=lambda command: published.append(command),
    )
    method = TrafficShortcutPolicyNode._publish_and_record_locked
    method(
        history_node,
        DriveCommand(20.0, 25.0),
        decision_sequence=7,
    )
    method(
        history_node,
        DriveCommand(20.0, 25.0),
        decision_sequence=7,
    )
    method(history_node, DriveCommand(), decision_sequence=None)
    assert len(published) == 3
    assert tuple(history_node._history)[-2:] == ((60, 75), (50, 50))


def test_integrated_node_without_gamepad_does_not_require_joy():
    node = _safe_node_state()
    node.require_gamepad_hold = False
    node._joy_valid = False
    node._last_joy_monotonic = None

    assert TrafficShortcutPolicyNode._unsafe_reason_locked(node, 1.0) is None
    assert TrafficShortcutPolicyNode._can_enable_locked(node, 1.0)


def test_traffic_shortcut_launch_ties_gamepad_to_hold_gate():
    launch_path = (
        Path(__file__).parents[1]
        / 'launch'
        / 'traffic_shortcut_policy.launch.py'
    )
    launch_text = launch_path.read_text(encoding='utf-8')

    assert "'require_gamepad_hold': ParameterValue(" in launch_text
    assert 'use_gamepad,' in launch_text


def test_traffic_light_viewer_launch_has_no_motion_endpoints():
    package_root = Path(__file__).parents[1]
    launch_text = (
        package_root / 'launch' / 'traffic_light_viewer.launch.py'
    ).read_text(encoding='utf-8')
    viewer_text = (
        package_root
        / 'xycar_ai_drive'
        / 'traffic_light_viewer.py'
    ).read_text(encoding='utf-8')

    assert "executable='traffic_light_viewer'" in launch_text
    assert "'use_camera'" in launch_text
    assert 'xycar_cam.launch.py' in launch_text
    assert 'allow_motion' not in launch_text
    assert 'game_controller_node' not in launch_text
    assert 'traffic_shortcut_policy' not in launch_text
    assert 'ExecuteProcess' not in launch_text
    assert '.create_publisher(' not in viewer_text
    assert 'UnixSocketPolicyClient' not in viewer_text


def test_shadow_bundle_requires_verified_paired_server_identity():
    base_artifact = object()
    shortcut_artifact = object()
    node = SimpleNamespace(
        bundle=SimpleNamespace(
            base=base_artifact,
            shortcut=shortcut_artifact,
            base_shadow_enabled=True,
        ),
        _base_policy=SimpleNamespace(
            _artifact=base_artifact,
            supports_pair_inference=True,
            paired_artifact_id='shortcut',
            paired_artifact_digest='digest',
        ),
        _shortcut_policy=SimpleNamespace(
            _artifact=shortcut_artifact,
            artifact_id='shortcut',
            artifact_digest='digest',
        ),
    )

    TrafficShortcutPolicyNode._validate_client_artifacts(node)
    node._base_policy.paired_artifact_digest = 'wrong'

    with pytest.raises(ValueError, match='paired shortcut identity'):
        TrafficShortcutPolicyNode._validate_client_artifacts(node)


def _bind_shadow_methods(node):
    for name in (
        '_decision_from_result',
        '_start_base_shadow_locked',
        '_base_shadow_snapshot_locked',
        '_infer_and_store_base_shadow',
        '_store_base_shadow_decision_locked',
        '_discard_base_shadow_locked',
        '_promote_base_shadow_locked',
    ):
        setattr(
            node,
            name,
            MethodType(getattr(TrafficShortcutPolicyNode, name), node),
        )
    return node


def test_base_shadow_recursively_uses_capped_predictions_without_publish():
    histories = []
    outputs = iter(
        (
            DriveCommand(angle=20.0, speed=30.0),
            DriveCommand(angle=30.0, speed=24.0),
        )
    )

    class FakeBasePolicy:
        def infer(self, _image, history):
            histories.append(tuple(history))
            return SimpleNamespace(
                command=next(outputs),
                inference_ms=2.0,
            )

    node = _bind_shadow_methods(
        SimpleNamespace(
            bundle=SimpleNamespace(
                base_shadow_enabled=True,
                base_speed_cap=25.0,
                shortcut_speed=23.0,
            ),
            _history=deque([(50, 75)] * 4, maxlen=4),
            _base_shadow_history=None,
            _base_shadow_decision=None,
            _base_shadow_epoch=0,
            _base_policy=FakeBasePolicy(),
            _fsm=SimpleNamespace(state=MissionState.SHORTCUT),
            _lock=threading.RLock(),
        )
    )
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    first = node._start_base_shadow_locked()
    node._infer_and_store_base_shadow(
        image=image,
        source_monotonic=1.0,
        sequence=1,
        shadow_work=first,
    )
    second = node._base_shadow_snapshot_locked()
    node._infer_and_store_base_shadow(
        image=image,
        source_monotonic=1.1,
        sequence=2,
        shadow_work=second,
    )

    assert histories[0] == ((50, 75),) * 4
    assert histories[1][-1] == (60, 75)
    assert tuple(node._history) == ((50, 75),) * 4
    assert tuple(node._base_shadow_history)[-2:] == ((60, 75), (65, 74))
    assert node._base_shadow_decision.command == DriveCommand(30.0, 24.0)


def _shadow_handoff_node(*, source_monotonic=9.9):
    fsm = TrafficShortcutFsm(seamless_base_handoff=True)
    fsm.enable()
    fsm.state = MissionState.SWITCH_TO_BASE
    published = []
    predictions = []
    node = _bind_shadow_methods(
        SimpleNamespace(
            bundle=SimpleNamespace(
                base_shadow_enabled=True,
                base_shadow_max_age_sec=0.25,
            ),
            _base_shadow_decision=MissionDecision(
                command=DriveCommand(12.0, 22.0),
                policy=PolicyChoice.BASE,
                state=MissionState.SHORTCUT,
                source_monotonic=source_monotonic,
                completed_monotonic=source_monotonic + 0.01,
                inference_ms=2.0,
                frame_sequence=42,
            ),
            _base_shadow_history=deque(
                [(55, 70), (56, 71), (57, 72), (62, 72)],
                maxlen=4,
            ),
            _base_shadow_epoch=1,
            _fsm=fsm,
            _history=deque([(50, 75)] * 4, maxlen=4),
            _decision=None,
            _last_executed_decision_sequence=7,
            _mission_generation=3,
            _minimum_next_frame_sequence=0,
            _awaiting_post_reset_decision=True,
            _history_reset_monotonic=8.0,
            _transition_stop_pending=False,
            _stop_reason='old',
            _publish=lambda command: published.append(command),
            _publish_prediction=lambda decision: predictions.append(decision),
        )
    )
    return node, published, predictions


def test_fresh_shadow_handoff_promotes_history_without_stop():
    node, published, predictions = _shadow_handoff_node()

    assert node._promote_base_shadow_locked(10.0) is None

    assert published == [DriveCommand(12.0, 22.0)]
    assert published[0] != DriveCommand()
    assert len(predictions) == 1
    assert node._fsm.state == MissionState.BASE
    assert node._fsm.shortcut_completed
    assert tuple(node._history) == (
        (55, 70),
        (56, 71),
        (57, 72),
        (62, 72),
    )
    assert node._decision.policy == PolicyChoice.BASE
    assert node._base_shadow_history is None


def test_stale_shadow_handoff_fails_without_publish_and_red_discards():
    node, published, _predictions = _shadow_handoff_node(
        source_monotonic=9.0
    )

    reason = node._promote_base_shadow_locked(10.0)

    assert 'stale' in reason
    assert not published
    assert node._fsm.state == MissionState.SWITCH_TO_BASE
    node._discard_base_shadow_locked()
    assert node._base_shadow_history is None
    assert node._base_shadow_decision is None
