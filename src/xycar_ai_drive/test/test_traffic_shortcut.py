# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from collections import OrderedDict, deque
from dataclasses import replace
import math
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from xycar_ai_drive.artifact import ArtifactContractError
from xycar_ai_drive.traffic_light_detector import (
    DetectionBox,
    ImageBounds,
    InitialStopPhase,
    InitialStopSignalLatch,
    InitialStopSignalLatchSnapshot,
    InitialWaitSignalLatch,
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
    should_update_signal_vote,
)
from xycar_ai_drive.traffic_light_viewer import (
    TrafficLightViewerNode,
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
    EXPECTED_STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_BASE_ARTIFACT_ID,
    EXPECTED_SPEED35_FIX_BASE_ARTIFACT_ID,
    EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID,
    EXPECTED_SHORTCUT_ARTIFACT_ID,
    EXPECTED_SIGNAL_VOTE_BUNDLE_ID,
    _expected_shortcut_artifact_id,
    _expected_base_artifact_id,
    _load_signal_vote_contract,
)
from xycar_ai_drive.traffic_shortcut_policy_node import (
    MissionDecision,
    SignalStatusLogGate,
    TrafficShortcutPolicyNode,
    YoloMissingReleaseCounter,
    format_signal_status,
)
from xycar_ai_drive.traffic_shortcut_diagnostics import (
    SIGNAL_DEBUG_SCHEMA_VERSION,
    SignalDebugBbox,
    SignalDebugContractError,
    SignalDebugCrop,
    SignalDebugSnapshot,
    decode_signal_debug,
    encode_signal_debug,
)
from xycar_ai_drive.traffic_shortcut_monitor import (
    MonitorFrame,
    SignalPanelMode,
    TrafficShortcutMonitorNode,
    TrafficShortcutMonitorSnapshot,
    drive_vector_endpoint,
    frame_stamp_key,
    monitor_crop_panel,
    monitor_frame_panel,
    monitor_live_panel,
    select_signal_panel,
)
from xycar_ai_drive.control import DriveCommand, ToggleAction


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
    assert (
        decode_detection_box(
            np.zeros((1, 5, 8400), dtype=np.float32),
            frame_height=480,
            frame_width=640,
        )
        is None
    )


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
    assert inspection.probabilities[2] == pytest.approx(reading.probability)
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
    classifier = _FakeClassifierSession(np.array([logits], dtype=np.float32))
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


def test_classifier_cadence_runs_fresh_yolo_cnn_only_every_three_frames():
    cadence = TrafficClassifierCadence(
        detector_every_n_frames=3,
        classification_every_n_frames_after_detection=3,
        reuse_detected_bbox_between_yolo_frames=False,
    )
    box = DetectionBox(10.0, 20.0, 100.0, 40.0, 0.9)

    for sequence in (1, 2):
        plan = cadence.plan(frame_sequence=sequence)
        assert not plan.run_detector
        assert plan.classification_box is None
    first = cadence.plan(frame_sequence=3)
    assert first.run_detector
    assert first.detector_frame_span == 3
    cadence.observe_detection(frame_sequence=3, box=box)
    for sequence in (4, 5):
        plan = cadence.plan(frame_sequence=sequence)
        assert not plan.run_detector
        assert plan.classification_box is None
    second = cadence.plan(frame_sequence=6)
    assert second.run_detector
    assert second.detector_frame_span == 3


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
    assert (
        latch.observe(_signal(SignalClass.UNKNOWN, width=100))
        == LampAction.RED
    )
    assert latch.observe(straight) == LampAction.RED
    assert latch.observe(straight) == LampAction.STRAIGHT


def test_action_classifier_stop10_go30_applies_before_and_after_stop():
    latch = TrafficSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        red_consecutive_reads=10,
        left_consecutive_reads=30,
        straight_consecutive_reads=30,
    )
    stop = _signal(SignalClass.STOP)
    left = _signal(SignalClass.LEFT)
    straight = _signal(SignalClass.STRAIGHT)
    fsm = TrafficShortcutFsm()
    fsm.enable()

    for step in range(9):
        assert latch.observe(stop) == LampAction.UNKNOWN
        assert latch.snapshot.required_reads == 10
        assert latch.snapshot.candidate_reads == step + 1
        assert (
            fsm.on_frame(
                LampAction.UNKNOWN,
                now_monotonic=float(step),
            ).policy
            == PolicyChoice.BASE
        )
    assert latch.observe(stop) == LampAction.RED
    assert (
        fsm.on_frame(
            LampAction.RED,
            now_monotonic=9.0,
        ).state
        == MissionState.RED_STOP
    )

    for step in range(29):
        assert latch.observe(left) == LampAction.RED
        assert latch.snapshot.required_reads == 30
        assert fsm.on_frame(
            LampAction.RED,
            now_monotonic=10.0 + step,
        ).publish_stop
    confirmed_left = latch.observe(left)
    assert confirmed_left == LampAction.LEFT
    left_plan = fsm.on_frame(confirmed_left, now_monotonic=39.0)
    assert left_plan.state == MissionState.SWITCH_TO_SHORTCUT
    assert left_plan.publish_stop

    latch.reset()
    fsm.enable()
    for _ in range(29):
        assert latch.observe(left) == LampAction.UNKNOWN
    assert latch.observe(left) == LampAction.LEFT
    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=50.0,
        ).state
        == MissionState.SWITCH_TO_SHORTCUT
    )

    latch.reset()
    fsm.enable()
    for _ in range(10):
        stopped = latch.observe(stop)
    assert stopped == LampAction.RED
    assert (
        fsm.on_frame(stopped, now_monotonic=60.0).state
        == MissionState.RED_STOP
    )
    for _ in range(29):
        assert latch.observe(straight) == LampAction.RED
    assert latch.observe(straight) == LampAction.STRAIGHT
    resumed = fsm.on_frame(LampAction.STRAIGHT, now_monotonic=61.0)
    assert resumed.state == MissionState.BASE
    assert resumed.policy == PolicyChoice.BASE


def test_action_specific_signal_vote_resets_on_unknown_and_class_change():
    latch = TrafficSignalLatch(
        red_consecutive_reads=10,
        left_consecutive_reads=30,
        straight_consecutive_reads=30,
    )
    left = _signal(SignalClass.LEFT)
    straight = _signal(SignalClass.STRAIGHT)

    for _ in range(29):
        assert latch.observe(left) == LampAction.UNKNOWN
    assert latch.snapshot.candidate_reads == 29
    assert latch.snapshot.required_reads == 30
    assert latch.observe(straight) == LampAction.UNKNOWN
    assert latch.snapshot.candidate == SignalClass.STRAIGHT
    assert latch.snapshot.candidate_reads == 1
    assert latch.snapshot.required_reads == 30
    assert latch.observe(None) == LampAction.UNKNOWN
    assert latch.snapshot.candidate == SignalClass.UNKNOWN
    assert latch.snapshot.candidate_reads == 0
    assert latch.snapshot.required_reads == 0


def test_action_classifier_stop15_go15_resumes_without_drive_gate_reset():
    latch = TrafficSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        red_consecutive_reads=15,
        left_consecutive_reads=15,
        straight_consecutive_reads=15,
    )
    stop = _signal(SignalClass.STOP)
    straight = _signal(SignalClass.STRAIGHT)
    fsm = TrafficShortcutFsm()
    fsm.enable()

    for step in range(14):
        assert latch.observe(stop) == LampAction.UNKNOWN
        assert (
            fsm.on_frame(
                LampAction.UNKNOWN,
                now_monotonic=float(step),
            ).policy
            == PolicyChoice.BASE
        )
    assert latch.observe(stop) == LampAction.RED
    assert (
        fsm.on_frame(
            LampAction.RED,
            now_monotonic=14.0,
        ).state
        == MissionState.RED_STOP
    )

    for step in range(14):
        assert latch.observe(straight) == LampAction.RED
        assert fsm.on_frame(
            LampAction.RED,
            now_monotonic=15.0 + step,
        ).publish_stop
    confirmed = latch.observe(straight)
    assert confirmed == LampAction.STRAIGHT
    resumed = fsm.on_frame(confirmed, now_monotonic=29.0)
    assert resumed.state == MissionState.BASE
    assert resumed.policy == PolicyChoice.BASE


def test_initial_stop_once_latch_clears_on_three_mixed_non_stop_reads():
    latch = InitialStopSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        stop_consecutive_reads=15,
        clear_consecutive_reads=3,
        left_consecutive_reads=15,
        straight_consecutive_reads=15,
    )
    stop = _signal(SignalClass.STOP)
    left = _signal(SignalClass.LEFT)
    straight = _signal(SignalClass.STRAIGHT)
    latch.reset(initial_stop_armed=True)

    for step in range(14):
        assert latch.observe(stop) == LampAction.UNKNOWN
        assert latch.snapshot.phase == InitialStopPhase.ARMED
        assert latch.snapshot.candidate_reads == step + 1
    assert latch.observe(stop) == LampAction.RED
    assert latch.snapshot.phase == InitialStopPhase.STOPPED

    assert latch.observe(left) == LampAction.RED
    assert latch.snapshot.candidate_reads == 1
    assert latch.observe(straight) == LampAction.RED
    assert latch.snapshot.candidate_reads == 2
    assert latch.observe(None) == LampAction.RED
    assert latch.snapshot.candidate_reads == 0
    assert latch.observe(straight) == LampAction.RED
    assert latch.observe(left) == LampAction.RED
    assert latch.observe(straight) == LampAction.STRAIGHT
    assert latch.snapshot.phase == InitialStopPhase.NAVIGATION

    assert latch.observe(stop) == LampAction.UNKNOWN
    assert latch.snapshot.stop_ignored
    for _ in range(14):
        assert latch.observe(left) == LampAction.UNKNOWN
    assert latch.observe(left) == LampAction.LEFT

    latch.reset(initial_stop_armed=True, wait_for_signal=True)
    assert latch.snapshot.phase == InitialStopPhase.WAIT_FOR_SIGNAL
    assert latch.observe(stop) == LampAction.RED
    assert latch.observe(straight) == LampAction.RED
    assert latch.observe(left) == LampAction.RED
    assert latch.observe(straight) == LampAction.STRAIGHT
    assert latch.snapshot.phase == InitialStopPhase.NAVIGATION


def test_initial_stop_once_fsm_arms_waits_and_ignores_later_red():
    fsm = TrafficShortcutFsm(one_shot_initial_stop=True)
    fsm.enable(initial_stop_armed=True)
    assert (
        fsm.on_frame(
            LampAction.UNKNOWN,
            now_monotonic=0.0,
        ).policy
        == PolicyChoice.BASE
    )
    stopped = fsm.on_frame(LampAction.RED, now_monotonic=0.1)
    assert stopped.state == MissionState.INITIAL_STOP
    assert stopped.publish_stop
    resumed = fsm.on_frame(LampAction.STRAIGHT, now_monotonic=0.2)
    assert resumed.state == MissionState.BASE
    assert resumed.policy == PolicyChoice.BASE
    assert fsm.initial_stop_consumed
    assert (
        fsm.on_frame(
            LampAction.RED,
            now_monotonic=0.3,
        ).policy
        == PolicyChoice.BASE
    )

    fsm.disable()
    fsm.enable(initial_stop_armed=True, wait_for_signal=True)
    assert fsm.state == MissionState.WAIT_FOR_SIGNAL
    assert fsm.on_frame(
        LampAction.RED,
        now_monotonic=0.4,
    ).publish_stop
    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=0.5,
        ).policy
        == PolicyChoice.BASE
    )

    fsm.disable()
    fsm.enable()
    assert (
        fsm.on_frame(
            LampAction.RED,
            now_monotonic=0.6,
        ).policy
        == PolicyChoice.BASE
    )
    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=0.7,
        ).state
        == MissionState.SWITCH_TO_SHORTCUT
    )


def test_initial_wait_latch_uses_fresh_stop5_and_go3_votes():
    latch = InitialWaitSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        stop_consecutive_reads=5,
        left_consecutive_reads=3,
        straight_consecutive_reads=3,
    )
    latch.reset(initial_stop_armed=True, wait_for_signal=True)
    left = _signal(SignalClass.LEFT)
    straight = _signal(SignalClass.STRAIGHT)
    stop = _signal(SignalClass.STOP)

    for _ in range(30):
        assert not should_update_signal_vote(
            reading_observed=True,
            detector_ran=False,
            fresh_yolo_only=True,
        )
    assert latch.snapshot.candidate_reads == 0
    assert latch.snapshot.phase == InitialStopPhase.WAIT_FOR_SIGNAL

    for step in range(4):
        assert should_update_signal_vote(
            reading_observed=True,
            detector_ran=True,
            fresh_yolo_only=True,
        )
        assert latch.observe(stop) == LampAction.RED
        assert latch.snapshot.candidate_reads == step + 1
    assert latch.observe(straight) == LampAction.RED
    assert latch.snapshot.candidate == SignalClass.STRAIGHT
    assert latch.snapshot.candidate_reads == 1
    assert latch.observe(None) == LampAction.RED
    assert latch.snapshot.candidate_reads == 0

    for _ in range(4):
        assert latch.observe(stop) == LampAction.RED
    assert latch.snapshot.phase == InitialStopPhase.WAIT_FOR_SIGNAL
    assert latch.observe(stop) == LampAction.RED
    assert latch.snapshot.phase == InitialStopPhase.STOPPED
    for _ in range(2):
        assert latch.observe(left) == LampAction.RED
    assert latch.observe(left) == LampAction.LEFT
    assert latch.snapshot.phase == InitialStopPhase.NAVIGATION

    assert latch.observe(stop) == LampAction.UNKNOWN
    assert latch.snapshot.stop_ignored
    for _ in range(2):
        assert latch.observe(straight) == LampAction.UNKNOWN
    assert latch.observe(straight) == LampAction.STRAIGHT

    latch.reset(initial_stop_armed=True, wait_for_signal=True)
    for _ in range(2):
        assert latch.observe(straight) == LampAction.RED
    assert latch.observe(straight) == LampAction.STRAIGHT
    latch.reset(initial_stop_armed=True, wait_for_signal=True)
    for _ in range(2):
        assert latch.observe(left) == LampAction.RED
    assert latch.observe(left) == LampAction.LEFT


def test_initial_wait_latch_uses_fresh_stop5_and_go1_votes():
    latch = InitialWaitSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        stop_consecutive_reads=5,
        left_consecutive_reads=1,
        straight_consecutive_reads=1,
    )
    latch.reset(initial_stop_armed=True, wait_for_signal=True)
    stop = _signal(SignalClass.STOP)
    straight = _signal(SignalClass.STRAIGHT)
    left = _signal(SignalClass.LEFT)

    for step in range(4):
        assert latch.observe(stop) == LampAction.RED
        assert latch.snapshot.candidate_reads == step + 1
        assert latch.snapshot.required_reads == 5
    assert latch.observe(stop) == LampAction.RED
    assert latch.snapshot.phase == InitialStopPhase.STOPPED
    assert latch.observe(straight) == LampAction.STRAIGHT
    assert latch.snapshot.phase == InitialStopPhase.NAVIGATION

    assert latch.observe(stop) == LampAction.UNKNOWN
    assert latch.snapshot.stop_ignored
    assert latch.observe(left) == LampAction.LEFT

    latch.reset(initial_stop_armed=True, wait_for_signal=True)
    assert latch.observe(straight) == LampAction.STRAIGHT
    latch.reset(initial_stop_armed=True, wait_for_signal=True)
    assert latch.observe(left) == LampAction.LEFT


def test_initial_wait_fsm_stops_then_routes_confirmed_left_directly():
    fsm = TrafficShortcutFsm(
        one_shot_initial_stop=True,
        initial_left_direct_shortcut=True,
    )
    fsm.enable(initial_stop_armed=True, wait_for_signal=True)
    assert fsm.on_frame(
        LampAction.UNKNOWN,
        now_monotonic=0.0,
    ).publish_stop
    direct_left = fsm.on_frame(LampAction.LEFT, now_monotonic=0.1)
    assert direct_left.state == MissionState.SWITCH_TO_SHORTCUT
    assert direct_left.publish_stop
    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=0.2,
        ).policy
        == PolicyChoice.SHORTCUT
    )

    fsm.disable()
    fsm.enable(initial_stop_armed=True, wait_for_signal=True)
    straight = fsm.on_frame(LampAction.STRAIGHT, now_monotonic=0.3)
    assert straight.state == MissionState.BASE
    assert straight.policy == PolicyChoice.BASE

    fsm.disable()
    fsm.enable()
    assert (
        fsm.on_frame(
            LampAction.RED,
            now_monotonic=0.4,
        ).policy
        == PolicyChoice.BASE
    )


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


def _debug_snapshot(**overrides):
    values = {
        'schema_version': SIGNAL_DEBUG_SCHEMA_VERSION,
        'bundle_id': 'traffic-bundle',
        'frame_sequence': 12,
        'stamp_sec': 42,
        'stamp_nanosec': 123,
        'source': 'YOLO_CNN',
        'vote_updated': True,
        'raw_class': 'LEFT',
        'final_action': 'UNKNOWN',
        'class_labels': ('STOP', 'STRAIGHT', 'LEFT'),
        'probabilities': (0.02, 0.08, 0.90),
        'bbox': SignalDebugBbox(
            x=200.0,
            y=200.0,
            width=60.0,
            height=30.0,
            confidence=0.88,
        ),
        'crop': SignalDebugCrop(x1=190, y1=190, x2=270, y2=240),
        'width_gate_accepted': True,
        'candidate': 'LEFT',
        'candidate_reads': 3,
        'required_reads': 5,
        'phase': 'WAIT_FOR_SIGNAL',
        'mission_state': 'RED_STOP',
        'shortcut_status': 'READY',
        'detector_inference_ms': 12.5,
    }
    values.update(overrides)
    return SignalDebugSnapshot(**values)


def test_signal_debug_json_round_trip_and_source_contracts():
    snapshot = _debug_snapshot()

    encoded = encode_signal_debug(snapshot)

    assert decode_signal_debug(encoded) == snapshot
    assert '\n' not in encoded
    no_box = _debug_snapshot(
        source='YOLO_NO_BOX',
        raw_class='UNKNOWN',
        probabilities=None,
        bbox=None,
        crop=None,
        width_gate_accepted=False,
        candidate='UNKNOWN',
        candidate_reads=0,
        required_reads=0,
    )
    assert decode_signal_debug(encode_signal_debug(no_box)) == no_box
    with pytest.raises(
        SignalDebugContractError,
        match='must not contain classification geometry',
    ):
        encode_signal_debug(replace(snapshot, source='YOLO_NO_BOX'))
    cached = replace(snapshot, source='CACHED_CNN', vote_updated=False)
    assert decode_signal_debug(encode_signal_debug(cached)) == cached
    with pytest.raises(SignalDebugContractError, match='finite'):
        encode_signal_debug(
            replace(snapshot, detector_inference_ms=float('nan'))
        )
    with pytest.raises(SignalDebugContractError, match='shortcut status'):
        encode_signal_debug(replace(snapshot, shortcut_status='UNKNOWN'))


def test_monitor_matches_exact_stamp_and_never_overlays_another_frame():
    signal = _debug_snapshot()
    exact_image = np.zeros((300, 400, 3), dtype=np.uint8)
    exact_image[190:240, 190:270] = (10, 120, 240)
    latest_image = np.zeros((300, 400, 3), dtype=np.uint8)
    exact = MonitorFrame(
        stamp_key=signal.stamp_key,
        image=exact_image,
        received_monotonic=1.0,
    )
    latest = MonitorFrame(
        stamp_key=(43, 456),
        image=latest_image,
        received_monotonic=1.1,
    )
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _signal=None,
        _signal_received_monotonic=None,
        _matched_frame=None,
        _frames=OrderedDict(),
        _latest_frame=None,
        frame_buffer_size=30,
        _prediction=None,
        _actual=None,
        _enabled=True,
        _enabled_received_monotonic=1.2,
        _camera_error=None,
        _signal_error=None,
        _control_error=None,
    )

    TrafficShortcutMonitorNode._store_camera_frame_locked(node, exact)
    TrafficShortcutMonitorNode._store_signal_debug_locked(
        node,
        signal,
        received_monotonic=1.2,
    )
    matched = TrafficShortcutMonitorNode.snapshot(node)
    selection = select_signal_panel(
        matched,
        now_monotonic=1.5,
        signal_stale_sec=1.0,
    )
    exact_overlay = monitor_frame_panel(
        selection.display_frame.image,
        selection.overlay_signal,
        display_mode=selection.mode,
    )
    unmatched_overlay = monitor_frame_panel(
        latest_image,
        signal,
        display_mode=SignalPanelMode.UNMATCHED,
    )
    crop_panel = monitor_crop_panel(
        exact_image,
        signal,
        display_mode=SignalPanelMode.MATCHED,
    )

    assert matched.matched_frame is exact
    assert selection.mode == SignalPanelMode.MATCHED
    assert np.count_nonzero(exact_overlay[200, 200]) > 0
    assert np.count_nonzero(unmatched_overlay[200, 200]) == 0
    assert crop_panel.shape == (170, 880, 3)
    crop_body = crop_panel[32:]
    changed = np.argwhere(np.any(crop_body != 64, axis=2))
    display_height = int(changed[:, 0].max() - changed[:, 0].min() + 1)
    display_width = int(changed[:, 1].max() - changed[:, 1].min() + 1)
    assert display_width / display_height == pytest.approx(80 / 50, abs=0.04)
    assert frame_stamp_key(42, 123) == (42, 123)
    assert frame_stamp_key(0, 0) is None

    for index in range(31):
        latest = MonitorFrame(
            stamp_key=(100 + index, index),
            image=latest_image,
            received_monotonic=1.3 + index * 0.01,
        )
        TrafficShortcutMonitorNode._store_camera_frame_locked(node, latest)
    retained = TrafficShortcutMonitorNode.snapshot(node)
    assert signal.stamp_key not in node._frames
    assert retained.matched_frame is exact
    assert (
        select_signal_panel(
            retained,
            now_monotonic=1.9,
            signal_stale_sec=1.0,
        ).mode
        == SignalPanelMode.MATCHED
    )

    stale = select_signal_panel(
        retained,
        now_monotonic=2.3,
        signal_stale_sec=1.0,
    )
    assert stale.mode == SignalPanelMode.STALE
    assert stale.display_frame is exact
    assert stale.overlay_signal is signal
    stale_crop = monitor_crop_panel(
        stale.display_frame.image,
        stale.overlay_signal,
        display_mode=stale.mode,
    )
    assert stale_crop.shape == (170, 880, 3)
    assert float(stale_crop.mean()) > 0.0

    live_panel = monitor_live_panel(
        retained.latest_frame.image,
        camera_age=0.1,
        stamp_key=retained.latest_frame.stamp_key,
    )
    assert np.count_nonzero(live_panel[200, 200]) == 0

    delayed = MonitorFrame(
        stamp_key=(50, 1),
        image=np.full((300, 400, 3), 255, dtype=np.uint8),
        received_monotonic=2.0,
    )
    TrafficShortcutMonitorNode._store_camera_frame_locked(node, delayed)
    assert node._latest_frame is latest
    assert node._frames[delayed.stamp_key] is delayed


def test_monitor_matches_when_debug_arrives_before_camera_and_clears_mismatch():
    signal = _debug_snapshot()
    frame = MonitorFrame(
        stamp_key=signal.stamp_key,
        image=np.full((300, 400, 3), 100, dtype=np.uint8),
        received_monotonic=2.0,
    )
    node = SimpleNamespace(
        _signal=None,
        _signal_received_monotonic=None,
        _matched_frame=None,
        _frames=OrderedDict(),
        _latest_frame=None,
        frame_buffer_size=30,
    )

    TrafficShortcutMonitorNode._store_signal_debug_locked(
        node,
        signal,
        received_monotonic=2.1,
    )
    assert node._matched_frame is None
    TrafficShortcutMonitorNode._store_camera_frame_locked(node, frame)
    assert node._matched_frame is frame

    unmatched_signal = replace(
        signal,
        frame_sequence=13,
        stamp_sec=43,
    )
    TrafficShortcutMonitorNode._store_signal_debug_locked(
        node,
        unmatched_signal,
        received_monotonic=2.2,
    )
    assert node._matched_frame is None
    snapshot = TrafficShortcutMonitorSnapshot(
        signal=node._signal,
        signal_received_monotonic=node._signal_received_monotonic,
        matched_frame=node._matched_frame,
        latest_frame=node._latest_frame,
        prediction=None,
        actual=None,
        enabled=True,
        enabled_received_monotonic=2.0,
        camera_error=None,
        signal_error=None,
        control_error=None,
    )
    selection = select_signal_panel(
        snapshot,
        now_monotonic=2.3,
        signal_stale_sec=1.0,
    )
    assert selection.mode == SignalPanelMode.UNMATCHED
    assert selection.display_frame is None
    assert selection.overlay_signal is unmatched_signal
    placeholder = monitor_crop_panel(
        selection.display_frame,
        unmatched_signal,
        display_mode=selection.mode,
    )
    assert float(placeholder.mean()) > 0.0

    TrafficShortcutMonitorNode._store_signal_debug_locked(
        node,
        signal,
        received_monotonic=3.0,
    )
    assert node._signal is unmatched_signal
    assert node._signal_received_monotonic == 2.2


def test_drive_vector_uses_normalized_angle_and_speed_magnitude():
    origin = (100.0, 100.0)
    straight = drive_vector_endpoint(
        0.0,
        35.0,
        origin=origin,
        maximum_length=100.0,
        speed_scale=35.0,
    )
    left = drive_vector_endpoint(
        -100.0,
        35.0,
        origin=origin,
        maximum_length=100.0,
        speed_scale=35.0,
    )
    right = drive_vector_endpoint(
        100.0,
        35.0,
        origin=origin,
        maximum_length=100.0,
        speed_scale=35.0,
    )
    half_speed = drive_vector_endpoint(
        0.0,
        17.5,
        origin=origin,
        maximum_length=100.0,
        speed_scale=35.0,
    )

    assert straight == pytest.approx((100.0, 0.0))
    assert left[0] < origin[0] < right[0]
    assert math.dist(origin, left) == pytest.approx(100.0)
    assert math.dist(origin, right) == pytest.approx(100.0)
    assert math.dist(origin, half_speed) == pytest.approx(50.0)
    assert drive_vector_endpoint(
        50.0,
        0.0,
        origin=origin,
        maximum_length=100.0,
        speed_scale=35.0,
    ) == pytest.approx(origin)


def test_lamp_width_gate_five_votes_priority_and_latch_clearing():
    latch = TrafficLampLatch()
    red = _reading((255, 10, 220, 230), width=45)
    left = _reading((10, 10, 220, 230))
    green = _reading((10, 10, 10, 230))
    unknown = _reading((10, 10, 10, 10))

    assert (
        latch.observe(_reading((255, 10, 10, 10), width=44))
        == LampAction.UNKNOWN
    )
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
        latch.observe(_signal(SignalClass.RED, width=44)) == LampAction.UNKNOWN
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
    stop10_adaptive_human_bbox_classifier_vote = {
        **stabilized_human_bbox_classifier_vote,
        'consecutive_reads_by_raw_class': {
            'STOP': 10,
            'STRAIGHT': 15,
            'LEFT': 15,
        },
    }
    stop30_go30_adaptive_human_bbox_classifier_vote = {
        **stabilized_human_bbox_classifier_vote,
        'consecutive_reads_by_raw_class': {
            'STOP': 30,
            'STRAIGHT': 30,
            'LEFT': 30,
        },
    }
    stop10_go30_adaptive_human_bbox_classifier_vote = {
        **stabilized_human_bbox_classifier_vote,
        'consecutive_reads_by_raw_class': {
            'STOP': 10,
            'STRAIGHT': 30,
            'LEFT': 30,
        },
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
    assert (
        _expected_shortcut_artifact_id(
            schema_version=2,
            artifact_id='legacy',
        )
        == EXPECTED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_shortcut_artifact_id(
            schema_version=3,
            artifact_id=EXPECTED_EXPANDED_SIGNAL_VOTE_BUNDLE_ID,
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
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
    assert (
        _expected_shortcut_artifact_id(
            schema_version=5,
            artifact_id=EXPECTED_YOLO_MISSING_RELEASE_BUNDLE_ID,
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': human_bbox_classifier_vote},
        schema_version=6,
        artifact_id=EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == (2, 2, 2)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=6,
            artifact_id=EXPECTED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': stabilized_human_bbox_classifier_vote},
        schema_version=7,
        artifact_id=EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == (3, 15, 15)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=7,
            artifact_id=EXPECTED_STABILIZED_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': stabilized_human_bbox_classifier_vote},
        schema_version=8,
        artifact_id=EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == (3, 15, 15)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=8,
            artifact_id=EXPECTED_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': stop10_adaptive_human_bbox_classifier_vote},
        schema_version=9,
        artifact_id=EXPECTED_STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
    ) == (10, 15, 15)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=9,
            artifact_id=EXPECTED_STOP10_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID,
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': stop30_go30_adaptive_human_bbox_classifier_vote},
        schema_version=10,
        artifact_id=(
            EXPECTED_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (30, 30, 30)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=10,
            artifact_id=(
                EXPECTED_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': stop30_go30_adaptive_human_bbox_classifier_vote},
        schema_version=11,
        artifact_id=(
            EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (30, 30, 30)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=11,
            artifact_id=(
                EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=11,
            artifact_id=(
                EXPECTED_SPEED35_STOP30_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_BASE_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': stop10_go30_adaptive_human_bbox_classifier_vote},
        schema_version=12,
        artifact_id=(
            EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (10, 30, 30)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=12,
            artifact_id=(
                EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=12,
            artifact_id=(
                EXPECTED_SPEED35_STOP10_GO30_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_BASE_ARTIFACT_ID
    )
    stop15_go15_adaptive_human_bbox_classifier_vote = {
        **stabilized_human_bbox_classifier_vote,
        'consecutive_reads_by_raw_class': {
            'STOP': 15,
            'STRAIGHT': 15,
            'LEFT': 15,
        },
    }
    assert _load_signal_vote_contract(
        {'signal_vote': stop15_go15_adaptive_human_bbox_classifier_vote},
        schema_version=13,
        artifact_id=(
            EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (15, 15, 15)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=13,
            artifact_id=(
                EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=13,
            artifact_id=(
                EXPECTED_SPEED35_STOP15_GO15_ADAPTIVE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_BASE_ARTIFACT_ID
    )
    initial_stop_once_vote = {
        'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
        'consecutive_reads_by_raw_class': {
            'STOP': 15,
            'STRAIGHT': 15,
            'LEFT': 15,
        },
        'unknown_behavior': 'reset_candidate',
        'different_raw_class_behavior': 'restart_candidate_at_one',
        'stop_classes': ['STOP'],
        'stop_vote_behavior': 'only_while_initial_stop_armed',
        'post_initial_stop_behavior': 'ignore_stop',
        'navigation_actions': ['LEFT', 'STRAIGHT'],
    }
    assert _load_signal_vote_contract(
        {'signal_vote': initial_stop_once_vote},
        schema_version=14,
        artifact_id=(
            EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (15, 15, 15)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=14,
            artifact_id=(
                EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=14,
            artifact_id=(
                EXPECTED_SPEED35_INITIAL_STOP_ONCE_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_BASE_ARTIFACT_ID
    )
    initial_wait_fresh5_vote = {
        'raw_classes': ['STOP', 'STRAIGHT', 'LEFT'],
        'consecutive_reads_by_raw_class': {
            'STOP': 5,
            'STRAIGHT': 5,
            'LEFT': 5,
        },
        'unknown_behavior': 'reset_candidate',
        'different_raw_class_behavior': 'restart_candidate_at_one',
        'stop_classes': ['STOP'],
        'stop_vote_behavior': 'only_while_initial_stop_armed',
        'post_initial_stop_behavior': 'ignore_stop',
        'navigation_actions': ['LEFT', 'STRAIGHT'],
        'control_vote_source': 'fresh_yolo_classifier_only',
        'cached_classifier_behavior': 'diagnostics_only',
    }
    assert _load_signal_vote_contract(
        {'signal_vote': initial_wait_fresh5_vote},
        schema_version=15,
        artifact_id=(
            EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (5, 5, 5)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=15,
            artifact_id=(
                EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=15,
            artifact_id=(
                EXPECTED_SPEED35_INITIAL_WAIT_FRESH5_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_BASE_ARTIFACT_ID
    )
    initial_wait_fresh3_vote = {
        **initial_wait_fresh5_vote,
        'consecutive_reads_by_raw_class': {
            'STOP': 5,
            'STRAIGHT': 3,
            'LEFT': 3,
        },
        'cached_classifier_behavior': 'disabled',
    }
    assert _load_signal_vote_contract(
        {'signal_vote': initial_wait_fresh3_vote},
        schema_version=16,
        artifact_id=(
            EXPECTED_SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (5, 3, 3)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=16,
            artifact_id=(
                EXPECTED_SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=16,
            artifact_id=(
                EXPECTED_SPEED35_INITIAL_WAIT_FRESH3_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_BASE_ARTIFACT_ID
    )
    initial_wait_go1_vote = {
        **initial_wait_fresh3_vote,
        'consecutive_reads_by_raw_class': {
            'STOP': 5,
            'STRAIGHT': 1,
            'LEFT': 1,
        },
    }
    assert _load_signal_vote_contract(
        {'signal_vote': initial_wait_go1_vote},
        schema_version=17,
        artifact_id=(
            EXPECTED_SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (5, 1, 1)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=17,
            artifact_id=(
                EXPECTED_SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=17,
            artifact_id=(
                EXPECTED_SPEED35_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_BASE_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': initial_wait_go1_vote},
        schema_version=18,
        artifact_id=(
            EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (5, 1, 1)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=18,
            artifact_id=(
                EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=18,
            artifact_id=(
                EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_FIX_BASE_ARTIFACT_ID
    )
    assert _load_signal_vote_contract(
        {'signal_vote': initial_wait_go1_vote},
        schema_version=19,
        artifact_id=(
            EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
        ),
    ) == (5, 1, 1)
    assert (
        _expected_shortcut_artifact_id(
            schema_version=19,
            artifact_id=(
                EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_EXPANDED_SHORTCUT_ARTIFACT_ID
    )
    assert (
        _expected_base_artifact_id(
            schema_version=19,
            artifact_id=(
                EXPECTED_SPEED35_FIX_INITIAL_WAIT_GO1_SESSION_HUMAN_BBOX_CLASSIFIER_BUNDLE_ID
            ),
        )
        == EXPECTED_SPEED35_FIX_BASE_ARTIFACT_ID
    )
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
    assert (
        fsm.on_frame(
            LampAction.RED,
            now_monotonic=0.0,
        ).state
        == MissionState.RED_STOP
    )

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

    assert (
        counter.observe(
            red_stop_active=True,
            detector_observed=True,
            yolo_box_found=True,
        )
        is False
    )
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


def test_signal_status_log_changes_immediately_and_heartbeats_at_two_hz():
    gate = SignalStatusLogGate(rate_hz=2.0)
    assert gate.should_emit(SignalClass.STOP, now_monotonic=1.0)
    assert not gate.should_emit(SignalClass.STOP, now_monotonic=1.49)
    assert gate.should_emit(SignalClass.STOP, now_monotonic=1.5)
    assert gate.should_emit(SignalClass.LEFT, now_monotonic=1.51)

    line = format_signal_status(
        reading=_signal(SignalClass.LEFT),
        raw_class=SignalClass.LEFT,
        source='CACHED_CNN',
        phase='NAVIGATION',
        candidate_reads=7,
        required_reads=15,
        stop_status='IGNORED',
        vote_updated=False,
    )
    assert line == (
        'SIGNAL raw=LEFT cnn=90.0% yolo=90.0% source=CACHED_CNN '
        'phase=NAVIGATION vote=7/15 vote_update=NO stop=IGNORED'
    )


def test_initial_wait_terminal_events_ignore_cached_vote_updates():
    latch = InitialWaitSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        stop_consecutive_reads=5,
        left_consecutive_reads=5,
        straight_consecutive_reads=5,
    )
    latch.reset(initial_stop_armed=True, wait_for_signal=True)
    fsm = TrafficShortcutFsm(
        one_shot_initial_stop=True,
        initial_left_direct_shortcut=True,
    )
    fsm.enable(initial_stop_armed=True, wait_for_signal=True)
    node = SimpleNamespace(
        _lamp_latch=latch,
        bundle=SimpleNamespace(
            schema_version=15,
            detector=SimpleNamespace(
                red_consecutive_reads=5,
                left_consecutive_reads=5,
            ),
        ),
        _ready_logged=False,
        _stop_ignored_logged=False,
        _signal_log_gate=SignalStatusLogGate(rate_hz=2.0),
    )
    reading = _signal(SignalClass.LEFT)
    log_method = TrafficShortcutPolicyNode._one_shot_signal_logs_locked

    previous = latch.snapshot
    signal = latch.observe(reading)
    plan = fsm.on_frame(signal, now_monotonic=1.0)
    first = log_method(
        node,
        previous=previous,
        reading=reading,
        raw_class=SignalClass.LEFT,
        source='YOLO_CNN',
        signal=signal,
        plan=plan,
        now_monotonic=1.0,
        vote_updated=True,
    )
    assert any(message.startswith('READY raw=LEFT') for message in first)
    assert 'NON_STOP_CLEAR 1/5' in first
    assert any('vote=1/5 vote_update=YES' in message for message in first)

    cached = log_method(
        node,
        previous=latch.snapshot,
        reading=reading,
        raw_class=SignalClass.LEFT,
        source='CACHED_CNN',
        signal=signal,
        plan=plan,
        now_monotonic=1.5,
        vote_updated=False,
    )
    assert latch.snapshot.candidate_reads == 1
    assert not any(message.startswith('READY') for message in cached)
    assert not any(message.startswith('NON_STOP_CLEAR') for message in cached)
    assert any('vote=1/5 vote_update=NO' in message for message in cached)

    final = []
    for step in range(2, 6):
        previous = latch.snapshot
        signal = latch.observe(reading)
        plan = fsm.on_frame(signal, now_monotonic=1.5 + step)
        final = log_method(
            node,
            previous=previous,
            reading=reading,
            raw_class=SignalClass.LEFT,
            source='YOLO_CNN',
            signal=signal,
            plan=plan,
            now_monotonic=1.5 + step,
            vote_updated=True,
        )
    assert plan.state == MissionState.SWITCH_TO_SHORTCUT
    assert 'LEFT_CONFIRMED vote=5/5' in final
    assert not any(message.startswith('READY') for message in final)


def test_completed_activation_logs_left_ignored_once_and_status():
    latch = InitialWaitSignalLatch(
        bbox_width_min=40,
        bbox_width_max=225,
        stop_consecutive_reads=5,
        left_consecutive_reads=1,
        straight_consecutive_reads=1,
    )
    latch.reset(initial_stop_armed=False, wait_for_signal=False)
    fsm = TrafficShortcutFsm(
        one_shot_initial_stop=True,
        initial_left_direct_shortcut=True,
    )
    fsm.enable()
    fsm.shortcut_completed = True
    node = SimpleNamespace(
        _lamp_latch=latch,
        _fsm=fsm,
        bundle=SimpleNamespace(
            schema_version=19,
            detector=SimpleNamespace(
                red_consecutive_reads=5,
                left_consecutive_reads=1,
            ),
        ),
        _ready_logged=False,
        _stop_ignored_logged=False,
        _left_ignored_logged=False,
        _signal_log_gate=SignalStatusLogGate(rate_hz=2.0),
    )
    reading = _signal(SignalClass.LEFT)
    previous = latch.snapshot
    signal = latch.observe(reading)
    plan = fsm.on_frame(signal, now_monotonic=1.0)

    messages = TrafficShortcutPolicyNode._one_shot_signal_logs_locked(
        node,
        previous=previous,
        reading=reading,
        raw_class=SignalClass.LEFT,
        source='YOLO_CNN',
        signal=signal,
        plan=plan,
        now_monotonic=1.0,
        vote_updated=True,
    )
    repeated = TrafficShortcutPolicyNode._one_shot_signal_logs_locked(
        node,
        previous=latch.snapshot,
        reading=reading,
        raw_class=SignalClass.LEFT,
        source='YOLO_CNN',
        signal=signal,
        plan=plan,
        now_monotonic=1.1,
        vote_updated=True,
    )

    assert 'LEFT_IGNORED reason=SESSION_COMPLETE action=RELEASE_A' in messages
    assert not any(message.startswith('LEFT_IGNORED') for message in repeated)
    assert (
        TrafficShortcutPolicyNode._shortcut_status_locked(
            node,
            MissionState.BASE,
        )
        == 'COMPLETED_THIS_ACTIVATION'
    )


def test_policy_signal_debug_snapshot_contains_runtime_vote_and_geometry():
    latch_snapshot = InitialStopSignalLatchSnapshot(
        candidate=SignalClass.LEFT,
        candidate_reads=2,
        required_reads=5,
        stop_latched=False,
        phase=InitialStopPhase.WAIT_FOR_SIGNAL,
        raw_class=SignalClass.LEFT,
        stop_ignored=False,
    )
    node = SimpleNamespace(
        _lamp_latch=SimpleNamespace(snapshot=latch_snapshot),
        _fsm=SimpleNamespace(shortcut_completed=False),
        _shortcut_status_locked=lambda state: (
            TrafficShortcutPolicyNode._shortcut_status_locked(node, state)
        ),
        bundle=SimpleNamespace(
            artifact_id='traffic-bundle',
            detector=SimpleNamespace(
                classifier_classes=('STOP', 'STRAIGHT', 'LEFT'),
                bbox_width_min=40,
                bbox_width_max=225,
            ),
        ),
    )
    inspection = SignalInspection(
        reading=SignalReading(
            signal_class=SignalClass.LEFT,
            probability=0.9,
            bbox_width=60.0,
            confidence=0.8,
        ),
        bbox=DetectionBox(10.5, 20.5, 60.0, 30.0, 0.8),
        crop_bounds=ImageBounds(5, 15, 80, 60),
        probabilities=(0.02, 0.08, 0.90),
    )

    snapshot = TrafficShortcutPolicyNode._make_signal_debug_snapshot_locked(
        node,
        frame_sequence=9,
        stamp_sec=7,
        stamp_nanosec=8,
        source='CACHED_CNN',
        vote_updated=False,
        raw_class=SignalClass.LEFT,
        final_action=LampAction.UNKNOWN,
        inspection=inspection,
        plan=SimpleNamespace(state=MissionState.RED_STOP),
        detector_inference_ms=4.5,
    )

    assert snapshot.stamp_key == (7, 8)
    assert snapshot.phase == InitialStopPhase.WAIT_FOR_SIGNAL.value
    assert snapshot.candidate_reads == 2
    assert snapshot.required_reads == 5
    assert snapshot.bbox == SignalDebugBbox(10.5, 20.5, 60.0, 30.0, 0.8)
    assert snapshot.crop == SignalDebugCrop(5, 15, 80, 60)
    assert decode_signal_debug(encode_signal_debug(snapshot)) == snapshot


def test_signal_debug_publish_is_subscriber_gated_and_motion_isolated():
    class Publisher:
        def __init__(self, subscriptions, *, fail=False):
            self.subscriptions = subscriptions
            self.fail = fail
            self.messages = []

        def get_subscription_count(self):
            return self.subscriptions

        def publish(self, message):
            if self.fail:
                raise RuntimeError('diagnostic transport failed')
            self.messages.append(message)

    class BrokenLogger:
        def warning(self, *args, **kwargs):
            raise RuntimeError('logger failed')

    no_subscriber = Publisher(0)
    node = SimpleNamespace(
        signal_debug_publisher=no_subscriber,
        get_logger=lambda: BrokenLogger(),
    )
    TrafficShortcutPolicyNode._publish_signal_debug(node, object())
    assert no_subscriber.messages == []

    publisher = Publisher(1)
    node.signal_debug_publisher = publisher
    snapshot = _debug_snapshot()
    TrafficShortcutPolicyNode._publish_signal_debug(node, snapshot)
    assert decode_signal_debug(publisher.messages[0].data) == snapshot

    node.signal_debug_publisher = Publisher(1, fail=True)
    TrafficShortcutPolicyNode._publish_signal_debug(node, snapshot)


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


def test_fsm_rearms_successful_shortcut_on_new_drive_activation():
    fsm = TrafficShortcutFsm(
        shortcut_duration_sec=8.0,
        seamless_base_handoff=True,
        one_shot_initial_stop=True,
        initial_left_direct_shortcut=True,
        rearm_shortcut_on_enable=True,
    )
    fsm.enable()
    assert fsm.on_frame(
        LampAction.LEFT,
        now_monotonic=1.0,
    ).publish_stop
    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=1.1,
        ).policy
        == PolicyChoice.SHORTCUT
    )
    fsm.on_shortcut_command_published(now_monotonic=2.0)
    assert fsm.on_control_tick(
        now_monotonic=10.0,
    ).promote_base_shadow
    fsm.on_base_shadow_promoted()
    assert fsm.shortcut_completed
    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=10.1,
        ).policy
        == PolicyChoice.BASE
    )

    fsm.disable()
    fsm.enable()

    assert not fsm.shortcut_completed
    retry = fsm.on_frame(LampAction.LEFT, now_monotonic=11.0)
    assert retry.publish_stop
    assert retry.state == MissionState.SWITCH_TO_SHORTCUT


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
    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=1.1,
        ).policy
        == PolicyChoice.SHORTCUT
    )
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
    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=1.3,
        ).policy
        == PolicyChoice.SHORTCUT
    )


def test_fsm_off_fault_and_red_priority_paths_stop():
    fsm = TrafficShortcutFsm()
    assert fsm.on_frame(LampAction.LEFT, now_monotonic=0.0).publish_stop
    fsm.enable()
    assert (
        fsm.on_frame(LampAction.RED, now_monotonic=0.1).state
        == MissionState.RED_STOP
    )
    assert (
        fsm.on_frame(LampAction.STRAIGHT, now_monotonic=0.2).policy
        == PolicyChoice.BASE
    )
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
        inference_timeout_sec=0.50,
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
        bundle=SimpleNamespace(base=SimpleNamespace(speed_output_max=30.0)),
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

    speed35_history_node = SimpleNamespace(
        bundle=SimpleNamespace(base=SimpleNamespace(speed_output_max=35.0)),
        _history=deque([(50, 85)] * 4, maxlen=4),
        _last_executed_decision_sequence=0,
        _publish=lambda _command: None,
    )
    method(
        speed35_history_node,
        DriveCommand(20.0, 35.0),
        decision_sequence=8,
    )
    assert tuple(speed35_history_node._history)[-1] == (60, 85)


def test_integrated_node_policy_freshness_allows_half_second_then_faults():
    node = _safe_node_state()
    node._decision = replace(node._decision, source_monotonic=0.50)
    assert TrafficShortcutPolicyNode._unsafe_reason_locked(node, 1.0) is None

    reason = TrafficShortcutPolicyNode._unsafe_reason_locked(node, 1.001)

    assert 'selected policy inference is stale' in reason
    assert 'age=0.501s' in reason
    assert 'limit=0.500s' in reason


def test_transition_stop_is_published_once_before_shortcut_decision():
    published = []
    events = []
    fsm = TrafficShortcutFsm(
        one_shot_initial_stop=True,
        initial_left_direct_shortcut=True,
    )
    fsm.enable()
    assert fsm.on_frame(
        LampAction.LEFT,
        now_monotonic=1.0,
    ).publish_stop
    node = SimpleNamespace(
        _next_graph_check_monotonic=math.inf,
        require_gamepad_hold=True,
        _drive_gate=SimpleNamespace(enabled=True),
        _lock=threading.RLock(),
        _unsafe_reason_locked=lambda _now: None,
        _fsm=fsm,
        _transition_stop_pending=True,
        _transition_stop_sent_waiting_decision=False,
        _mission_generation=0,
        _frame_sequence=3,
        _minimum_next_frame_sequence=0,
        _decision=None,
        _awaiting_post_reset_decision=False,
        _history_reset_monotonic=None,
        _stop_reason=None,
        _publish_and_record_locked=lambda command, decision_sequence: (
            published.append((command, decision_sequence))
        ),
        _emit_operator_logs=lambda messages: events.extend(messages),
        _publish_stop=lambda: published.append(('fallback-stop', None)),
    )

    TrafficShortcutPolicyNode._on_control_timer(node)
    TrafficShortcutPolicyNode._on_control_timer(node)

    assert published == [(DriveCommand(), None)]
    assert events == ['TRANSITION_STOP cycle=1/1']

    assert (
        fsm.on_frame(
            LampAction.LEFT,
            now_monotonic=1.1,
        ).policy
        == PolicyChoice.SHORTCUT
    )
    node._transition_stop_sent_waiting_decision = False
    node._awaiting_post_reset_decision = False
    node._decision = MissionDecision(
        command=DriveCommand(-20.0, 23.0),
        policy=PolicyChoice.SHORTCUT,
        state=MissionState.SHORTCUT,
        source_monotonic=1.1,
        completed_monotonic=1.12,
        inference_ms=20.0,
        frame_sequence=6,
    )

    TrafficShortcutPolicyNode._on_control_timer(node)

    assert published[-1] == (DriveCommand(-20.0, 23.0), 6)
    assert len(published) == 2


def test_integrated_node_without_gamepad_does_not_require_joy():
    node = _safe_node_state()
    node.require_gamepad_hold = False
    node._joy_valid = False
    node._last_joy_monotonic = None

    assert TrafficShortcutPolicyNode._unsafe_reason_locked(node, 1.0) is None
    assert TrafficShortcutPolicyNode._can_enable_locked(node, 1.0)


def test_integrated_node_maps_sdl_lb_button_nine_to_initial_wait():
    observations = []
    operator_logs = []

    def observe(_self, **kwargs):
        observations.append(kwargs)
        return ToggleAction.ENABLED

    node = SimpleNamespace(
        require_gamepad_hold=True,
        bundle=SimpleNamespace(schema_version=16),
        a_button_index=0,
        initial_stop_arm_button_index=9,
        _lock=threading.Lock(),
        _joy_valid=False,
        _last_joy_monotonic=None,
        _observe_drive_gate_locked=lambda **kwargs: observe(None, **kwargs),
        _force_fault=lambda _reason: None,
        get_logger=lambda: SimpleNamespace(
            warning=lambda *_args, **_kwargs: None
        ),
        _safe_log_warning=operator_logs.append,
    )
    buttons = [0] * 10
    buttons[0] = 1
    buttons[9] = 1

    TrafficShortcutPolicyNode._on_joy(
        node,
        SimpleNamespace(buttons=buttons),
    )

    assert observations[-1]['pressed']
    assert observations[-1]['initial_stop_armed']
    assert observations[-1]['wait_for_signal']
    assert operator_logs[-1] == 'WAIT_FOR_SIGNAL source=LB+A'

    buttons[9] = 0
    buttons[4] = 1
    TrafficShortcutPolicyNode._on_joy(
        node,
        SimpleNamespace(buttons=buttons),
    )
    assert not observations[-1]['initial_stop_armed']
    assert not observations[-1]['wait_for_signal']


def test_integrated_node_initial_wait_control_tick_publishes_stop():
    published = []
    fsm = TrafficShortcutFsm(
        one_shot_initial_stop=True,
        initial_left_direct_shortcut=True,
    )
    fsm.enable(initial_stop_armed=True, wait_for_signal=True)
    node = SimpleNamespace(
        _next_graph_check_monotonic=math.inf,
        require_gamepad_hold=True,
        _drive_gate=SimpleNamespace(enabled=True),
        _lock=threading.Lock(),
        _unsafe_reason_locked=lambda _now: None,
        _fsm=fsm,
        _transition_stop_pending=False,
        _transition_stop_sent_waiting_decision=False,
        _decision=None,
        _publish_stop=lambda: published.append(DriveCommand()),
    )

    TrafficShortcutPolicyNode._on_control_timer(node)

    assert fsm.state == MissionState.WAIT_FOR_SIGNAL
    assert published == [DriveCommand()]


def test_traffic_shortcut_launch_ties_gamepad_to_hold_gate():
    launch_path = (
        Path(__file__).parents[1]
        / 'launch'
        / 'traffic_shortcut_policy.launch.py'
    )
    launch_text = launch_path.read_text(encoding='utf-8')

    assert "'require_gamepad_hold': ParameterValue(" in launch_text
    assert 'use_gamepad,' in launch_text
    assert "'initial_stop_arm_button_index'" in launch_text
    assert "default_value='9'" in launch_text
    assert "'signal_status_log_hz'" in launch_text
    assert "'use_monitor_gui'" in launch_text
    assert "default_value='false'" in launch_text
    assert "'monitor_refresh_hz'" in launch_text
    assert "default_value='15.0'" in launch_text
    assert "'inference_timeout_sec'" in launch_text
    assert "default_value='0.50'" in launch_text
    assert "default_value='0.40'" in launch_text
    assert "executable='traffic_shortcut_monitor'" in launch_text
    assert 'IfCondition(use_monitor_gui)' in launch_text
    assert 'traffic shortcut monitor exited; stopping mission' in launch_text
    assert 'traffic shortcut gamepad exited; stopping mission' in launch_text

    jetson_launch_text = (
        Path(__file__).parents[1]
        / 'launch'
        / 'jetson_traffic_shortcut.launch.py'
    ).read_text(encoding='utf-8')
    assert "'initial_stop_arm_button_index:='" in jetson_launch_text
    assert "default_value='9'" in jetson_launch_text
    assert "'signal_status_log_hz:='" in jetson_launch_text
    assert "'use_monitor_gui:='" in jetson_launch_text
    assert "'monitor_refresh_hz:='" in jetson_launch_text
    assert "'inference_timeout_sec:='" in jetson_launch_text
    assert "default_value='0.50'" in jetson_launch_text
    assert "default_value='0.40'" in jetson_launch_text

    camera_launch_text = (
        Path(__file__).parents[3]
        / 'src'
        / 'xycar_device'
        / 'xycar_cam'
        / 'launch'
        / 'xycar_cam.launch.py'
    ).read_text(encoding='utf-8')
    assert 'front camera exited; stopping dependent launch' in (
        camera_launch_text
    )


def test_traffic_shortcut_wrapper_rejects_orphans_and_cleans_process_group():
    wrapper_path = (
        Path(__file__).parents[3]
        / 'deploy'
        / 'jetson'
        / 'run_gpu_traffic_shortcut.sh'
    )
    wrapper_text = wrapper_path.read_text(encoding='utf-8')

    assert "'(^|/)traffic_shortcut_policy([[:space:]]|$)'" in wrapper_text
    assert 'kill -0 -- "-${INNER_LAUNCH_PID}"' in wrapper_text
    assert 'kill -KILL -- "-${INNER_LAUNCH_PID}"' in wrapper_text
    assert "trap 'exit 129' HUP" in wrapper_text
    assert 'if [ "${CONTAINER_STARTED}" = true ]; then' in wrapper_text
    assert '--history-reset-timeout-sec 0.50' in wrapper_text
    assert 'CONTAINER_STARTED=true\ndocker run --detach --rm' in wrapper_text
    assert 'INNER_LAUNCH_PID=""\nexit "${STATUS}"' not in wrapper_text


def test_traffic_light_viewer_launch_has_no_motion_endpoints():
    package_root = Path(__file__).parents[1]
    launch_text = (
        package_root / 'launch' / 'traffic_light_viewer.launch.py'
    ).read_text(encoding='utf-8')
    viewer_text = (
        package_root / 'xycar_ai_drive' / 'traffic_light_viewer.py'
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


def test_traffic_shortcut_monitor_is_passive_and_has_no_inference_runtime():
    package_root = Path(__file__).parents[1]
    monitor_text = (
        package_root / 'xycar_ai_drive' / 'traffic_shortcut_monitor.py'
    ).read_text(encoding='utf-8')
    policy_text = (
        package_root / 'xycar_ai_drive' / 'traffic_shortcut_policy_node.py'
    ).read_text(encoding='utf-8')
    setup_text = (package_root / 'setup.py').read_text(encoding='utf-8')
    config_text = (
        package_root / 'config' / 'traffic_shortcut_policy.yaml'
    ).read_text(encoding='utf-8')

    assert '.create_publisher(' not in monitor_text
    assert 'create_onnx_detector' not in monitor_text
    assert 'TrafficClassifierDetector' not in monitor_text
    assert 'UnixSocketPolicyClient' not in monitor_text
    assert 'create_subscription(' in monitor_text
    assert '/xycar_motor' in monitor_text
    assert '/traffic_shortcut/signal_debug' in monitor_text
    assert 'traffic_shortcut_monitor:main' in setup_text
    assert 'frame_buffer_size: 30' in config_text
    assert 'monitor_refresh_hz: 15.0' in config_text
    assert 'signal_debug_topic: /traffic_shortcut/signal_debug' in config_text
    assert 'SingleThreadedExecutor' in monitor_text
    assert 'rclpy.spin_once' not in monitor_text
    assert 'LIVE CAMERA' in monitor_text
    assert 'MODEL INPUT (EXACT)' in monitor_text
    assert (
        'self.signal_debug_publisher = self.create_publisher(' in policy_text
    )
    assert 'get_subscription_count() < 1' in policy_text


@pytest.mark.parametrize(
    ('schema_version', 'votes', 'classification_every', 'reuse_bbox'),
    [
        (11, (30, 30, 30), 1, True),
        (12, (10, 30, 30), 1, True),
        (13, (15, 15, 15), 1, True),
        (14, (15, 15, 15), 1, True),
        (15, (5, 5, 5), 1, True),
        (16, (5, 3, 3), 3, False),
        (17, (5, 1, 1), 3, False),
        (18, (5, 1, 1), 3, False),
        (19, (5, 1, 1), 3, False),
    ],
)
def test_traffic_light_viewer_accepts_speed35_bundle_contracts(
    schema_version,
    votes,
    classification_every,
    reuse_bbox,
):
    node = SimpleNamespace(
        bundle=SimpleNamespace(
            schema_version=schema_version,
            detector=SimpleNamespace(
                mode='yolo_cnn_classifier',
                bbox_width_min=40,
                bbox_width_max=225,
                inference_every_n_frames=3,
                classification_every_n_frames_after_detection=(
                    classification_every
                ),
                reuse_detected_bbox_between_yolo_frames=reuse_bbox,
                red_consecutive_reads=votes[0],
                left_consecutive_reads=votes[1],
                straight_consecutive_reads=votes[2],
            ),
        ),
    )

    TrafficLightViewerNode._validate_bundle_contract(node)


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
                base=SimpleNamespace(speed_output_max=30.0),
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
                base_shadow_max_age_sec=0.50,
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
    node, published, _predictions = _shadow_handoff_node(source_monotonic=9.0)

    reason = node._promote_base_shadow_locked(10.0)

    assert 'stale' in reason
    assert not published
    assert node._fsm.state == MissionState.SWITCH_TO_BASE
    node._discard_base_shadow_locked()
    assert node._base_shadow_history is None
    assert node._base_shadow_decision is None
