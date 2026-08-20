# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from xycar_ai_drive.traffic_light_detector import (
    DetectionBox,
    LampAction,
    LampReading,
    TrafficLampLatch,
    TrafficLightDetector,
    TrafficLightError,
    decode_detection_box,
    lamp_scores,
)
from xycar_ai_drive.traffic_shortcut_fsm import (
    MissionState,
    PolicyChoice,
    TrafficShortcutFsm,
)
from xycar_ai_drive.traffic_shortcut_policy_node import (
    MissionDecision,
    TrafficShortcutPolicyNode,
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
    output = np.zeros((1, 5, 8400), dtype=np.float32)
    output[0, 0, 0] = np.nan
    with pytest.raises(TrafficLightError, match='NaN or Inf'):
        decode_detection_box(
            output,
            frame_height=480,
            frame_width=640,
        )


def test_lamp_width_gate_red_votes_priority_and_latch_clearing():
    latch = TrafficLampLatch()
    red = _reading((255, 10, 220, 230))
    left = _reading((10, 10, 220, 230))
    green = _reading((10, 10, 10, 230))
    unknown = _reading((10, 10, 10, 10))

    assert latch.observe(_reading((255, 10, 10, 10), width=44)) == LampAction.UNKNOWN
    assert latch.observe(red) == LampAction.UNKNOWN
    assert latch.observe(red) == LampAction.UNKNOWN
    assert latch.observe(red) == LampAction.RED
    assert latch.red_latched
    assert latch.observe(None) == LampAction.RED
    assert latch.observe(unknown) == LampAction.RED
    assert latch.observe(left) == LampAction.LEFT
    assert not latch.red_latched
    assert latch.observe(green) == LampAction.STRAIGHT


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
