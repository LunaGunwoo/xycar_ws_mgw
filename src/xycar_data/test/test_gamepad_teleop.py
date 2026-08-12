# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import math
from types import SimpleNamespace

import pytest

from xycar_data.gamepad_teleop import (
    DriveCommand,
    GamepadConfig,
    GamepadTeleopNode,
    InvalidJoyInput,
    NeutralArmingGate,
    RecordingAction,
    RecordingGate,
    RecordingState,
    _PendingRecordingFinish,
    _is_paired_unnamed_relay,
    _validate_recording_parameters,
    _validate_runtime_parameters,
    is_input_fresh,
    map_joy_axes,
)


def _endpoint(participant: bytes, entity: bytes, *, named: bool = False):
    return SimpleNamespace(
        endpoint_gid=list(participant + entity),
        node_name='publisher' if named else '_NODE_NAME_UNKNOWN_',
        node_namespace='/' if named else '_NODE_NAMESPACE_UNKNOWN_',
    )


def test_only_matching_unnamed_dds_pair_is_a_motor_relay():
    participant = bytes(range(12))
    publisher = _endpoint(participant, b'\x00\x00\x12\x03')
    matching_subscription = _endpoint(
        participant,
        b'\x00\x00\x13\x04',
    )
    other_subscription = _endpoint(
        bytes(reversed(range(12))),
        b'\x00\x00\x13\x04',
    )

    assert _is_paired_unnamed_relay(
        publisher,
        [matching_subscription],
    )
    assert not _is_paired_unnamed_relay(
        publisher,
        [other_subscription],
    )
    assert not _is_paired_unnamed_relay(
        _endpoint(participant, b'\x00\x00\x12\x03', named=True),
        [matching_subscription],
    )


def test_neutral_input_stops_with_centered_steering():
    axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert map_joy_axes(axes) == DriveCommand(0.0, 0.0)


@pytest.mark.parametrize(
    ('steering', 'expected_angle'),
    [(-1.0, 100.0), (-0.5, 50.0), (0.5, -50.0), (1.0, -100.0)],
)
def test_steering_maps_to_full_angle_range(steering, expected_angle):
    axes = [steering, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert map_joy_axes(axes).angle == expected_angle


def test_lt_maps_to_reverse_speed():
    command = map_joy_axes([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    assert command == DriveCommand(0.0, -5.0)


def test_rt_maps_to_forward_speed():
    command = map_joy_axes([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    assert command == DriveCommand(0.0, 7.0)


def test_triggers_are_combined_for_partial_and_simultaneous_input():
    partial = map_joy_axes([0.0, 0.0, 0.0, 0.0, -0.4, -0.5])
    both_full = map_joy_axes([0.0, 0.0, 0.0, 0.0, -1.0, -1.0])
    assert partial.speed == pytest.approx(1.5)
    assert both_full.speed == pytest.approx(2.0)


@pytest.mark.parametrize(
    ('raw_axis', 'expected_depth'),
    [
        (1.0, 0.0),
        (0.5, 0.25),
        (0.0, 0.5),
        (-0.5, 0.75),
        (-1.0, 1.0),
    ],
)
def test_signed_trigger_profile_uses_full_axis_range(
    raw_axis,
    expected_depth,
):
    config = GamepadConfig(trigger_axis_mode='signed')
    command = map_joy_axes(
        [0.0, 0.0, 0.0, 0.0, 1.0, raw_axis],
        config,
    )
    assert command.speed == pytest.approx(7.0 * expected_depth)


def test_positive_trigger_profile_maps_depth_to_signed_speed():
    config = GamepadConfig(trigger_axis_mode='positive')
    partial = map_joy_axes(
        [0.0, 0.0, 0.0, 0.0, 0.4, 0.5],
        config,
    )
    both_full = map_joy_axes(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        config,
    )
    assert partial.speed == pytest.approx(1.5)
    assert both_full.speed == pytest.approx(2.0)


def test_default_negative_trigger_profile_maps_vehicle_input():
    command = map_joy_axes([0.0, 0.0, 0.0, 0.0, -0.4, -0.5])
    assert command.speed == pytest.approx(1.5)


def test_steering_is_preserved_while_speed_is_zero():
    command = map_joy_axes([0.75, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert command == DriveCommand(-75.0, 0.0)


def test_axes_are_clamped_before_mapping():
    command = map_joy_axes([2.0, 0.0, 0.0, 0.0, -2.0, 2.0])
    assert command == DriveCommand(-100.0, -5.0)


@pytest.mark.parametrize(
    'axes',
    [
        [0.0] * 5,
        [math.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, math.inf, 0.0],
    ],
)
def test_invalid_axes_are_rejected(axes):
    with pytest.raises(InvalidJoyInput):
        map_joy_axes(axes)


def test_custom_mapping_and_limits_are_supported():
    config = GamepadConfig(
        steering_axis=2,
        lt_axis=0,
        rt_axis=1,
        invert_steering=False,
        max_angle=30.0,
        max_reverse_speed=3.0,
        max_forward_speed=4.0,
    )
    command = map_joy_axes([-0.25, -0.5, -0.5], config)
    assert command == DriveCommand(-15.0, 1.25)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('max_angle', math.nan),
        ('max_angle', math.inf),
        ('max_reverse_speed', -math.inf),
        ('max_forward_speed', math.nan),
    ],
)
def test_non_finite_output_config_is_rejected(field, value):
    config = GamepadConfig(**{field: value})
    with pytest.raises(ValueError, match='finite and positive'):
        map_joy_axes([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], config)


def test_unknown_trigger_axis_mode_is_rejected():
    config = GamepadConfig(trigger_axis_mode='automatic')
    with pytest.raises(ValueError, match='trigger_axis_mode'):
        map_joy_axes([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], config)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('publish_rate_hz', math.nan),
        ('publish_rate_hz', math.inf),
        ('joy_timeout_sec', math.nan),
        ('graph_check_period_sec', math.inf),
        ('neutral_trigger_threshold', math.nan),
        ('neutral_trigger_threshold', 1.0),
    ],
)
def test_non_finite_or_unsafe_runtime_parameter_is_rejected(field, value):
    parameters = {
        'publish_rate_hz': 20.0,
        'joy_timeout_sec': 0.25,
        'graph_check_period_sec': 0.5,
        'neutral_trigger_threshold': 0.05,
        'stop_publish_count': 5,
    }
    parameters[field] = value
    with pytest.raises(ValueError):
        _validate_runtime_parameters(**parameters)


def test_neutral_arming_gate_requires_neutral_again_after_disarm():
    gate = NeutralArmingGate(threshold=0.05)

    assert not gate.observe(0.0, 0.5)
    assert gate.observe(0.05, 0.04)

    gate.disarm()
    assert not gate.observe(0.06, 0.0)
    assert gate.observe(0.0, 0.0)


def test_input_freshness_timeout():
    assert is_input_fresh(10.2, 10.0, 0.25)
    assert is_input_fresh(10.25, 10.0, 0.25)
    assert not is_input_fresh(10.251, 10.0, 0.25)
    assert not is_input_fresh(10.0, None, 0.25)
    assert not is_input_fresh(9.9, 10.0, 0.25)


def test_recording_gate_waits_for_positive_published_speed():
    gate = RecordingGate()

    assert gate.observe_buttons(
        a_pressed=True,
        b_pressed=False,
    ) == RecordingAction.WAITING_STARTED
    assert gate.state == RecordingState.WAITING_FORWARD
    assert gate.observe_published_speed(0.0) == RecordingAction.NONE
    assert gate.observe_published_speed(-1.0) == RecordingAction.NONE
    assert gate.observe_published_speed(0.1) == (
        RecordingAction.START_RECORDING
    )
    assert gate.state == RecordingState.RECORDING


def test_releasing_a_cancels_waiting_but_not_active_recording():
    gate = RecordingGate()
    gate.observe_buttons(a_pressed=True, b_pressed=False)

    assert gate.observe_buttons(
        a_pressed=False,
        b_pressed=False,
    ) == RecordingAction.WAITING_CANCELLED
    assert gate.state == RecordingState.IDLE

    gate.observe_buttons(a_pressed=True, b_pressed=False)
    gate.observe_published_speed(1.0)
    assert gate.observe_buttons(
        a_pressed=False,
        b_pressed=False,
    ) == RecordingAction.NONE
    assert gate.state == RecordingState.RECORDING


def test_b_finishes_normally_and_requires_a_release_before_restart():
    gate = RecordingGate()
    gate.observe_buttons(a_pressed=True, b_pressed=False)
    gate.observe_published_speed(1.0)

    assert gate.observe_buttons(
        a_pressed=True,
        b_pressed=True,
    ) == RecordingAction.FINISH_NORMAL
    gate.finish_completed()
    assert gate.state == RecordingState.IDLE

    gate.observe_buttons(a_pressed=True, b_pressed=False)
    assert gate.state == RecordingState.IDLE
    gate.observe_buttons(a_pressed=False, b_pressed=False)
    assert gate.observe_buttons(
        a_pressed=True,
        b_pressed=False,
    ) == RecordingAction.WAITING_STARTED


@pytest.mark.parametrize('speed', [0.0, -0.1])
def test_nonpositive_speed_requests_emergency_recording_finish(speed):
    gate = RecordingGate()
    gate.observe_buttons(a_pressed=True, b_pressed=False)
    gate.observe_published_speed(1.0)

    assert gate.observe_published_speed(speed) == (
        RecordingAction.FINISH_EMERGENCY
    )
    assert gate.state == RecordingState.FINISHING
    assert speed <= 0.0


def test_pending_writer_finish_is_retried_until_accepted():
    class RetryWriter:
        failure = None

        def __init__(self):
            self.calls = []

        def finish(self, token, reason, **kwargs):
            self.calls.append((token, reason, kwargs))
            return len(self.calls) >= 2

    pending = _PendingRecordingFinish(
        token=7,
        reason='b_button',
        complete=True,
        extra_metadata={'emergency_discard_count': 0},
        final_samples=(),
    )
    node = SimpleNamespace(
        writer=RetryWriter(),
        _pending_recording_finish=pending,
    )

    GamepadTeleopNode._retry_pending_recording_finish(node)
    assert node._pending_recording_finish is pending
    GamepadTeleopNode._retry_pending_recording_finish(node)
    assert node._pending_recording_finish is None
    assert len(node.writer.calls) == 2


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('camera_topic', ''),
        ('recording_root_dir', ''),
        ('record_start_button', -1),
        ('record_stop_button', -1),
        ('emergency_discard_frames', -1),
        ('recording_png_compression', 10),
        ('recording_queue_size', 0),
        ('recording_min_free_space_mb', -1),
    ],
)
def test_invalid_recording_parameter_is_rejected(field, value):
    parameters = {
        'camera_topic': '/image_raw',
        'recording_root_dir': '/tmp/xycar_data_test',
        'record_start_button': 0,
        'record_stop_button': 1,
        'emergency_discard_frames': 15,
        'recording_png_compression': 3,
        'recording_queue_size': 128,
        'recording_min_free_space_mb': 0,
    }
    parameters[field] = value
    with pytest.raises(ValueError):
        _validate_recording_parameters(**parameters)


def test_recording_buttons_must_be_distinct():
    with pytest.raises(ValueError, match='must be different'):
        _validate_recording_parameters(
            camera_topic='/image_raw',
            recording_root_dir='/tmp/xycar_data_test',
            record_start_button=0,
            record_stop_button=0,
            emergency_discard_frames=15,
            recording_png_compression=3,
            recording_queue_size=128,
            recording_min_free_space_mb=0,
        )
