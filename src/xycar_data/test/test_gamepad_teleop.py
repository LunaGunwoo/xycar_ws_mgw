# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import ament_index_python.packages
from launch import LaunchDescription, LaunchService
from launch.actions import SetLaunchConfiguration
from launch_ros.actions import Node
import numpy as np
import pytest
import yaml

from xycar_data.gamepad_teleop import (
    DriveCommand,
    GamepadConfig,
    GamepadTeleopNode,
    InvalidJoyInput,
    MissionSequenceRecordingGate,
    NeutralArmingGate,
    RecordingAction,
    RecordingGate,
    RecordingState,
    _PendingRecordingFinish,
    _is_paired_unnamed_relay,
    _validate_collection_profile,
    _validate_lidar_parameters,
    _validate_recording_parameters,
    _validate_runtime_parameters,
    is_input_fresh,
    map_joy_axes,
    matching_lidar_snapshot,
)
from xycar_data.session_writer import LidarSnapshot
from xycar_data.traffic_signal_collector import (
    SignalSelection,
    select_signal_class,
)


SIGNAL_BUTTONS = {
    'red': 2,
    'yellow': 3,
    'straight_green': 1,
    'left_green': 0,
}


def _endpoint(participant: bytes, entity: bytes, *, named: bool = False):
    return SimpleNamespace(
        endpoint_gid=list(participant + entity),
        node_name='publisher' if named else '_NODE_NAME_UNKNOWN_',
        node_namespace='/' if named else '_NODE_NAMESPACE_UNKNOWN_',
    )


def _lidar_snapshot(received_monotonic: float) -> LidarSnapshot:
    return LidarSnapshot(
        sequence=1,
        ranges=np.asarray([1.0, 2.0], dtype=np.float32),
        intensities=np.asarray([3.0, 4.0], dtype=np.float32),
        angle_min=-1.0,
        angle_max=1.0,
        angle_increment=0.1,
        time_increment=0.001,
        scan_time=0.1,
        range_min=0.1,
        range_max=16.0,
        frame_id='laser_frame',
        stamp_sec=10,
        stamp_nanosec=20,
        received_monotonic=received_monotonic,
        received_wall_time_ns=30,
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


def test_traffic_signal_speed_profile_is_symmetric_and_proportional():
    config = GamepadConfig(
        max_forward_speed=5.0,
        max_reverse_speed=5.0,
    )

    forward = map_joy_axes(
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
        config,
    )
    reverse = map_joy_axes(
        [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],
        config,
    )
    cancelled = map_joy_axes(
        [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
        config,
    )

    assert forward.speed == pytest.approx(2.5)
    assert reverse.speed == pytest.approx(-2.5)
    assert cancelled.speed == pytest.approx(0.0)


@pytest.mark.parametrize(
    ('buttons', 'expected'),
    [
        ([0, 0, 1, 0], SignalSelection('red')),
        ([0, 0, 0, 1], SignalSelection('yellow')),
        ([0, 1, 0, 0], SignalSelection('straight_green')),
        ([1, 0, 0, 0], SignalSelection('left_green')),
        ([0, 0, 0, 0], SignalSelection(None)),
        ([0, 0, 1, 1], SignalSelection(None, ambiguous=True)),
    ],
)
def test_signal_class_selection_requires_exactly_one_button(
    buttons,
    expected,
):
    assert select_signal_class(buttons, SIGNAL_BUTTONS) == expected


def test_signal_class_selection_rejects_short_or_duplicate_mapping():
    with pytest.raises(InvalidJoyInput, match='button index 3'):
        select_signal_class([0, 0], SIGNAL_BUTTONS)
    with pytest.raises(ValueError, match='must be distinct'):
        select_signal_class(
            [0, 0, 0, 0],
            {**SIGNAL_BUTTONS, 'yellow': 2},
        )


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


def test_angle_above_normalized_range_is_rejected():
    config = GamepadConfig(max_angle=100.1)
    with pytest.raises(ValueError, match='max_angle must be in'):
        map_joy_axes([0.0] * 6, config)


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


def test_lidar_matching_accepts_only_recent_preceding_scans():
    lidar = _lidar_snapshot(10.0)

    matched, skew = matching_lidar_snapshot(
        10.19,
        lidar,
        lidar_timeout_sec=0.30,
        max_lidar_skew_sec=0.20,
    )
    assert matched is lidar
    assert skew == pytest.approx(0.19)

    for camera_time in (9.99, 10.21, 10.31):
        assert matching_lidar_snapshot(
            camera_time,
            lidar,
            lidar_timeout_sec=0.30,
            max_lidar_skew_sec=0.20,
        ) == (None, None)
    assert matching_lidar_snapshot(
        10.0,
        None,
        lidar_timeout_sec=0.30,
        max_lidar_skew_sec=0.20,
    ) == (None, None)


@pytest.mark.parametrize(
    ('topic', 'timeout_sec', 'max_skew_sec'),
    [
        ('', 0.30, 0.20),
        ('/scan', 0.0, 0.20),
        ('/scan', math.nan, 0.20),
        ('/scan', 0.30, 0.0),
        ('/scan', 0.30, math.inf),
        ('/scan', 0.20, 0.30),
    ],
)
def test_invalid_lidar_parameters_are_rejected(
    topic,
    timeout_sec,
    max_skew_sec,
):
    with pytest.raises(ValueError):
        _validate_lidar_parameters(topic, timeout_sec, max_skew_sec)


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


def test_mission_sequence_gate_starts_immediately_and_ignores_zero_speed():
    gate = MissionSequenceRecordingGate()

    assert gate.observe_buttons(
        a_pressed=True,
        b_pressed=False,
    ) == RecordingAction.START_RECORDING
    assert gate.state == RecordingState.RECORDING
    assert gate.observe_published_speed(0.0) == RecordingAction.NONE
    assert gate.observe_published_speed(-1.0) == RecordingAction.NONE
    assert gate.state == RecordingState.RECORDING
    assert gate.observe_buttons(
        a_pressed=False,
        b_pressed=True,
    ) == RecordingAction.FINISH_NORMAL


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
        ('recording_image_format', 'bmp'),
        ('recording_jpeg_quality', 0),
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
        'recording_image_format': 'jpeg',
        'recording_jpeg_quality': 95,
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
            recording_image_format='jpeg',
            recording_jpeg_quality=95,
            recording_png_compression=3,
            recording_queue_size=128,
            recording_min_free_space_mb=0,
        )


def test_normalized_collection_templates_are_versioned_and_strict(tmp_path):
    package_root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            package_root
            / 'config'
            / 'gamepad_stateless_manual_normalized_v1.yaml'
        ).read_text(encoding='utf-8')
    )['gamepad_teleop']['ros__parameters']
    launch_text = (
        package_root / 'launch' / 'gamepad_teleop.launch.py'
    ).read_text(encoding='utf-8')

    assert config['max_forward_speed'] == 25.0
    assert config['max_reverse_speed'] == 10.0
    assert config['lidar_topic'] == '/scan'
    assert config['lidar_timeout_sec'] == pytest.approx(0.30)
    assert config['max_lidar_skew_sec'] == pytest.approx(0.20)
    assert config['recording_root_dir'].endswith('/stateless_manual')
    assert config['recording_image_format'] == 'jpeg'
    assert config['recording_jpeg_quality'] == 95
    assert config['steering_contract'] == 'normalized_percent_v1'
    assert "'collection_profile_path': ParameterValue(" in launch_text
    assert "default_value='/gamepad_teleop/joy'" in launch_text
    assert "remappings=[('joy', joy_topic)]" in launch_text
    assert "'joy_topic': ParameterValue(" in launch_text
    assert "'use_lidar'" in launch_text
    assert "default_value='true'" in launch_text
    assert "'xycar_lidar.launch.py'" in launch_text
    _validate_collection_profile(
        str(
            package_root
            / 'config'
            / 'gamepad_stateless_manual_normalized_v1.yaml'
        )
    )
    legacy = package_root / 'config' / 'gamepad_stateless_manual.yaml'
    with pytest.raises(ValueError, match='steering_contract'):
        _validate_collection_profile(str(legacy))
    invalid = tmp_path / 'invalid.yaml'
    invalid.write_text(
        'gamepad_teleop:\n  ros__parameters:\n'
        '    steering_contract: legacy\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='steering_contract'):
        _validate_collection_profile(str(invalid))


def test_traffic_signal_profile_and_launch_preserve_flat_class_contract():
    package_root = Path(__file__).parents[1]
    profile = (
        package_root
        / 'config'
        / 'traffic_signal_collection_normalized_v1.yaml'
    )
    config = yaml.safe_load(profile.read_text(encoding='utf-8'))[
        'traffic_signal_collector'
    ]['ros__parameters']
    launch_text = (
        package_root / 'launch' / 'traffic_signal_collection.launch.py'
    ).read_text(encoding='utf-8')

    assert config['max_forward_speed'] == 5.0
    assert config['max_reverse_speed'] == 5.0
    assert config['red_button'] == 2
    assert config['yellow_button'] == 3
    assert config['straight_green_button'] == 1
    assert config['left_green_button'] == 0
    assert config['recording_root_dir'].endswith('/traffic_signal_images')
    assert config['recording_jpeg_quality'] == 95
    assert config['steering_contract'] == 'normalized_percent_v1'
    assert config['preview_enabled'] is False
    assert "default_value='/traffic_signal_collector/joy'" in launch_text
    assert "default_value='false'" in launch_text
    assert "('image', '/traffic_signal_collector/preview')" in launch_text
    assert "'collection_profile_path': ParameterValue(" in launch_text
    _validate_collection_profile(
        str(profile),
        node_name='traffic_signal_collector',
    )


def test_lidar_include_keeps_teleop_params_file_scoped(monkeypatch):
    package_root = Path(__file__).parents[1]
    source_root = package_root.parent
    profile = (
        package_root
        / 'config'
        / 'gamepad_stateless_manual_normalized_v1.yaml'
    )
    lidar_params = (
        source_root
        / 'xycar_device'
        / 'xycar_lidar'
        / 'params'
        / 'ydlidar.yaml'
    )
    package_shares = {
        'xycar_data': package_root,
        'xycar_cam': source_root / 'xycar_device' / 'xycar_cam',
        'xycar_lidar': source_root / 'xycar_device' / 'xycar_lidar',
    }

    monkeypatch.setattr(
        ament_index_python.packages,
        'get_package_share_directory',
        lambda name: str(package_shares[name]),
    )
    launch_path = package_root / 'launch' / 'gamepad_teleop.launch.py'
    spec = importlib.util.spec_from_file_location(
        'gamepad_teleop_launch_scope_test',
        launch_path,
    )
    assert spec is not None and spec.loader is not None
    launch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_module)
    monkeypatch.setattr(
        launch_module,
        'get_package_share_directory',
        lambda name: str(package_shares[name]),
    )

    expanded_parameter_files = {}

    def inspect_node_without_starting_process(node, context):
        node._perform_substitutions(context)
        expanded_parameter_files[node.node_executable] = [
            path
            for path, is_file in node._Node__expanded_parameter_arguments
            if is_file
        ]
        return None

    monkeypatch.setattr(Node, 'execute', inspect_node_without_starting_process)
    root_description = LaunchDescription(
        [
            SetLaunchConfiguration('params_file', str(profile)),
            SetLaunchConfiguration('use_camera', 'false'),
            SetLaunchConfiguration('use_lidar', 'true'),
            launch_module.generate_launch_description(),
        ]
    )
    launch_service = LaunchService()
    launch_service.include_launch_description(root_description)

    assert launch_service.run() == 0
    assert expanded_parameter_files['xycar_lidar_node'][0] == str(
        lidar_params
    )
    for executable in ('game_controller_node', 'gamepad_teleop'):
        assert expanded_parameter_files[executable][0] == str(profile)
        assert str(lidar_params) not in expanded_parameter_files[executable]
