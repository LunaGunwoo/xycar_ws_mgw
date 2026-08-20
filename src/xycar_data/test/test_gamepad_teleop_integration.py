# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import csv
import time

import cv2
import numpy as np
import pytest
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Joy, LaserScan
from std_msgs.msg import Float32MultiArray

from xycar_data.gamepad_teleop import (
    GamepadTeleopNode,
    RecordingState,
)
from xycar_data.traffic_signal_collector import (
    SIGNAL_CLASSES,
    TrafficSignalCollectorNode,
)


JOY_TOPIC = '/xycar_data_test/gamepad/joy'
MOTOR_TOPIC = '/xycar_data_test/gamepad/motor'
CAMERA_TOPIC = '/xycar_data_test/gamepad/camera'
LIDAR_TOPIC = '/xycar_data_test/gamepad/lidar'
SIGNAL_JOY_TOPIC = '/xycar_data_test/signal/joy'
SIGNAL_MOTOR_TOPIC = '/xycar_data_test/signal/motor'
SIGNAL_CAMERA_TOPIC = '/xycar_data_test/signal/camera'
SIGNAL_PREVIEW_TOPIC = '/xycar_data_test/signal/preview'


def _joy_message(
    steering: float = 0.0,
    lt: float = 0.0,
    rt: float = 0.0,
    a: bool = False,
    b: bool = False,
    x: bool = False,
    y: bool = False,
) -> Joy:
    message = Joy()
    # Callers provide 0..1 depth; the vehicle controller reports 0..-1.
    lt_axis = -lt
    rt_axis = -rt
    message.axes = [steering, 0.0, 0.0, 0.0, lt_axis, rt_axis]
    message.buttons = [int(a), int(b), int(x), int(y)]
    return message


def _image_message(sequence: int) -> Image:
    message = Image()
    message.header.stamp.sec = sequence
    message.header.stamp.nanosec = sequence * 1000
    message.height = 4
    message.width = 6
    message.encoding = 'bgr8'
    message.is_bigendian = False
    message.step = message.width * 3
    message.data = bytes([sequence % 256] * (message.step * message.height))
    return message


def _lidar_message(sequence: int) -> LaserScan:
    message = LaserScan()
    message.header.stamp.sec = sequence
    message.header.stamp.nanosec = sequence * 1000
    message.header.frame_id = 'laser_frame'
    message.angle_min = -1.0
    message.angle_max = 1.0
    message.angle_increment = 1.0
    message.time_increment = 0.01
    message.scan_time = 0.10
    message.range_min = 0.10
    message.range_max = 16.0
    message.ranges = [1.0, 2.0, 3.0]
    message.intensities = [10.0, 20.0, 30.0]
    return message


def _spin_for(harness, duration_sec, joy_message=None):
    deadline = time.monotonic() + duration_sec
    next_joy_publish = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if joy_message is not None and now >= next_joy_publish:
            harness['joy_publisher'].publish(joy_message)
            next_joy_publish = now + 0.02
        harness['executor'].spin_once(timeout_sec=0.01)


def _spin_until(
    harness,
    predicate,
    timeout_sec,
    joy_message=None,
):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        _spin_for(harness, 0.05, joy_message)
        if predicate():
            return True
    return predicate()


def _publish_camera_frames(harness, count, joy_message):
    start_sequence = harness['teleop']._camera_sequence + 1
    for sequence in range(start_sequence, start_sequence + count):
        harness['camera_publisher'].publish(_image_message(sequence))
        _spin_for(harness, 0.03, joy_message)


@pytest.fixture
def ros_harness(monkeypatch, tmp_path):
    # Keep this test graph away from the vehicle domain and motor topic.
    monkeypatch.setenv('ROS_DOMAIN_ID', '222')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    rclpy.init(args=[])

    overrides = [
        Parameter('joy_topic', value=JOY_TOPIC),
        Parameter('motor_topic', value=MOTOR_TOPIC),
        Parameter('camera_topic', value=CAMERA_TOPIC),
        Parameter('lidar_topic', value=LIDAR_TOPIC),
        Parameter('lidar_timeout_sec', value=0.30),
        Parameter('max_lidar_skew_sec', value=0.20),
        Parameter('steering_contract', value='normalized_percent_v1'),
        Parameter('publish_rate_hz', value=20.0),
        Parameter('joy_timeout_sec', value=0.25),
        Parameter('graph_check_period_sec', value=0.05),
        Parameter('neutral_trigger_threshold', value=0.05),
        Parameter('stop_publish_count', value=1),
        Parameter('recording_root_dir', value=str(tmp_path / 'teleop')),
        Parameter('emergency_discard_frames', value=15),
        Parameter('recording_image_format', value='jpeg'),
        Parameter('recording_jpeg_quality', value=95),
        Parameter('recording_png_compression', value=0),
        Parameter('recording_queue_size', value=64),
        Parameter('recording_min_free_space_mb', value=0),
    ]
    profile = tmp_path / 'gamepad_stateless_manual_normalized_v1.yaml'
    profile.write_text(
        'gamepad_teleop:\n'
        '  ros__parameters:\n'
        '    steering_contract: normalized_percent_v1\n',
        encoding='utf-8',
    )
    overrides.append(
        Parameter('collection_profile_path', value=str(profile))
    )
    teleop = GamepadTeleopNode(parameter_overrides=overrides)
    peer = Node('gamepad_teleop_test_peer')
    received_commands = []

    joy_publisher = peer.create_publisher(Joy, JOY_TOPIC, 10)
    camera_publisher = peer.create_publisher(Image, CAMERA_TOPIC, 10)
    lidar_publisher = peer.create_publisher(
        LaserScan,
        LIDAR_TOPIC,
        qos_profile_sensor_data,
    )

    def on_motor(message):
        received_commands.append(tuple(message.data))

    motor_subscription = peer.create_subscription(
        Float32MultiArray,
        MOTOR_TOPIC,
        on_motor,
        10,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(teleop)
    executor.add_node(peer)

    harness = {
        'executor': executor,
        'teleop': teleop,
        'peer': peer,
        'joy_publisher': joy_publisher,
        'camera_publisher': camera_publisher,
        'lidar_publisher': lidar_publisher,
        'motor_subscription': motor_subscription,
        'on_motor': on_motor,
        'received_commands': received_commands,
        'recording_root': tmp_path / 'teleop',
    }
    assert teleop.motor_topic == MOTOR_TOPIC
    assert teleop.lidar_topic == LIDAR_TOPIC
    assert teleop.config.trigger_axis_mode == 'negative'
    yield harness

    teleop.shutdown()
    executor.remove_node(teleop)
    executor.remove_node(peer)
    teleop.destroy_node()
    peer.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


@pytest.fixture
def signal_harness(monkeypatch, tmp_path):
    # Keep this test graph away from the vehicle domain and motor topic.
    monkeypatch.setenv('ROS_DOMAIN_ID', '223')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    rclpy.init(args=[])

    root = tmp_path / 'traffic_signal_images'
    profile = tmp_path / 'traffic_signal_collection_normalized_v1.yaml'
    profile.write_text(
        'traffic_signal_collector:\n'
        '  ros__parameters:\n'
        '    steering_contract: normalized_percent_v1\n',
        encoding='utf-8',
    )
    overrides = [
        Parameter('joy_topic', value=SIGNAL_JOY_TOPIC),
        Parameter('motor_topic', value=SIGNAL_MOTOR_TOPIC),
        Parameter('camera_topic', value=SIGNAL_CAMERA_TOPIC),
        Parameter('preview_topic', value=SIGNAL_PREVIEW_TOPIC),
        Parameter('collection_profile_path', value=str(profile)),
        Parameter('steering_contract', value='normalized_percent_v1'),
        Parameter('max_forward_speed', value=5.0),
        Parameter('max_reverse_speed', value=5.0),
        Parameter('publish_rate_hz', value=20.0),
        Parameter('joy_timeout_sec', value=0.25),
        Parameter('camera_timeout_sec', value=0.20),
        Parameter('graph_check_period_sec', value=0.05),
        Parameter('neutral_trigger_threshold', value=0.05),
        Parameter('stop_publish_count', value=1),
        Parameter('recording_root_dir', value=str(root)),
        Parameter('recording_jpeg_quality', value=95),
        Parameter('recording_queue_size', value=64),
        Parameter('recording_min_free_space_mb', value=0),
        Parameter('preview_enabled', value=True),
    ]
    collector = TrafficSignalCollectorNode(parameter_overrides=overrides)

    def fail_outgoing_cv_bridge(*_args, **_kwargs):
        raise KeyError(16)

    monkeypatch.setattr(
        collector.bridge,
        'cv2_to_imgmsg',
        fail_outgoing_cv_bridge,
    )
    peer = Node('traffic_signal_collector_test_peer')
    commands = []
    previews = []

    joy_publisher = peer.create_publisher(Joy, SIGNAL_JOY_TOPIC, 10)
    camera_publisher = peer.create_publisher(
        Image,
        SIGNAL_CAMERA_TOPIC,
        qos_profile_sensor_data,
    )
    motor_subscription = peer.create_subscription(
        Float32MultiArray,
        SIGNAL_MOTOR_TOPIC,
        lambda message: commands.append(tuple(message.data)),
        10,
    )
    preview_subscription = peer.create_subscription(
        Image,
        SIGNAL_PREVIEW_TOPIC,
        lambda message: previews.append(message),
        10,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(collector)
    executor.add_node(peer)
    harness = {
        'executor': executor,
        'teleop': collector,
        'peer': peer,
        'joy_publisher': joy_publisher,
        'camera_publisher': camera_publisher,
        'motor_subscription': motor_subscription,
        'preview_subscription': preview_subscription,
        'received_commands': commands,
        'previews': previews,
        'recording_root': root,
    }
    yield harness

    collector.shutdown()
    executor.remove_node(collector)
    executor.remove_node(peer)
    collector.destroy_node()
    peer.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


def test_neutral_rearming_repeat_and_graph_fail_safes(ros_harness):
    harness = ros_harness
    teleop = harness['teleop']
    commands = harness['received_commands']
    held_forward = _joy_message(rt=0.5)
    neutral = _joy_message()

    # A held trigger at startup must not arm motion.
    _spin_for(harness, 0.30, held_forward)
    assert teleop._has_motor_subscriber
    assert commands
    assert all(command[1] == 0.0 for command in commands)
    assert not teleop._arming_gate.armed

    # Neutral arms the node. One fresh input is then repeated by the control
    # timer until its freshness timeout.
    _spin_for(harness, 0.15, neutral)
    assert teleop._arming_gate.armed
    commands.clear()
    harness['joy_publisher'].publish(held_forward)
    _spin_for(harness, 0.15)
    forward_commands = [
        command for command in commands if command[1] == pytest.approx(3.5)
    ]
    assert len(forward_commands) >= 2

    # Stale Joy input stops and disarms. A still-held trigger cannot resume.
    commands.clear()
    _spin_for(harness, 0.20)
    assert commands[-1] == pytest.approx((0.0, 0.0))
    assert not teleop._arming_gate.armed
    commands.clear()
    _spin_for(harness, 0.15, held_forward)
    assert commands
    assert all(command[1] == 0.0 for command in commands)

    # Neutral restores motion after the stale-input stop.
    _spin_for(harness, 0.15, neutral)
    assert teleop._arming_gate.armed
    commands.clear()
    _spin_for(harness, 0.10, held_forward)
    assert any(command[1] == pytest.approx(3.5) for command in commands)

    # Missing A/B entries disable recording controls but never block driving.
    short_buttons = _joy_message(rt=0.5)
    short_buttons.buttons = []
    commands.clear()
    _spin_for(harness, 0.10, short_buttons)
    assert any(command[1] == pytest.approx(3.5) for command in commands)

    # A competing publisher stops and disarms the teleop.
    competitor = harness['peer'].create_publisher(
        Float32MultiArray,
        MOTOR_TOPIC,
        10,
    )
    commands.clear()
    assert _spin_until(
        harness,
        lambda: (
            bool(teleop._competitors)
            and bool(commands)
            and commands[-1][1] == 0.0
        ),
        2.0,
        held_forward,
    )
    assert commands[-1] == pytest.approx((0.0, 0.0))
    assert not teleop._arming_gate.armed
    harness['peer'].destroy_publisher(competitor)
    assert _spin_until(
        harness,
        lambda: not teleop._competitors,
        2.0,
        held_forward,
    )

    # Removing the only motor subscriber also disarms. Recreating it while RT
    # remains held must not resume until another neutral input is observed.
    harness['peer'].destroy_subscription(harness['motor_subscription'])
    harness['motor_subscription'] = None
    assert _spin_until(
        harness,
        lambda: not teleop._has_motor_subscriber,
        2.0,
        held_forward,
    )
    assert not teleop._arming_gate.armed

    harness['motor_subscription'] = harness['peer'].create_subscription(
        Float32MultiArray,
        MOTOR_TOPIC,
        harness['on_motor'],
        10,
    )
    commands.clear()
    assert _spin_until(
        harness,
        lambda: teleop._has_motor_subscriber and bool(commands),
        2.0,
        held_forward,
    )
    assert commands
    assert all(command[1] == 0.0 for command in commands)
    assert not teleop._arming_gate.armed

    _spin_for(harness, 0.15, neutral)
    assert teleop._arming_gate.armed
    commands.clear()
    _spin_for(harness, 0.10, held_forward)
    assert any(command[1] == pytest.approx(3.5) for command in commands)


def test_a_waits_for_forward_and_b_saves_all_frames_without_stopping(
    ros_harness,
):
    harness = ros_harness
    teleop = harness['teleop']
    commands = harness['received_commands']
    neutral = _joy_message()
    waiting = _joy_message(a=True)
    forward = _joy_message(steering=0.25, rt=1.0, a=True)

    _spin_for(harness, 0.15, neutral)
    assert teleop._arming_gate.armed
    _spin_for(harness, 0.10, waiting)
    assert teleop._recording_gate.state == RecordingState.WAITING_FORWARD
    assert teleop._session_token is None

    _spin_for(harness, 0.15, forward)
    assert teleop._recording_gate.state == RecordingState.RECORDING
    assert teleop._session_token is not None

    # Missing or malformed camera data does not stop driving or the session.
    _spin_for(harness, 0.10, forward)
    malformed = Image()
    malformed.encoding = 'unsupported'
    harness['camera_publisher'].publish(malformed)
    _spin_for(harness, 0.05, forward)
    assert teleop._recording_gate.state == RecordingState.RECORDING

    _publish_camera_frames(harness, 10, forward)
    _spin_for(harness, 0.30, forward)
    assert teleop._recording_gate.state == RecordingState.RECORDING
    _publish_camera_frames(harness, 10, forward)
    assert teleop._camera_sequence == 20

    commands.clear()
    finish = _joy_message(steering=0.25, rt=1.0, b=True)
    _spin_for(harness, 0.15, finish)
    assert _spin_until(
        harness,
        lambda: teleop._recording_gate.state == RecordingState.IDLE,
        2.0,
        _joy_message(steering=0.25, rt=1.0),
    )

    assert any(command[1] == pytest.approx(7.0) for command in commands)
    sessions = list(harness['recording_root'].glob('*_session*'))
    assert len(sessions) == 1
    with (sessions[0] / 'samples.csv').open(
        encoding='utf-8',
        newline='',
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    assert all(row['image'].endswith('.jpg') for row in rows)
    assert all((sessions[0] / row['image']).is_file() for row in rows)
    assert {row['angle'] for row in rows} == {'-25.000000'}
    assert {row['speed'] for row in rows} == {'7.000000'}
    assert {row['input_key'] for row in rows} == {'gamepad'}
    metadata = yaml.safe_load(
        (sessions[0] / 'metadata.yaml').read_text(encoding='utf-8')
    )
    assert metadata['control_mode'] == 'gamepad'
    assert metadata['stop_reason'] == 'b_button'
    assert metadata['emergency_discard_count'] == 0
    assert metadata['collection_profile']['path'].endswith(
        'gamepad_stateless_manual_normalized_v1.yaml'
    )
    assert len(metadata['collection_profile']['sha256']) == 64
    assert metadata['recording']['root_dir'] == str(
        harness['recording_root']
    )
    assert metadata['recording']['image_format'] == 'jpeg'
    assert metadata['recording']['jpeg_quality'] == 95
    assert metadata['steering_contract'] == {
        'schema_version': 1,
        'name': 'normalized_percent_v1',
        'command_min': -100.0,
        'command_max': 100.0,
        'driver_min': -40.0,
        'driver_max': 40.0,
        'mapping': 'linear_scale_0.4',
        'motor_topic': MOTOR_TOPIC,
        'driver_topic': '/xycar_motor_safe',
    }


def test_lidar_links_to_camera_samples_and_missing_scans_do_not_stop(
    ros_harness,
):
    harness = ros_harness
    teleop = harness['teleop']
    forward = _joy_message(rt=1.0, a=True)

    _spin_for(harness, 0.15, _joy_message())
    _spin_for(harness, 0.15, forward)
    assert teleop._recording_gate.state == RecordingState.RECORDING

    # Camera remains primary: a frame before any scan is retained as missing.
    harness['camera_publisher'].publish(_image_message(1))
    _spin_for(harness, 0.03, forward)

    # One 10 Hz scan may be linked by multiple faster camera frames.
    harness['lidar_publisher'].publish(_lidar_message(7))
    _spin_for(harness, 0.03, forward)
    for sequence in (2, 3):
        harness['camera_publisher'].publish(_image_message(sequence))
        _spin_for(harness, 0.03, forward)

    # A stale scan produces lidar_valid=false without stopping the command.
    harness['received_commands'].clear()
    _spin_for(harness, 0.23, forward)
    harness['camera_publisher'].publish(_image_message(4))
    _spin_for(harness, 0.05, forward)
    assert any(
        command[1] == pytest.approx(7.0)
        for command in harness['received_commands']
    )

    _spin_for(harness, 0.15, _joy_message(rt=1.0, b=True))
    assert _spin_until(
        harness,
        lambda: teleop._recording_gate.state == RecordingState.IDLE,
        2.0,
        _joy_message(rt=1.0),
    )

    sessions = list(harness['recording_root'].glob('*_session*'))
    assert len(sessions) == 1
    session = sessions[0]
    with (session / 'samples.csv').open(
        encoding='utf-8',
        newline='',
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row['lidar_valid'] for row in rows] == [
        'false',
        'true',
        'true',
        'false',
    ]
    assert rows[1]['lidar'] == rows[2]['lidar'] == 'Lidar/000001.npz'
    assert rows[1]['lidar_sequence'] == rows[2]['lidar_sequence'] == '1'

    lidar_path = session / rows[1]['lidar']
    assert lidar_path.is_file()
    with np.load(lidar_path) as lidar:
        np.testing.assert_allclose(lidar['ranges'], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            lidar['intensities'],
            [10.0, 20.0, 30.0],
        )
        assert lidar['frame_id'].item() == 'laser_frame'
        assert lidar['stamp_sec'].item() == 7

    metadata = yaml.safe_load(
        (session / 'metadata.yaml').read_text(encoding='utf-8')
    )
    assert metadata['lidar_is_optional'] is True
    assert metadata['topics']['lidar_topic'] == LIDAR_TOPIC
    assert metadata['sensors'] == {
        'lidar_timeout_sec': pytest.approx(0.30),
        'max_lidar_skew_sec': pytest.approx(0.20),
    }
    assert metadata['sample_count'] == 4
    assert metadata['lidar_linked_count'] == 2
    assert metadata['lidar_missing_count'] == 2


def test_nonpositive_speed_discards_fifteen_and_keeps_driving(
    ros_harness,
):
    harness = ros_harness
    teleop = harness['teleop']
    commands = harness['received_commands']
    neutral = _joy_message()
    forward_with_a = _joy_message(rt=0.5, a=True)

    _spin_for(harness, 0.15, neutral)
    _spin_for(harness, 0.15, forward_with_a)
    assert teleop._recording_gate.state == RecordingState.RECORDING
    _publish_camera_frames(harness, 20, forward_with_a)

    commands.clear()
    reverse_with_a = _joy_message(lt=1.0, a=True)
    _spin_for(harness, 0.20, reverse_with_a)
    assert _spin_until(
        harness,
        lambda: teleop._recording_gate.state == RecordingState.IDLE,
        2.0,
        reverse_with_a,
    )

    assert any(command[1] == pytest.approx(-5.0) for command in commands)
    sessions = list(harness['recording_root'].glob('*_session*'))
    assert len(sessions) == 1
    with (sessions[0] / 'samples.csv').open(
        encoding='utf-8',
        newline='',
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    metadata = yaml.safe_load(
        (sessions[0] / 'metadata.yaml').read_text(encoding='utf-8')
    )
    assert metadata['stop_reason'] == 'speed_nonpositive'
    assert metadata['emergency_discard_count'] == 15

    # Holding A across the emergency finish must not create a second session.
    _spin_for(harness, 0.20, forward_with_a)
    assert teleop._recording_gate.state == RecordingState.IDLE
    assert teleop._session_token is None

    # Releasing and pressing A again permits a new recording.
    _spin_for(harness, 0.10, _joy_message(rt=0.5))
    _spin_for(harness, 0.15, forward_with_a)
    assert teleop._recording_gate.state == RecordingState.RECORDING
    _publish_camera_frames(harness, 3, forward_with_a)
    _spin_for(harness, 0.15, _joy_message(rt=0.5, b=True))
    assert _spin_until(
        harness,
        lambda: teleop._recording_gate.state == RecordingState.IDLE,
        2.0,
        _joy_message(rt=0.5),
    )
    sessions = list(harness['recording_root'].glob('*_session*'))
    assert len(sessions) == 2
    sample_counts = []
    for session in sessions:
        with (session / 'samples.csv').open(
            encoding='utf-8',
            newline='',
        ) as stream:
            sample_counts.append(len(list(csv.DictReader(stream))))
    assert sorted(sample_counts) == [3, 5]


def test_signal_buttons_capture_each_frame_without_controlling_motion(
    signal_harness,
):
    harness = signal_harness
    collector = harness['teleop']
    neutral = _joy_message()

    harness['camera_publisher'].publish(_image_message(1))
    _spin_for(harness, 0.15, neutral)
    assert collector._has_motor_subscriber
    assert collector._arming_gate.armed

    # X captures red even while speed remains zero.
    red_hold = _joy_message(x=True)
    _spin_for(harness, 0.05, red_hold)
    for sequence in range(2, 5):
        harness['camera_publisher'].publish(_image_message(sequence))
        _spin_for(harness, 0.03, red_hold)
    assert _spin_until(
        harness,
        lambda: collector.writer.counts['red'] == 3,
        2.0,
        red_hold,
    )

    # The same held label remains independent of proportional RT motion.
    harness['received_commands'].clear()
    moving_red = _joy_message(rt=0.5, x=True)
    harness['camera_publisher'].publish(_image_message(5))
    _spin_for(harness, 0.10, moving_red)
    assert any(
        command[1] == pytest.approx(2.5)
        for command in harness['received_commands']
    )
    assert _spin_until(
        harness,
        lambda: collector.writer.counts['red'] == 4,
        2.0,
        moving_red,
    )

    # Releasing every class button stops capture on the next camera frame.
    before_release = sum(collector.writer.counts.values())
    _spin_for(harness, 0.05, neutral)
    harness['camera_publisher'].publish(_image_message(6))
    _spin_for(harness, 0.05, neutral)
    assert sum(collector.writer.counts.values()) == before_release

    class_messages = (
        ('yellow', _joy_message(y=True)),
        ('straight_green', _joy_message(b=True)),
        ('left_green', _joy_message(a=True)),
    )
    for sequence, (class_name, joy_message) in enumerate(
        class_messages,
        start=7,
    ):
        _spin_for(harness, 0.05, joy_message)
        harness['camera_publisher'].publish(_image_message(sequence))
        _spin_for(harness, 0.05, joy_message)
        assert _spin_until(
            harness,
            lambda name=class_name: collector.writer.counts[name] == 1,
            2.0,
            joy_message,
        )

    # Simultaneous X+Y is visible as ambiguous but never duplicated.
    before_ambiguous = sum(collector.writer.counts.values())
    ambiguous = _joy_message(x=True, y=True)
    _spin_for(harness, 0.05, ambiguous)
    harness['camera_publisher'].publish(_image_message(10))
    _spin_for(harness, 0.05, ambiguous)
    assert collector._selection.ambiguous
    assert sum(collector.writer.counts.values()) == before_ambiguous
    assert harness['previews']
    preview = harness['previews'][-1]
    assert preview.encoding == 'bgr8'
    assert preview.step == preview.width * 3
    assert len(preview.data) == preview.step * preview.height

    assert set(collector.writer.counts) == set(SIGNAL_CLASSES)
    for class_name in SIGNAL_CLASSES:
        images = list((harness['recording_root'] / class_name).iterdir())
        assert images
        assert all(path.suffix == '.jpg' for path in images)
    saved_red = cv2.imread(
        str(next((harness['recording_root'] / 'red').glob('*.jpg')))
    )
    assert saved_red is not None
    # Stored frames stay clean; preview text is rendered only on its topic.
    assert int(saved_red.max()) - int(saved_red.min()) <= 2


def test_signal_collector_stops_for_stale_camera_and_writer_failure(
    signal_harness,
):
    harness = signal_harness
    collector = harness['teleop']
    neutral = _joy_message()
    forward = _joy_message(rt=0.5)

    harness['camera_publisher'].publish(_image_message(1))
    _spin_for(harness, 0.15, neutral)
    harness['received_commands'].clear()
    harness['camera_publisher'].publish(_image_message(2))
    _spin_for(harness, 0.10, forward)
    assert any(
        command[1] == pytest.approx(2.5)
        for command in harness['received_commands']
    )

    harness['received_commands'].clear()
    _spin_for(harness, 0.35, forward)
    assert harness['received_commands'][-1] == pytest.approx((0.0, 0.0))
    assert not collector._arming_gate.armed

    class FailedWriter:
        failure = 'forced writer failure'
        counts = {name: 0 for name in SIGNAL_CLASSES}

        def shutdown(self):
            return True

    assert collector.writer.shutdown()
    collector.writer = FailedWriter()
    harness['received_commands'].clear()
    _spin_for(harness, 0.10, neutral)
    assert collector._capture_disabled
    assert harness['received_commands'][-1] == pytest.approx((0.0, 0.0))
    assert not collector._arming_gate.armed


def test_signal_collector_stops_instead_of_dropping_a_full_queue(
    signal_harness,
):
    harness = signal_harness
    collector = harness['teleop']

    class BackloggedWriter:
        failure = None
        counts = {name: 0 for name in SIGNAL_CLASSES}

        def submit(self, _sample):
            return False

        def shutdown(self):
            return True

    assert collector.writer.shutdown()
    collector.writer = BackloggedWriter()

    neutral = _joy_message()
    red_hold = _joy_message(x=True)
    harness['camera_publisher'].publish(_image_message(1))
    _spin_for(harness, 0.15, neutral)
    _spin_for(harness, 0.05, red_hold)
    harness['received_commands'].clear()
    harness['camera_publisher'].publish(_image_message(2))
    _spin_for(harness, 0.10, red_hold)

    assert collector._capture_disabled
    assert harness['received_commands'][-1] == pytest.approx((0.0, 0.0))
    assert not collector._arming_gate.armed
