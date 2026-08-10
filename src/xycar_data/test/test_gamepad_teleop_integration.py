# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import csv
import time

import pytest
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float32MultiArray

from xycar_data.gamepad_teleop import (
    GamepadTeleopNode,
    RecordingState,
)


JOY_TOPIC = '/xycar_data_test/gamepad/joy'
MOTOR_TOPIC = '/xycar_data_test/gamepad/motor'
CAMERA_TOPIC = '/xycar_data_test/gamepad/camera'


def _joy_message(
    steering: float = 0.0,
    lt: float = 0.0,
    rt: float = 0.0,
    a: bool = False,
    b: bool = False,
) -> Joy:
    message = Joy()
    # Callers provide 0..1 depth; the vehicle controller reports +1..-1.
    lt_axis = 1.0 - (2.0 * lt)
    rt_axis = 1.0 - (2.0 * rt)
    message.axes = [steering, 0.0, 0.0, 0.0, lt_axis, rt_axis]
    message.buttons = [int(a), int(b)]
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
        Parameter('publish_rate_hz', value=20.0),
        Parameter('joy_timeout_sec', value=0.25),
        Parameter('graph_check_period_sec', value=0.05),
        Parameter('neutral_trigger_threshold', value=0.05),
        Parameter('stop_publish_count', value=1),
        Parameter('recording_root_dir', value=str(tmp_path / 'teleop')),
        Parameter('emergency_discard_frames', value=15),
        Parameter('recording_png_compression', value=0),
        Parameter('recording_queue_size', value=64),
        Parameter('recording_min_free_space_mb', value=0),
    ]
    teleop = GamepadTeleopNode(parameter_overrides=overrides)
    peer = Node('gamepad_teleop_test_peer')
    received_commands = []

    joy_publisher = peer.create_publisher(Joy, JOY_TOPIC, 10)
    camera_publisher = peer.create_publisher(Image, CAMERA_TOPIC, 10)

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
        'motor_subscription': motor_subscription,
        'on_motor': on_motor,
        'received_commands': received_commands,
        'recording_root': tmp_path / 'teleop',
    }
    assert teleop.motor_topic == MOTOR_TOPIC
    assert teleop.config.trigger_axis_mode == 'signed'
    yield harness

    teleop.shutdown()
    executor.remove_node(teleop)
    executor.remove_node(peer)
    teleop.destroy_node()
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
        lambda: bool(teleop._competitors) and bool(commands),
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
    assert {row['angle'] for row in rows} == {'-25.000000'}
    assert {row['speed'] for row in rows} == {'7.000000'}
    assert {row['input_key'] for row in rows} == {'gamepad'}
    metadata = yaml.safe_load(
        (sessions[0] / 'metadata.yaml').read_text(encoding='utf-8')
    )
    assert metadata['control_mode'] == 'gamepad'
    assert metadata['stop_reason'] == 'b_button'
    assert metadata['emergency_discard_count'] == 0


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
    _spin_for(harness, 0.15, _joy_message(rt=0.5, b=True))
    assert _spin_until(
        harness,
        lambda: teleop._recording_gate.state == RecordingState.IDLE,
        2.0,
        _joy_message(rt=0.5),
    )
    assert len(list(harness['recording_root'].glob('*_session*'))) == 1
