# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

from xycar_data.gamepad_teleop import GamepadTeleopNode


JOY_TOPIC = '/xycar_data_test/gamepad/joy'
MOTOR_TOPIC = '/xycar_data_test/gamepad/motor'


def _joy_message(
    steering: float = 0.0,
    lt: float = 0.0,
    rt: float = 0.0,
) -> Joy:
    message = Joy()
    message.axes = [steering, 0.0, 0.0, 0.0, lt, rt]
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


@pytest.fixture
def ros_harness(monkeypatch):
    # Keep this test graph away from the vehicle domain and motor topic.
    monkeypatch.setenv('ROS_DOMAIN_ID', '222')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    rclpy.init(args=[])

    overrides = [
        Parameter('joy_topic', value=JOY_TOPIC),
        Parameter('motor_topic', value=MOTOR_TOPIC),
        Parameter('publish_rate_hz', value=20.0),
        Parameter('joy_timeout_sec', value=0.25),
        Parameter('graph_check_period_sec', value=0.05),
        Parameter('neutral_trigger_threshold', value=0.05),
        Parameter('stop_publish_count', value=1),
    ]
    teleop = GamepadTeleopNode(parameter_overrides=overrides)
    peer = Node('gamepad_teleop_test_peer')
    received_commands = []

    joy_publisher = peer.create_publisher(Joy, JOY_TOPIC, 10)

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
        'motor_subscription': motor_subscription,
        'on_motor': on_motor,
        'received_commands': received_commands,
    }
    assert teleop.motor_topic == MOTOR_TOPIC
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

    # A competing publisher stops and disarms the teleop.
    competitor = harness['peer'].create_publisher(
        Float32MultiArray,
        MOTOR_TOPIC,
        10,
    )
    commands.clear()
    _spin_for(harness, 0.15, held_forward)
    assert teleop._competitors
    assert commands[-1] == pytest.approx((0.0, 0.0))
    assert not teleop._arming_gate.armed
    harness['peer'].destroy_publisher(competitor)

    # Removing the only motor subscriber also disarms. Recreating it while RT
    # remains held must not resume until another neutral input is observed.
    harness['peer'].destroy_subscription(harness['motor_subscription'])
    harness['motor_subscription'] = None
    _spin_for(harness, 0.15, held_forward)
    assert not teleop._has_motor_subscriber
    assert not teleop._arming_gate.armed

    harness['motor_subscription'] = harness['peer'].create_subscription(
        Float32MultiArray,
        MOTOR_TOPIC,
        harness['on_motor'],
        10,
    )
    commands.clear()
    _spin_for(harness, 0.20, held_forward)
    assert teleop._has_motor_subscriber
    assert commands
    assert all(command[1] == 0.0 for command in commands)
    assert not teleop._arming_gate.armed

    _spin_for(harness, 0.15, neutral)
    assert teleop._arming_gate.armed
    commands.clear()
    _spin_for(harness, 0.10, held_forward)
    assert any(command[1] == pytest.approx(3.5) for command in commands)
