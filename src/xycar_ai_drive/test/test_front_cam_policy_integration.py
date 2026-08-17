# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import time
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool, Float32MultiArray
from xycar_msgs.msg import XycarMotor
from xycar_ai_drive.control import DriveCommand
from xycar_ai_drive.front_cam_policy_node import (
    FrontCamPolicyNode,
    _is_paired_unnamed_relay,
)
from xycar_ai_drive.history_policy_node import HistoryPolicyNode

JOY_TOPIC = '/xycar_ai_drive_test/joy'
CAMERA_TOPIC = '/xycar_ai_drive_test/camera'
MOTOR_TOPIC = '/xycar_ai_drive_test/motor'
PREDICTION_TOPIC = '/xycar_ai_drive_test/prediction'
ENABLED_TOPIC = '/xycar_ai_drive_test/enabled'


class _FakePolicy:
    def __init__(self, **_kwargs):
        self.reset_count = 0

    def reset_history(self):
        self.reset_count += 1

    def infer(self, _image):
        return SimpleNamespace(
            command=DriveCommand(angle=-18.0, speed=25.0),
            inference_ms=2.5,
        )


class _FakeHistoryPolicy:
    def __init__(self, **_kwargs):
        self.artifact = SimpleNamespace(
            history=SimpleNamespace(
                frames=4,
                initial_class_ids=(100, 100),
                update='externally_executed_commands',
                sample_clock='camera_frame',
            )
        )
        self.histories = []
        self.reset_count = 0

    def reset_history(self):
        self.reset_count += 1

    def infer(self, _image, history):
        self.histories.append(tuple(history))
        time.sleep(0.05)
        return SimpleNamespace(
            command=DriveCommand(angle=-10.0, speed=20.0),
            inference_ms=50.0,
        )


def _joy(a=False):
    message = Joy()
    message.buttons = [int(a)]
    return message


def _image(sequence):
    message = Image()
    message.header.stamp.sec = sequence
    message.height = 8
    message.width = 12
    message.encoding = 'rgb8'
    message.is_bigendian = False
    message.step = message.width * 3
    message.data = bytes([sequence % 256] * (message.step * message.height))
    return message


def _endpoint(name, namespace, participant, entity):
    return SimpleNamespace(
        node_name=name,
        node_namespace=namespace,
        endpoint_gid=bytes(participant) + bytes(entity),
    )


def test_only_paired_unnamed_bridge_endpoint_is_recognized():
    participant = bytes(range(12))
    publisher = _endpoint(
        '_NODE_NAME_UNKNOWN_',
        '_NODE_NAMESPACE_UNKNOWN_',
        participant,
        [0, 0, 0x12, 0x03],
    )
    subscriber = _endpoint(
        '_NODE_NAME_UNKNOWN_',
        '_NODE_NAMESPACE_UNKNOWN_',
        participant,
        [0, 0, 0x13, 0x04],
    )
    assert _is_paired_unnamed_relay(publisher, [subscriber])

    different_participant = bytes([99]) + participant[1:]
    unrelated = _endpoint(
        '_NODE_NAME_UNKNOWN_',
        '_NODE_NAMESPACE_UNKNOWN_',
        different_participant,
        [0, 0, 0x13, 0x04],
    )
    assert not _is_paired_unnamed_relay(publisher, [unrelated])

    named_publisher = _endpoint(
        'other_motor_publisher',
        '/',
        participant,
        [0, 0, 0x12, 0x03],
    )
    assert not _is_paired_unnamed_relay(named_publisher, [subscriber])


def _spin_for(harness, duration_sec, *, a=None, publish_camera=True):
    deadline = time.monotonic() + duration_sec
    next_publish = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_publish:
            if a is not None:
                harness['joy_publisher'].publish(_joy(a))
            if publish_camera:
                harness['camera_sequence'] += 1
                harness['camera_publisher'].publish(
                    _image(harness['camera_sequence'])
                )
            next_publish = now + 0.02
        harness['executor'].spin_once(timeout_sec=0.01)


def _spin_until(harness, predicate, timeout_sec, **spin_kwargs):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        _spin_for(harness, 0.05, **spin_kwargs)
        if predicate():
            return True
    return predicate()


@pytest.fixture
def ros_harness(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '223')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    rclpy.init(args=[])
    overrides = [
        Parameter('artifact_dir', value='/unused/fake-artifact'),
        Parameter('joy_topic', value=JOY_TOPIC),
        Parameter('camera_topic', value=CAMERA_TOPIC),
        Parameter('motor_topic', value=MOTOR_TOPIC),
        Parameter('prediction_topic', value=PREDICTION_TOPIC),
        Parameter('enabled_topic', value=ENABLED_TOPIC),
        Parameter('publish_rate_hz', value=20.0),
        Parameter('joy_timeout_sec', value=0.15),
        Parameter('inference_timeout_sec', value=0.15),
        Parameter('graph_check_period_sec', value=0.03),
        Parameter('stop_publish_count', value=1),
        Parameter('warmup_count', value=0),
    ]
    policy = FrontCamPolicyNode(
        parameter_overrides=overrides,
        policy_factory=_FakePolicy,
    )
    peer = Node('front_cam_policy_test_peer')
    commands = []
    predictions = []
    enabled_states = []
    joy_publisher = peer.create_publisher(Joy, JOY_TOPIC, 10)
    camera_publisher = peer.create_publisher(Image, CAMERA_TOPIC, 10)
    motor_subscription = peer.create_subscription(
        Float32MultiArray,
        MOTOR_TOPIC,
        lambda message: commands.append(tuple(message.data)),
        10,
    )
    prediction_subscription = peer.create_subscription(
        Float32MultiArray,
        PREDICTION_TOPIC,
        lambda message: predictions.append(tuple(message.data)),
        10,
    )
    enabled_subscription = peer.create_subscription(
        Bool,
        ENABLED_TOPIC,
        lambda message: enabled_states.append(message.data),
        10,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(policy)
    executor.add_node(peer)
    harness = {
        'executor': executor,
        'policy': policy,
        'peer': peer,
        'joy_publisher': joy_publisher,
        'camera_publisher': camera_publisher,
        'motor_subscription': motor_subscription,
        'prediction_subscription': prediction_subscription,
        'enabled_subscription': enabled_subscription,
        'commands': commands,
        'predictions': predictions,
        'enabled_states': enabled_states,
        'camera_sequence': 0,
    }
    yield harness

    policy.shutdown()
    executor.remove_node(policy)
    executor.remove_node(peer)
    policy.destroy_node()
    peer.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


def test_a_toggle_publish_rate_and_fault_rearming(ros_harness):
    harness = ros_harness
    policy = harness['policy']
    commands = harness['commands']

    assert _spin_until(
        harness,
        lambda: policy._has_motor_subscriber and bool(harness['predictions']),
        2.0,
        a=False,
    )
    commands.clear()
    _spin_for(harness, 0.15, a=False)
    assert commands
    assert all(command == pytest.approx((0.0, 0.0)) for command in commands)

    harness['joy_publisher'].publish(_joy(True))
    _spin_for(harness, 0.20, a=True)
    assert policy._toggle.enabled
    # An inference that started before the enable edge is discarded and may
    # trigger one additional reset before a post-reset frame is accepted.
    assert policy._policy.reset_count >= 1
    moving = [command for command in commands if command[1] > 0.0]
    assert len(moving) >= 2
    assert all(command == pytest.approx((-18.0, 25.0)) for command in moving)

    commands.clear()
    _spin_for(harness, 0.15, a=False)
    assert policy._toggle.enabled
    assert any(command[1] == pytest.approx(25.0) for command in commands)

    harness['joy_publisher'].publish(_joy(True))
    _spin_for(harness, 0.10, a=True)
    assert not policy._toggle.enabled
    assert commands[-1] == pytest.approx((0.0, 0.0))

    _spin_for(harness, 0.10, a=False)
    harness['joy_publisher'].publish(_joy(True))
    _spin_for(harness, 0.10, a=True)
    assert policy._toggle.enabled

    commands.clear()
    _spin_for(harness, 0.25, a=None)
    assert not policy._toggle.enabled
    assert commands[-1] == pytest.approx((0.0, 0.0))

    commands.clear()
    _spin_for(harness, 0.15, a=True)
    assert not policy._toggle.enabled
    assert all(command[1] == 0.0 for command in commands)
    _spin_for(harness, 0.10, a=False)
    harness['joy_publisher'].publish(_joy(True))
    _spin_for(harness, 0.10, a=True)
    assert policy._toggle.enabled

    competitor = harness['peer'].create_publisher(
        Float32MultiArray,
        MOTOR_TOPIC,
        10,
    )
    assert _spin_until(
        harness,
        lambda: (
            bool(policy._competitors)
            and not policy._toggle.enabled
            and bool(commands)
            and commands[-1] == (0.0, 0.0)
        ),
        2.0,
        a=True,
    )
    assert commands[-1] == pytest.approx((0.0, 0.0))
    harness['peer'].destroy_publisher(competitor)
    assert _spin_until(
        harness,
        lambda: not policy._competitors,
        2.0,
        a=True,
    )
    assert not policy._toggle.enabled
    _spin_for(harness, 0.10, a=False)
    harness['joy_publisher'].publish(_joy(True))
    _spin_for(harness, 0.10, a=True)
    assert policy._toggle.enabled

    commands.clear()
    policy.shutdown()
    _spin_for(harness, 0.05, a=None, publish_camera=False)
    assert commands
    assert commands[-1] == pytest.approx((0.0, 0.0))


@pytest.fixture
def history_ros_harness(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '224')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    rclpy.init(args=[])
    prefix = '/xycar_history_test'
    overrides = [
        Parameter('artifact_dir', value='/unused/fake-history-artifact'),
        Parameter('joy_topic', value=f'{prefix}/joy'),
        Parameter('camera_topic', value=f'{prefix}/camera'),
        Parameter('motor_topic', value=f'{prefix}/motor_command'),
        Parameter('executed_topic', value=f'{prefix}/motor_executed'),
        Parameter('ready_topic', value=f'{prefix}/ready'),
        Parameter('prediction_topic', value=f'{prefix}/prediction'),
        Parameter('enabled_topic', value=f'{prefix}/enabled'),
        Parameter('allow_motion', value=True),
        Parameter('joy_timeout_sec', value=0.30),
        Parameter('inference_timeout_sec', value=0.30),
        Parameter('execution_timeout_sec', value=0.20),
        Parameter('graph_check_period_sec', value=0.02),
        Parameter('metrics_period_sec', value=60.0),
        Parameter('stop_publish_count', value=1),
        Parameter('warmup_count', value=0),
    ]
    policy = HistoryPolicyNode(
        parameter_overrides=overrides,
        policy_factory=_FakeHistoryPolicy,
    )
    peer = Node('history_policy_test_peer')
    commands = []
    state = {'echo': True, 'camera_sequence': 0}
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    ready_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    executed_publisher = peer.create_publisher(
        XycarMotor,
        f'{prefix}/motor_executed',
        qos,
    )

    def execute(message):
        commands.append(message)
        if not state['echo']:
            return
        executed = XycarMotor()
        executed.header = message.header
        executed.angle = message.angle
        executed.speed = message.speed
        executed_publisher.publish(executed)

    motor_subscription = peer.create_subscription(
        XycarMotor,
        f'{prefix}/motor_command',
        execute,
        qos,
    )
    joy_publisher = peer.create_publisher(Joy, f'{prefix}/joy', 10)
    camera_publisher = peer.create_publisher(Image, f'{prefix}/camera', 10)
    ready_publisher = peer.create_publisher(
        Bool,
        f'{prefix}/ready',
        ready_qos,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(policy)
    executor.add_node(peer)
    harness = {
        'executor': executor,
        'policy': policy,
        'peer': peer,
        'commands': commands,
        'state': state,
        'joy_publisher': joy_publisher,
        'camera_publisher': camera_publisher,
        'ready_publisher': ready_publisher,
        'executed_publisher': executed_publisher,
        'motor_subscription': motor_subscription,
    }
    yield harness

    policy.shutdown()
    executor.remove_node(policy)
    executor.remove_node(peer)
    policy.destroy_node()
    peer.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


def _spin_history_for(harness, duration_sec, *, a=False, camera=True):
    deadline = time.monotonic() + duration_sec
    next_frame = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        ready = Bool()
        ready.data = True
        harness['ready_publisher'].publish(ready)
        harness['joy_publisher'].publish(_joy(a))
        if camera and now >= next_frame:
            harness['state']['camera_sequence'] += 1
            harness['camera_publisher'].publish(
                _image(harness['state']['camera_sequence'])
            )
            next_frame = now + 1.0 / 30.0
        harness['executor'].spin_once(timeout_sec=0.005)


def test_history_policy_uses_latest_frame_and_mock_vesc_echo(
    history_ros_harness,
):
    harness = history_ros_harness
    policy = harness['policy']
    _spin_history_for(harness, 0.45, a=False)
    assert policy._has_motor_subscriber
    assert policy._last_inference_source_monotonic is not None

    _spin_history_for(harness, 0.10, a=False)
    _spin_history_for(harness, 0.55, a=True)
    assert policy._toggle.enabled
    moving = [
        command
        for command in harness['commands']
        if command.header.frame_id == 'history_policy_camera'
        and command.speed > 0.0
    ]
    assert moving
    assert policy._skipped_frames > 0
    assert any(
        history[-1] == (90, 120)
        for history in policy._policy.histories
    )

    _spin_history_for(harness, 0.20, a=True, camera=False)
    assert policy._pending is None
    assert policy._last_execution_stamp is not None
    duplicate = XycarMotor()
    duplicate.header.stamp.sec = policy._last_execution_stamp[0]
    duplicate.header.stamp.nanosec = policy._last_execution_stamp[1]
    duplicate.header.frame_id = 'history_policy_camera'
    duplicate.angle = -10.0
    duplicate.speed = 20.0
    harness['executed_publisher'].publish(duplicate)
    _spin_history_for(harness, 0.10, a=True, camera=False)
    assert policy._duplicate_executions == 1
    assert not policy._toggle.enabled
