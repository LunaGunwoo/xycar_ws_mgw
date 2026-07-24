# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import yaml

from xycar_data.teleop_recorder import TeleopRecorderNode


CAMERA_TOPIC = "/xycar_data_test/terminal/camera"
LIDAR_TOPIC = "/xycar_data_test/terminal/lidar"
MOTOR_TOPIC = "/xycar_data_test/terminal/motor"


def _image_message() -> Image:
    message = Image()
    message.height = 4
    message.width = 6
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = message.width * 3
    message.data = bytes([64] * (message.step * message.height))
    return message


def _write_tuning(path, recording_root):
    payload = {
        "topics": {
            "camera_topic": CAMERA_TOPIC,
            "lidar_topic": LIDAR_TOPIC,
            "motor_topic": MOTOR_TOPIC,
        },
        "control": {
            "publish_rate_hz": 20.0,
            "key_timeout_sec": 1.0,
            "graph_check_period_sec": 0.05,
            "stop_publish_count": 1,
            "min_angle": -100.0,
            "max_angle": 100.0,
            "angle_step": 10.0,
            "forward_speed": 8.0,
            "forward_boost_multiplier": 1.5,
            "reverse_speed": -8.0,
        },
        "sensors": {
            "camera_timeout_sec": 0.10,
            "camera_auto_start": True,
            "camera_discovery_timeout_sec": 0.10,
            "camera_start_timeout_sec": 1.0,
            "camera_shutdown_timeout_sec": 0.10,
            "lidar_timeout_sec": 0.30,
            "max_lidar_skew_sec": 0.20,
        },
        "recording": {
            "root_dir": str(recording_root),
            "png_compression": 0,
            "queue_size": 16,
            "min_free_space_mb": 0,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_fresh_synthetic_frame_enables_wasd_and_stale_frame_stops(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "223")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    tuning_path = tmp_path / "teleop_recorder.yaml"
    _write_tuning(tuning_path, tmp_path / "recording")
    rclpy.init(args=[])

    teleop = TeleopRecorderNode(
        parameter_overrides=[
            Parameter("tuning_file", value=str(tuning_path)),
        ]
    )
    peer = Node("terminal_teleop_test_peer")
    camera_publisher = peer.create_publisher(
        Image,
        CAMERA_TOPIC,
        qos_profile_sensor_data,
    )
    received_commands = []
    motor_subscription = peer.create_subscription(
        Float32MultiArray,
        MOTOR_TOPIC,
        lambda message: received_commands.append(tuple(message.data)),
        10,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(teleop)
    executor.add_node(peer)
    process_factories = []

    def spin_with_camera(_node, *, timeout_sec):
        camera_publisher.publish(_image_message())
        executor.spin_once(timeout_sec=timeout_sec)

    try:
        teleop.prepare_camera(
            spin_once=spin_with_camera,
            process_factory=lambda: process_factories.append(object()),
        )
        assert process_factories == []
        assert teleop._camera is not None

        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert teleop._has_motor_subscriber

        received_commands.clear()
        camera_publisher.publish(_image_message())
        executor.spin_once(timeout_sec=0.05)
        teleop.handle_key("w")
        deadline = time.monotonic() + 0.08
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert any(
            command == pytest.approx((0.0, 8.0))
            for command in received_commands
        )

        deadline = time.monotonic() + 0.20
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert received_commands[-1] == pytest.approx((0.0, 0.0))
        assert teleop.command_state.command.is_stop
    finally:
        teleop.shutdown()
        executor.remove_node(teleop)
        executor.remove_node(peer)
        peer.destroy_subscription(motor_subscription)
        peer.destroy_publisher(camera_publisher)
        teleop.destroy_node()
        peer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
