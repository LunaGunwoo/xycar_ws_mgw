# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_share = get_package_share_directory('xycar_cam')
    use_camera = LaunchConfiguration('use_camera')
    camera_topic = LaunchConfiguration('camera_topic')
    initial_config_path = LaunchConfiguration('initial_config_path')
    output_config_path = LaunchConfiguration('output_config_path')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'use_camera',
                default_value='true',
                description='Start the vehicle camera driver.',
            ),
            DeclareLaunchArgument(
                'camera_topic',
                default_value='/image_raw',
                description='Live sensor_msgs/Image topic.',
            ),
            DeclareLaunchArgument(
                'initial_config_path',
                default_value=(
                    '/home/xytron/xycar_ws_mgw/ai/config/'
                    'front_cam_policy_preprocess.yaml'
                ),
                description='Read-only seed warp YAML.',
            ),
            DeclareLaunchArgument(
                'output_config_path',
                default_value=(
                    '/home/xytron/.config/xycar/'
                    'front_cam_policy_preprocess.yaml'
                ),
                description='Untracked YAML written only when S is pressed.',
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        camera_share,
                        'launch',
                        'xycar_cam.launch.py',
                    )
                ),
                condition=IfCondition(use_camera),
            ),
            Node(
                package='xycar_ai_drive',
                executable='live_warp_tuner',
                name='live_warp_tuner',
                namespace='/',
                output='screen',
                parameters=[
                    {
                        'camera_topic': camera_topic,
                        'initial_config_path': initial_config_path,
                        'output_config_path': output_config_path,
                    }
                ],
                on_exit=Shutdown(
                    reason='live warp tuner exited; stopping camera launch'
                ),
            ),
        ]
    )
