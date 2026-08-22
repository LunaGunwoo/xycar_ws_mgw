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
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    camera_share = get_package_share_directory('xycar_cam')
    bundle_root = LaunchConfiguration('bundle_root')
    bundle_id = LaunchConfiguration('bundle_id')
    camera_topic = LaunchConfiguration('camera_topic')
    use_camera = LaunchConfiguration('use_camera')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'bundle_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument('bundle_id'),
            DeclareLaunchArgument('camera_topic', default_value='/image_raw'),
            DeclareLaunchArgument('use_camera', default_value='true'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(camera_share, 'launch', 'xycar_cam.launch.py')
                ),
                condition=IfCondition(use_camera),
            ),
            Node(
                package='xycar_ai_drive',
                executable='traffic_light_viewer',
                name='traffic_light_viewer',
                namespace='/',
                output='screen',
                parameters=[
                    {
                        'bundle_dir': PathJoinSubstitution(
                            [bundle_root, bundle_id]
                        ),
                        'camera_topic': camera_topic,
                    }
                ],
                on_exit=Shutdown(
                    reason=(
                        'traffic-light viewer exited; stopping camera launch'
                    )
                ),
            ),
        ]
    )
