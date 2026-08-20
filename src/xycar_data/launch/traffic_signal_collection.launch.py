# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('xycar_data'),
        'config',
        'traffic_signal_collection_normalized_v2.yaml',
    )
    params_file = LaunchConfiguration('params_file')
    device_id = LaunchConfiguration('device_id')
    joy_topic = LaunchConfiguration('joy_topic')
    use_camera = LaunchConfiguration('use_camera')
    show_preview = LaunchConfiguration('show_preview')
    camera_share = get_package_share_directory('xycar_cam')

    return LaunchDescription(
        [
            SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '1'),
            DeclareLaunchArgument(
                'params_file',
                default_value=default_params,
                description='Traffic-signal collection parameter YAML.',
            ),
            DeclareLaunchArgument(
                'device_id',
                default_value='0',
                description='SDL game-controller device index.',
            ),
            DeclareLaunchArgument(
                'joy_topic',
                default_value='/traffic_signal_collector/joy',
                description='Dedicated Joy topic for this collector.',
            ),
            DeclareLaunchArgument(
                'use_camera',
                default_value='true',
                description='Start the camera driver.',
            ),
            DeclareLaunchArgument(
                'show_preview',
                default_value='false',
                description='Show the annotated collector preview.',
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
                package='joy',
                executable='game_controller_node',
                name='game_controller_node',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'device_id': ParameterValue(
                            device_id,
                            value_type=int,
                        )
                    },
                ],
                remappings=[('joy', joy_topic)],
            ),
            Node(
                package='xycar_data',
                executable='traffic_signal_collector',
                name='traffic_signal_collector',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'collection_profile_path': ParameterValue(
                            params_file,
                            value_type=str,
                        ),
                        'joy_topic': ParameterValue(
                            joy_topic,
                            value_type=str,
                        ),
                        'preview_enabled': ParameterValue(
                            show_preview,
                            value_type=bool,
                        ),
                    },
                ],
            ),
            Node(
                package='image_view',
                executable='image_view',
                name='traffic_signal_image_view',
                namespace='/',
                output='screen',
                remappings=[
                    ('image', '/traffic_signal_collector/preview'),
                ],
                condition=IfCondition(show_preview),
            ),
        ]
    )
