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
        'competition_mission_collection_normalized_v2.yaml',
    )
    params_file = LaunchConfiguration('params_file')
    capture_kind = LaunchConfiguration('capture_kind')
    device_id = LaunchConfiguration('device_id')
    use_camera = LaunchConfiguration('use_camera')
    camera_share = get_package_share_directory('xycar_cam')

    return LaunchDescription(
        [
            SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '1'),
            DeclareLaunchArgument(
                'params_file',
                default_value=default_params,
                description='Mission sequence collection parameter YAML.',
            ),
            DeclareLaunchArgument(
                'capture_kind',
                description='Required dataset kind: signal or shortcut.',
            ),
            DeclareLaunchArgument('device_id', default_value='0'),
            DeclareLaunchArgument('use_camera', default_value='true'),
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
                remappings=[('joy', '/joy')],
            ),
            Node(
                package='xycar_data',
                executable='mission_sequence_collector',
                name='mission_sequence_collector',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'capture_kind': ParameterValue(
                            capture_kind,
                            value_type=str,
                        ),
                        'collection_profile_path': ParameterValue(
                            params_file,
                            value_type=str,
                        ),
                    },
                ],
            ),
        ]
    )
