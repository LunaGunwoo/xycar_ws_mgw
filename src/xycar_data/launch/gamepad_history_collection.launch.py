# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('xycar_data')
    camera_share = get_package_share_directory('xycar_cam')
    params_file = LaunchConfiguration('params_file')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    device_id = LaunchConfiguration('device_id')
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=os.path.join(
                    package_share,
                    'config',
                    'gamepad_history_manual.yaml',
                ),
            ),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument('device_id', default_value='0'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(camera_share, 'launch', 'xycar_cam.launch.py')
                ),
                condition=IfCondition(use_camera),
            ),
            Node(
                package='joy',
                executable='game_controller_node',
                name='game_controller_node',
                namespace='/',
                output='screen',
                condition=IfCondition(use_gamepad),
                parameters=[
                    params_file,
                    {'device_id': ParameterValue(device_id, value_type=int)},
                ],
                remappings=[('joy', '/joy')],
            ),
            Node(
                package='xycar_data',
                executable='history_gamepad_collector',
                name='history_gamepad_collector',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {'collection_profile_path': params_file},
                ],
                on_exit=Shutdown(
                    reason='history manual collector exited; stopping sensors'
                ),
            ),
        ]
    )
