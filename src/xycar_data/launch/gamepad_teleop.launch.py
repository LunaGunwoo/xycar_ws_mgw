# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('xycar_data'),
        'config',
        'gamepad_teleop.yaml',
    )
    params_file = LaunchConfiguration('params_file')
    device_id = LaunchConfiguration('device_id')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=default_params,
                description='Gamepad and teleop parameter YAML file.',
            ),
            DeclareLaunchArgument(
                'device_id',
                default_value='0',
                description='SDL game-controller device index.',
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
                executable='gamepad_teleop',
                name='gamepad_teleop',
                namespace='/',
                output='screen',
                parameters=[params_file],
            ),
        ]
    )
