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
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('xycar_ai_drive')
    camera_share = get_package_share_directory('xycar_cam')
    params_file = LaunchConfiguration('params_file')
    artifact_id = LaunchConfiguration('artifact_id')
    artifact_root = LaunchConfiguration('artifact_root')
    run_mode = LaunchConfiguration('run_mode')
    allow_motion = LaunchConfiguration('allow_motion')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    device_id = LaunchConfiguration('device_id')
    socket_path = LaunchConfiguration('inference_socket_path')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=os.path.join(
                    package_share,
                    'config',
                    'competition_policy.yaml',
                ),
            ),
            DeclareLaunchArgument('artifact_id'),
            DeclareLaunchArgument(
                'artifact_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument(
                'run_mode',
                default_value='signal_shadow',
                description='signal_shadow, shortcut_only, or combined',
            ),
            DeclareLaunchArgument('allow_motion', default_value='false'),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument('device_id', default_value='0'),
            DeclareLaunchArgument(
                'inference_socket_path',
                default_value='/run/user/1000/xycar-ai/competition.sock',
            ),
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
                package='xycar_ai_drive',
                executable='competition_policy',
                name='competition_policy',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'artifact_dir': PathJoinSubstitution(
                            [artifact_root, artifact_id]
                        ),
                        'run_mode': ParameterValue(run_mode, value_type=str),
                        'allow_motion': ParameterValue(
                            allow_motion,
                            value_type=bool,
                        ),
                        'inference_socket_path': ParameterValue(
                            socket_path,
                            value_type=str,
                        ),
                    },
                ],
                on_exit=Shutdown(
                    reason='competition_policy exited; stopping sensor launch'
                ),
            ),
        ]
    )
