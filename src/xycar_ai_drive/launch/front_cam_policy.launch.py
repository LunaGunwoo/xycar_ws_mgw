# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('xycar_ai_drive')
    params_file = LaunchConfiguration('params_file')
    artifact_root = LaunchConfiguration('artifact_root')
    artifact_id = LaunchConfiguration('artifact_id')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    allow_motion = LaunchConfiguration('allow_motion')
    inference_backend = LaunchConfiguration('inference_backend')
    inference_device = LaunchConfiguration('inference_device')
    inference_socket_path = LaunchConfiguration('inference_socket_path')
    inference_rpc_timeout_sec = LaunchConfiguration(
        'inference_rpc_timeout_sec'
    )
    device_id = LaunchConfiguration('device_id')
    camera_share = get_package_share_directory('xycar_cam')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=os.path.join(
                    package_share,
                    'config',
                    'front_cam_policy.yaml',
                ),
                description='Front-camera policy parameter YAML.',
            ),
            DeclareLaunchArgument(
                'artifact_root',
                default_value=(
                    '/home/xytron/xycar_ws_mgw/artifacts/models'
                ),
                description='Vehicle model artifact parent directory.',
            ),
            DeclareLaunchArgument(
                'artifact_id',
                default_value='front-cam-policy-baseline-e6-20260810',
                description='Versioned front-camera policy artifact id.',
            ),
            DeclareLaunchArgument(
                'use_camera',
                default_value='true',
                description='Start the camera driver.',
            ),
            DeclareLaunchArgument(
                'use_gamepad',
                default_value='true',
                description='Start joy/game_controller_node.',
            ),
            DeclareLaunchArgument(
                'allow_motion',
                default_value='true',
                description='Allow A-button toggles to publish nonzero motor commands.',
            ),
            DeclareLaunchArgument(
                'device_id',
                default_value='0',
                description='SDL game-controller device index.',
            ),
            DeclareLaunchArgument(
                'inference_backend',
                default_value='local',
                description='Policy backend: local or unix.',
            ),
            DeclareLaunchArgument(
                'inference_device',
                default_value='cpu',
                description='Required policy device: cpu or cuda.',
            ),
            DeclareLaunchArgument(
                'inference_socket_path',
                default_value='/run/user/1000/xycar-ai/policy.sock',
                description='Unix socket for the isolated policy server.',
            ),
            DeclareLaunchArgument(
                'inference_rpc_timeout_sec',
                default_value='0.20',
                description='Fail-closed policy RPC timeout.',
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
                executable='front_cam_policy',
                name='front_cam_policy',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'artifact_dir': PathJoinSubstitution(
                            [artifact_root, artifact_id]
                        ),
                        'allow_motion': ParameterValue(
                            allow_motion,
                            value_type=bool,
                        ),
                        'inference_backend': inference_backend,
                        'inference_device': inference_device,
                        'inference_socket_path': inference_socket_path,
                        'inference_rpc_timeout_sec': ParameterValue(
                            inference_rpc_timeout_sec,
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
