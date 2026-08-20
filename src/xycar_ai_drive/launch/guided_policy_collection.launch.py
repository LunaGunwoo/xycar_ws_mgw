# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from xycar_ai_drive.collection_profile import validate_collection_profile


def generate_launch_description():
    package_share = get_package_share_directory('xycar_ai_drive')
    camera_share = get_package_share_directory('xycar_cam')
    params_file = LaunchConfiguration('params_file')
    artifact_root = LaunchConfiguration('artifact_root')
    artifact_id = LaunchConfiguration('artifact_id')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    allow_motion = LaunchConfiguration('allow_motion')
    device_id = LaunchConfiguration('device_id')
    inference_backend = LaunchConfiguration('inference_backend')
    inference_device = LaunchConfiguration('inference_device')
    inference_socket_path = LaunchConfiguration('inference_socket_path')
    inference_rpc_timeout_sec = LaunchConfiguration(
        'inference_rpc_timeout_sec'
    )
    curriculum_generation = LaunchConfiguration('curriculum_generation')
    speed_cap = LaunchConfiguration('speed_cap')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=os.path.join(
                    package_share,
                    'config',
                    'guided_policy_collection_normalized_v2.yaml',
                ),
            ),
            DeclareLaunchArgument(
                'artifact_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument(
                'artifact_id',
                description='Normalized steering policy artifact ID.',
            ),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument('allow_motion', default_value='true'),
            DeclareLaunchArgument('device_id', default_value='0'),
            DeclareLaunchArgument('curriculum_generation', default_value='1'),
            DeclareLaunchArgument('speed_cap', default_value='30.0'),
            DeclareLaunchArgument('inference_backend', default_value='local'),
            DeclareLaunchArgument('inference_device', default_value='cpu'),
            DeclareLaunchArgument(
                'inference_socket_path',
                default_value='/run/user/1000/xycar-ai/policy.sock',
            ),
            DeclareLaunchArgument(
                'inference_rpc_timeout_sec',
                default_value='0.20',
            ),
            OpaqueFunction(function=_require_params_file),
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
                package='xycar_ai_drive',
                executable='guided_policy_collector',
                name='guided_policy_collector',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'artifact_dir': PathJoinSubstitution(
                            [artifact_root, artifact_id]
                        ),
                        'collection_profile_path': ParameterValue(
                            params_file,
                            value_type=str,
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
                        'curriculum_generation': ParameterValue(
                            curriculum_generation,
                            value_type=int,
                        ),
                        'speed_cap': ParameterValue(
                            speed_cap,
                            value_type=float,
                        ),
                    },
                ],
                on_exit=Shutdown(
                    reason='guided collector exited; stopping sensor launch'
                ),
            ),
        ]
    )


def _require_params_file(context):
    configured = Path(LaunchConfiguration('params_file').perform(context))
    try:
        validate_collection_profile(str(configured))
    except ValueError as exc:
        raise RuntimeError(f'invalid params_file {configured}: {exc}') from exc
    return []
