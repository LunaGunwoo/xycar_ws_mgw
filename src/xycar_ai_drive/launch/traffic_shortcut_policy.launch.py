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
    bundle_root = LaunchConfiguration('bundle_root')
    bundle_id = LaunchConfiguration('bundle_id')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    allow_motion = LaunchConfiguration('allow_motion')
    device_id = LaunchConfiguration('device_id')
    base_socket_path = LaunchConfiguration('base_socket_path')
    shortcut_socket_path = LaunchConfiguration('shortcut_socket_path')
    inference_timeout = LaunchConfiguration('inference_timeout_sec')
    rpc_timeout = LaunchConfiguration('inference_rpc_timeout_sec')
    initial_stop_arm_button_index = LaunchConfiguration(
        'initial_stop_arm_button_index'
    )
    signal_status_log_hz = LaunchConfiguration('signal_status_log_hz')
    signal_bbox_width_min = LaunchConfiguration(
        'signal_bbox_width_min_px'
    )
    signal_stop_wait = LaunchConfiguration('signal_stop_wait_sec')
    shortcut_duration = LaunchConfiguration('shortcut_duration_sec')
    use_monitor_gui = LaunchConfiguration('use_monitor_gui')
    monitor_refresh_hz = LaunchConfiguration('monitor_refresh_hz')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=os.path.join(
                    package_share,
                    'config',
                    'traffic_shortcut_policy.yaml',
                ),
            ),
            DeclareLaunchArgument(
                'bundle_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument('bundle_id'),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument('allow_motion', default_value='true'),
            DeclareLaunchArgument('device_id', default_value='0'),
            DeclareLaunchArgument(
                'base_socket_path',
                default_value='/run/user/1000/xycar-ai/traffic-base.sock',
            ),
            DeclareLaunchArgument(
                'shortcut_socket_path',
                default_value='/run/user/1000/xycar-ai/traffic-shortcut.sock',
            ),
            DeclareLaunchArgument(
                'inference_timeout_sec',
                default_value='0.50',
            ),
            DeclareLaunchArgument(
                'inference_rpc_timeout_sec',
                default_value='0.40',
            ),
            DeclareLaunchArgument(
                'initial_stop_arm_button_index',
                default_value='9',
            ),
            DeclareLaunchArgument(
                'signal_status_log_hz',
                default_value='2.0',
            ),
            DeclareLaunchArgument(
                'signal_bbox_width_min_px',
                default_value='40',
            ),
            DeclareLaunchArgument(
                'signal_stop_wait_sec',
                default_value='1.0',
            ),
            DeclareLaunchArgument(
                'shortcut_duration_sec',
                default_value='5.0',
            ),
            DeclareLaunchArgument(
                'use_monitor_gui',
                default_value='false',
            ),
            DeclareLaunchArgument(
                'monitor_refresh_hz',
                default_value='15.0',
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
                on_exit=Shutdown(
                    reason=(
                        'traffic shortcut gamepad exited; stopping mission'
                    )
                ),
            ),
            Node(
                package='xycar_ai_drive',
                executable='traffic_shortcut_policy',
                name='traffic_shortcut_policy',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'bundle_dir': PathJoinSubstitution(
                            [bundle_root, bundle_id]
                        ),
                        'allow_motion': ParameterValue(
                            allow_motion,
                            value_type=bool,
                        ),
                        'require_gamepad_hold': ParameterValue(
                            use_gamepad,
                            value_type=bool,
                        ),
                        'inference_device': 'cuda',
                        'base_socket_path': base_socket_path,
                        'shortcut_socket_path': shortcut_socket_path,
                        'inference_timeout_sec': ParameterValue(
                            inference_timeout,
                            value_type=float,
                        ),
                        'inference_rpc_timeout_sec': ParameterValue(
                            rpc_timeout,
                            value_type=float,
                        ),
                        'initial_stop_arm_button_index': ParameterValue(
                            initial_stop_arm_button_index,
                            value_type=int,
                        ),
                        'signal_status_log_hz': ParameterValue(
                            signal_status_log_hz,
                            value_type=float,
                        ),
                        'signal_bbox_width_min_px': ParameterValue(
                            signal_bbox_width_min,
                            value_type=int,
                        ),
                        'signal_stop_wait_sec': ParameterValue(
                            signal_stop_wait,
                            value_type=float,
                        ),
                        'shortcut_duration_sec': ParameterValue(
                            shortcut_duration,
                            value_type=float,
                        ),
                    },
                ],
                on_exit=Shutdown(
                    reason='traffic shortcut policy exited; stopping sensors'
                ),
            ),
            Node(
                package='xycar_ai_drive',
                executable='traffic_shortcut_monitor',
                name='traffic_shortcut_monitor',
                namespace='/',
                output='screen',
                condition=IfCondition(use_monitor_gui),
                parameters=[
                    params_file,
                    {
                        'bundle_dir': PathJoinSubstitution(
                            [bundle_root, bundle_id]
                        ),
                        'monitor_refresh_hz': ParameterValue(
                            monitor_refresh_hz,
                            value_type=float,
                        ),
                    },
                ],
                on_exit=Shutdown(
                    reason=(
                        'traffic shortcut monitor exited; stopping mission'
                    )
                ),
            ),
        ]
    )
