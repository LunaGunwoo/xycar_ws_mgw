# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bundle_id = LaunchConfiguration('bundle_id')
    bundle_root = LaunchConfiguration('bundle_root')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    allow_motion = LaunchConfiguration('allow_motion')
    device_id = LaunchConfiguration('device_id')
    rpc_timeout = LaunchConfiguration('inference_rpc_timeout_sec')
    initial_stop_arm_button_index = LaunchConfiguration(
        'initial_stop_arm_button_index'
    )
    signal_status_log_hz = LaunchConfiguration('signal_status_log_hz')
    use_monitor_gui = LaunchConfiguration('use_monitor_gui')
    monitor_refresh_hz = LaunchConfiguration('monitor_refresh_hz')
    wrapper = (
        '/home/xytron/.local/lib/xycar-ai-gpu/'
        'run_gpu_traffic_shortcut.sh'
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('bundle_id'),
            DeclareLaunchArgument(
                'bundle_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument('allow_motion', default_value='true'),
            DeclareLaunchArgument('device_id', default_value='0'),
            DeclareLaunchArgument(
                'inference_rpc_timeout_sec',
                default_value='0.20',
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
                'use_monitor_gui',
                default_value='false',
            ),
            DeclareLaunchArgument(
                'monitor_refresh_hz',
                default_value='15.0',
            ),
            SetEnvironmentVariable('TRAFFIC_SHORTCUT_BUNDLE_ID', bundle_id),
            SetEnvironmentVariable('ARTIFACT_ROOT', bundle_root),
            ExecuteProcess(
                cmd=[
                    wrapper,
                    ['use_camera:=', use_camera],
                    ['use_gamepad:=', use_gamepad],
                    ['allow_motion:=', allow_motion],
                    ['device_id:=', device_id],
                    ['inference_rpc_timeout_sec:=', rpc_timeout],
                    [
                        'initial_stop_arm_button_index:=',
                        initial_stop_arm_button_index,
                    ],
                    ['signal_status_log_hz:=', signal_status_log_hz],
                    ['use_monitor_gui:=', use_monitor_gui],
                    ['monitor_refresh_hz:=', monitor_refresh_hz],
                ],
                output='screen',
                emulate_tty=True,
            ),
        ]
    )
