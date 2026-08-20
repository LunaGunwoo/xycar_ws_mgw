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
                ],
                output='screen',
                emulate_tty=True,
            ),
        ]
    )
