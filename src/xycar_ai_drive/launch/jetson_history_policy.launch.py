# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    artifact_id = LaunchConfiguration('artifact_id')
    artifact_root = LaunchConfiguration('artifact_root')
    params_file = LaunchConfiguration('params_file')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    allow_motion = LaunchConfiguration('allow_motion')
    force_speed_zero = LaunchConfiguration('force_speed_zero')
    require_schema4 = LaunchConfiguration('require_schema4')
    wrapper = '/home/xytron/.local/lib/xycar-ai-gpu/run_gpu_policy.sh'
    return LaunchDescription(
        [
            DeclareLaunchArgument('artifact_id'),
            DeclareLaunchArgument(
                'artifact_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument(
                'params_file',
                default_value='/home/xytron/.config/xycar/history_policy.yaml',
            ),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument('allow_motion', default_value='true'),
            DeclareLaunchArgument('force_speed_zero', default_value='false'),
            DeclareLaunchArgument('require_schema4', default_value='true'),
            SetEnvironmentVariable('ARTIFACT_ID', artifact_id),
            SetEnvironmentVariable('ARTIFACT_ROOT', artifact_root),
            SetEnvironmentVariable('HOST_POLICY_LAUNCH', 'history_policy.launch.py'),
            ExecuteProcess(
                cmd=[
                    wrapper,
                    ['params_file:=', params_file],
                    ['use_camera:=', use_camera],
                    ['use_gamepad:=', use_gamepad],
                    ['allow_motion:=', allow_motion],
                    ['force_speed_zero:=', force_speed_zero],
                    ['require_schema4:=', require_schema4],
                ],
                output='screen',
                emulate_tty=True,
            ),
        ]
    )
