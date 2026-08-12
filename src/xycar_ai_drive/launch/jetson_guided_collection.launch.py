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
    artifact_id = LaunchConfiguration('artifact_id')
    artifact_root = LaunchConfiguration('artifact_root')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    allow_motion = LaunchConfiguration('allow_motion')
    device_id = LaunchConfiguration('device_id')
    curriculum_generation = LaunchConfiguration('curriculum_generation')
    speed_cap = LaunchConfiguration('speed_cap')
    wrapper = '/home/xytron/.local/lib/xycar-ai-gpu/run_gpu_policy.sh'

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'artifact_id',
                default_value=(
                    'front-cam-policy-warp-angle-mean5-ar4-shared-'
                    'e14-20260811'
                ),
            ),
            DeclareLaunchArgument(
                'artifact_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument('allow_motion', default_value='true'),
            DeclareLaunchArgument('device_id', default_value='0'),
            DeclareLaunchArgument('curriculum_generation', default_value='1'),
            DeclareLaunchArgument('speed_cap', default_value='27.0'),
            SetEnvironmentVariable('ARTIFACT_ID', artifact_id),
            SetEnvironmentVariable('ARTIFACT_ROOT', artifact_root),
            SetEnvironmentVariable(
                'HOST_POLICY_LAUNCH',
                'guided_policy_collection.launch.py',
            ),
            ExecuteProcess(
                cmd=[
                    wrapper,
                    ['use_camera:=', use_camera],
                    ['use_gamepad:=', use_gamepad],
                    ['allow_motion:=', allow_motion],
                    ['device_id:=', device_id],
                    ['curriculum_generation:=', curriculum_generation],
                    ['speed_cap:=', speed_cap],
                ],
                output='screen',
                emulate_tty=True,
            ),
        ]
    )
