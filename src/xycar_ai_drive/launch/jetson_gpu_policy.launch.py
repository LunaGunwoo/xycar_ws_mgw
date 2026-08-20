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
    speed_cap = LaunchConfiguration('speed_cap')
    device_id = LaunchConfiguration('device_id')
    inference_rpc_timeout_sec = LaunchConfiguration(
        'inference_rpc_timeout_sec'
    )

    # Use the validated runtime copy installed together with images.lock.env.
    # This avoids depending on a mutable source checkout or shell symlink.
    wrapper = (
        '/home/xytron/.local/lib/xycar-ai-gpu/run_gpu_policy.sh'
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'artifact_id',
                description='Versioned Jetson CUDA policy artifact id.',
            ),
            DeclareLaunchArgument(
                'artifact_root',
                default_value=(
                    '/home/xytron/xycar_ws_mgw/artifacts/models'
                ),
                description='Jetson model artifact parent directory.',
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
                description='Allow A-button toggles to command motion.',
            ),
            DeclareLaunchArgument(
                'speed_cap',
                default_value='30.0',
                description='Hard ceiling applied before motor publish and history.',
            ),
            DeclareLaunchArgument(
                'device_id',
                default_value='0',
                description='SDL game-controller device index.',
            ),
            DeclareLaunchArgument(
                'inference_rpc_timeout_sec',
                default_value='0.20',
                description='Fail-closed policy RPC timeout.',
            ),
            SetEnvironmentVariable('ARTIFACT_ID', artifact_id),
            SetEnvironmentVariable('ARTIFACT_ROOT', artifact_root),
            ExecuteProcess(
                cmd=[
                    wrapper,
                    ['use_camera:=', use_camera],
                    ['use_gamepad:=', use_gamepad],
                    ['allow_motion:=', allow_motion],
                    ['speed_cap:=', speed_cap],
                    ['device_id:=', device_id],
                    [
                        'inference_rpc_timeout_sec:=',
                        inference_rpc_timeout_sec,
                    ],
                ],
                output='screen',
                emulate_tty=True,
            ),
        ]
    )
