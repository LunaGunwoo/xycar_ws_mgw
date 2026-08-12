# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
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
                'params_file',
                default_value=(
                    '/home/xytron/.config/xycar/'
                    'guided_stateless_collection.yaml'
                ),
                description='External guided collection parameter YAML.',
            ),
            DeclareLaunchArgument(
                'artifact_id',
                description='Schema v1 stateless artifact ID.',
            ),
            DeclareLaunchArgument(
                'artifact_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument(
                'allow_motion',
                description='Explicit motion authorization for this run.',
            ),
            DeclareLaunchArgument('device_id', default_value='0'),
            DeclareLaunchArgument(
                'curriculum_generation',
                description='Explicit guided dataset generation.',
            ),
            DeclareLaunchArgument(
                'speed_cap',
                description='Explicit maximum executed forward speed.',
            ),
            OpaqueFunction(function=_require_params_file),
            SetEnvironmentVariable('ARTIFACT_ID', artifact_id),
            SetEnvironmentVariable('ARTIFACT_ROOT', artifact_root),
            SetEnvironmentVariable(
                'HOST_POLICY_LAUNCH',
                'guided_policy_collection.launch.py',
            ),
            ExecuteProcess(
                cmd=[
                    wrapper,
                    ['params_file:=', params_file],
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


def _require_params_file(context):
    configured = Path(LaunchConfiguration('params_file').perform(context))
    if not configured.is_absolute() or not configured.is_file():
        raise RuntimeError(
            f'params_file must be an existing absolute YAML file: {configured}'
        )
    return []
