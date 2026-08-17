# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    artifact_id = LaunchConfiguration('artifact_id')
    artifact_root = LaunchConfiguration('artifact_root')
    use_camera = LaunchConfiguration('use_camera')
    use_gamepad = LaunchConfiguration('use_gamepad')
    allow_motion = LaunchConfiguration('allow_motion')
    curriculum_generation = LaunchConfiguration('curriculum_generation')
    speed_cap = LaunchConfiguration('speed_cap')
    wrapper = '/home/xytron/.local/lib/xycar-ai-gpu/run_gpu_policy.sh'
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value='/home/xytron/.config/xycar/guided_history_collection.yaml',
            ),
            DeclareLaunchArgument('artifact_id'),
            DeclareLaunchArgument(
                'artifact_root',
                default_value='/home/xytron/xycar_ws_mgw/artifacts/models',
            ),
            DeclareLaunchArgument('use_camera', default_value='true'),
            DeclareLaunchArgument('use_gamepad', default_value='true'),
            DeclareLaunchArgument('allow_motion'),
            DeclareLaunchArgument('curriculum_generation'),
            DeclareLaunchArgument('speed_cap'),
            OpaqueFunction(function=_validate_arguments),
            SetEnvironmentVariable('ARTIFACT_ID', artifact_id),
            SetEnvironmentVariable('ARTIFACT_ROOT', artifact_root),
            SetEnvironmentVariable(
                'HOST_POLICY_LAUNCH',
                'history_guided_collection.launch.py',
            ),
            ExecuteProcess(
                cmd=[
                    wrapper,
                    ['params_file:=', params_file],
                    ['use_camera:=', use_camera],
                    ['use_gamepad:=', use_gamepad],
                    ['allow_motion:=', allow_motion],
                    ['curriculum_generation:=', curriculum_generation],
                    ['speed_cap:=', speed_cap],
                ],
                output='screen',
                emulate_tty=True,
            ),
        ]
    )


def _validate_arguments(context):
    configured = Path(LaunchConfiguration('params_file').perform(context))
    if not configured.is_absolute() or not configured.is_file():
        raise RuntimeError(f'params_file must be an existing absolute YAML: {configured}')
    if float(LaunchConfiguration('speed_cap').perform(context)) != 30.0:
        raise RuntimeError('history guided speed_cap must be exactly 30')
    generation = int(LaunchConfiguration('curriculum_generation').perform(context))
    if generation < 0:
        raise RuntimeError('curriculum_generation must be non-negative')
    return []
