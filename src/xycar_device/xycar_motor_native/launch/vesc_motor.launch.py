# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


_LEGACY_CONTAINERS = {'ros1_container', 'ros1_bridge_container'}


def generate_launch_description():
    package_share = get_package_share_directory('xycar_motor_native')
    params_file = LaunchConfiguration('params_file')
    motor_device = LaunchConfiguration('motor_device')
    mock_driver = LaunchConfiguration('mock_driver')
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=os.path.join(
                    package_share,
                    'config',
                    'native_vesc.yaml',
                ),
                description='Native gateway and VESC parameter YAML.',
            ),
            DeclareLaunchArgument(
                'motor_device',
                default_value='/dev/ttyMOTOR',
                description='Exclusive VESC serial device.',
            ),
            DeclareLaunchArgument(
                'mock_driver',
                default_value='false',
                description='Do not open serial; intended only for tests.',
            ),
            OpaqueFunction(function=_preflight),
            Node(
                package='xycar_motor_native',
                executable='native_motor_gateway',
                name='native_motor_gateway',
                namespace='/',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'require_vesc_feedback': ParameterValue(
                            PythonExpression(
                                [
                                    "'",
                                    mock_driver,
                                    "'.lower() not in "
                                    "['1', 'true', 'yes', 'on']",
                                ]
                            ),
                            value_type=bool,
                        )
                    },
                ],
                on_exit=Shutdown(
                    reason='native motor gateway exited; stopping VESC'
                ),
            ),
            Node(
                package='vesc_driver',
                executable='vesc_driver_node',
                name='vesc_driver_node',
                namespace='/xycar_native',
                output='screen',
                parameters=[params_file, {'port': motor_device}],
                condition=UnlessCondition(mock_driver),
                on_exit=Shutdown(
                    reason='VESC driver exited; stopping native gateway'
                ),
            ),
        ]
    )


def _preflight(context):
    configured = Path(LaunchConfiguration('params_file').perform(context))
    if not configured.is_absolute() or not configured.is_file():
        raise RuntimeError(
            f'params_file must be an existing absolute YAML file: {configured}'
        )
    mock = LaunchConfiguration('mock_driver').perform(context).lower()
    if mock in {'1', 'true', 'yes', 'on'}:
        return []
    device = Path(LaunchConfiguration('motor_device').perform(context))
    if not (device.is_char_device() or device.is_symlink()):
        raise RuntimeError(f'motor device is missing: {device}')
    resolved = device.resolve(strict=True)
    if not resolved.is_char_device():
        raise RuntimeError(f'motor device target is not a character device: {resolved}')
    if not os.access(resolved, os.R_OK | os.W_OK):
        raise RuntimeError(f'motor device is not readable and writable: {resolved}')
    result = subprocess.run(
        ['docker', 'ps', '--format', '{{.Names}}'],
        check=True,
        capture_output=True,
        text=True,
    )
    active = set(result.stdout.splitlines()) & _LEGACY_CONTAINERS
    if active:
        raise RuntimeError(
            'legacy ROS1 motor containers must be stopped first: '
            + ', '.join(sorted(active))
        )
    return []
