# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    camera_share = get_package_share_directory("xycar_cam")
    lidar_share = get_package_share_directory("xycar_lidar")
    use_lidar = LaunchConfiguration("use_lidar")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_lidar",
                default_value="true",
                description="Start the optional LiDAR driver with the camera.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        camera_share,
                        "launch",
                        "xycar_cam.launch.py",
                    )
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        lidar_share,
                        "launch",
                        "xycar_lidar.launch.py",
                    )
                ),
                condition=IfCondition(use_lidar),
            ),
        ]
    )
