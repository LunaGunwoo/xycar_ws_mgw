# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    track_share = get_package_share_directory("track_drive")
    debug_share = get_package_share_directory("xycar_debug")
    tuning_file = LaunchConfiguration("tuning_file")
    viewer_tuning_file = LaunchConfiguration("viewer_tuning_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "tuning_file",
                default_value=os.path.join(
                    track_share,
                    "config",
                    "cone_drive.yaml",
                ),
            ),
            DeclareLaunchArgument(
                "viewer_tuning_file",
                default_value=os.path.join(
                    debug_share,
                    "config",
                    "cone_viewer.yaml",
                ),
            ),
            Node(
                package="xycar_debug",
                executable="cone_path_viewer",
                name="cone_path_viewer",
                output="screen",
                parameters=[
                    {"tuning_file": tuning_file},
                    {"viewer_tuning_file": viewer_tuning_file},
                ],
            ),
        ]
    )
