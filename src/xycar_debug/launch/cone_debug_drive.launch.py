# Copyright 2026 Gunwoo Moon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    debug_drive_tuning_file = LaunchConfiguration("debug_drive_tuning_file")
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
                "debug_drive_tuning_file",
                default_value=os.path.join(
                    debug_share,
                    "config",
                    "cone_debug_drive.yaml",
                ),
            ),
            Node(
                package="xycar_debug",
                executable="cone_debug_drive",
                name="cone_debug_drive",
                output="screen",
                parameters=[
                    {"tuning_file": tuning_file},
                    {"debug_drive_tuning_file": debug_drive_tuning_file},
                ],
            ),
        ]
    )
