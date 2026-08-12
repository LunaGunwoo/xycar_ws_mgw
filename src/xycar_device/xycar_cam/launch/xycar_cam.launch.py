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

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='v4l2_camera',
                executable='v4l2_camera_node',
                name='xycar_cam',
                output='screen',
                parameters=[
                    {
                        'video_device': '/dev/videoCAM',
                        'pixel_format': 'YUYV',
                        'output_encoding': 'rgb8',
                        'image_size': [640, 480],
                        'time_per_frame': [1, 30],
                        'camera_frame_id': 'camera',
                    }
                ],
            )
        ]
    )
