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
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    usb_cam_dir = get_package_share_directory('usb_cam')

    # usb_cam package revisions in Humble use either params.yaml or
    # params_1.yaml.  Keep the vehicle launch compatible with both layouts.
    params_path = next((
        os.path.join(usb_cam_dir, 'config', filename)
        for filename in ('params.yaml', 'params_1.yaml')
        if os.path.isfile(os.path.join(usb_cam_dir, 'config', filename))
    ), None)
    if params_path is None:
        raise RuntimeError('usb_cam parameter file was not found')

    print(params_path)

    ld.add_action(Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='xycar_cam',
        arguments=['--ros-args', '--log-level', 'error'],
        parameters=[
            params_path,
            {
                'video_device': '/dev/videoCAM',
                'image_raw.enable_pub_plugins': [
                    'image_transport/raw',
                ],
            },
        ]
        ))

    return ld
