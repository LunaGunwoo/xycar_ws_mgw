#!/bin/bash
source /home/xytron/.bashrc
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
ros2 launch xycar_cam xycar_cam_viewer.launch.py 2>&1 | grep -v "Compressed Depth Image Transport"
