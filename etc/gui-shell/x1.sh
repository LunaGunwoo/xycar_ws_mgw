#!/bin/bash
source /home/xytron/.bashrc
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
export NO_AT_BRIDGE=1
ros2 launch app_viewer_all app_viewer_all.launch.py
