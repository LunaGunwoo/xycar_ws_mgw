#!/bin/bash
source /home/xytron/.bashrc
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
cd ~/xycar_ws/src/yolo_ros
ros2 launch yolo_bringup yolo.launch.py 2>&1 | grep -v "Compressed Depth Image Transport"
cd

