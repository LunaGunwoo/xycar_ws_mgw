#!/bin/bash
source /home/xytron/.bashrc
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
python /home/xytron/xycar_ws/src/xycar_application/app_cam_exposure/app_cam_exposure/pygame_cam_exposure.py
