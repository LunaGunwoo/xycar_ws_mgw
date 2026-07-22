#!/bin/bash
source /home/xytron/.bashrc
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar

cd /home/xytron/xycar_ws/src/xycar_application/app_web_controller/web_server || exit
# 첫 번째 터미널에서 http.server 실행
gnome-terminal -- bash -c "echo 'Running Web Server...'; python3 -m http.server 8000; exec bash" &

ros2 launch app_web_controller app_web_controller.launch.py 2>&1 | grep -v "Compressed Depth Image Transport"
