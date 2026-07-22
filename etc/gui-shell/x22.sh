#!/bin/bash
source /home/xytron/.bashrc
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar

# 첫 번째 터미널에서 ROS2 launch 실행
gnome-terminal -- bash -c "echo 'Running turtlesim node...'; ros2 run turtlesim turtlesim_node; exec bash" &
# 두 번째 터미널에서 turtle_teleop_key 실행
gnome-terminal -- bash -c "echo 'Running turtle_teleop_key...'; ros2 run turtlesim turtle_teleop_key; exec bash"
