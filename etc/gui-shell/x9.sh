#!/bin/bash
source /home/xytron/.bashrc
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_MASTER_URI=http://10.42.0.1:11311
export ROS_HOSTNAME=10.42.0.1

export PYTHONDONTWRITEBYTECODE=1 

alias cm='cd /home/xytron/xycar_ws && catkin_make'

export PYTHONPATH=/home/xytron/xycar_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages

roslaunch xycar_sim_dqn dqn_study_demo.launch
