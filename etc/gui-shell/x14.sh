#!/bin/bash
export OPENBLAS_CORETYPE=ARMV8
export PYTHONDONTWRITEBYTECODE=1
source /home/xytron/.bashrc
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws/install/setup.bash
export ROS_MASTER_URI=http://10.42.0.1:11311
export ROS_HOSTNAME=10.42.0.1
tmp=`grep -r "export lidar_version" ~/.bashrc`
export lidar_version=`echo $tmp | awk -F'=' '{print $2}'`
tmp=`grep -r "export motor_version" ~/.bashrc`
export motor_version=`echo $tmp | awk -F'=' '{print $2}'`
tmp=`grep -r "export ROS_MASTER_URI" ~/.bashrc`
export ROS_MASTER_URI=`echo $tmp | awk -F'=' '{print $2}'`
tmp=`grep -r "export ROS_HOSTNAME" ~/.bashrc`
export ROS_HOSTNAME=`echo $tmp | awk -F'=' '{print $2}'`
cd /home/xytron/
notify-send "Carla Simulator" "잠시 후 Carla Simulator가 실행됩니다."
/home/xytron/carla/CarlaUE4.sh -ResX=640 -ResY=480 &
sleep 5
notify-send "Carla Ros Bridge" "잠시 후 Carla Ros Bridge가 실행됩니다."
roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch
