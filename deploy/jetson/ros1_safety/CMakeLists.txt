cmake_minimum_required(VERSION 3.0.2)
project(xycar_motor_safety)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS rospy std_msgs)

catkin_install_python(
  PROGRAMS scripts/xycar_motor_watchdog.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
