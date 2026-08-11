#!/bin/bash
set -euo pipefail

# Noetic's roslaunch environment hook reads this variable while setup.bash is
# sourced.  Keep nounset enabled for the runtime after supplying the standard
# local-master default used by the motor container.
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11311}
set +u
source /opt/ros/noetic/setup.bash
source /opt/ros1_bridge/setup.bash
set -u
exec "$@"
