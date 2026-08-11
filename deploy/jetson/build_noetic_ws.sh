#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/images.lock.env"
NOETIC_WS=${NOETIC_WS:-/home/xytron/noetic_ws}

if [ ! -d "${NOETIC_WS}/src" ]; then
    echo "[ERROR] noetic workspace source가 없습니다: ${NOETIC_WS}/src" >&2
    exit 1
fi

docker run --rm \
    --volume "${NOETIC_WS}:/root/noetic_ws" \
    --workdir /root/noetic_ws \
    "${MOTOR_IMAGE}" \
    bash -lc 'source /opt/ros/noetic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release'
