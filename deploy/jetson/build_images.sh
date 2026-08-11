#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AI_DRIVE_CONTEXT=$(cd -- "${SCRIPT_DIR}/../../src/xycar_ai_drive" && pwd)
source "${SCRIPT_DIR}/images.lock.env"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "[ERROR] Jetson ARM64에서만 image를 build할 수 있습니다." >&2
    exit 1
fi

docker build --pull \
    --build-arg "ROS_NOETIC_IMAGE=${ROS_NOETIC_IMAGE}" \
    --tag "${MOTOR_IMAGE}" \
    --file "${SCRIPT_DIR}/Dockerfile.motor" \
    "${SCRIPT_DIR}"

docker build --pull \
    --build-arg "ROS_NOETIC_IMAGE=${ROS_NOETIC_IMAGE}" \
    --build-arg "ROS2_REPOS_URL=${ROS2_REPOS_URL}" \
    --build-arg "ROS2_REPOS_SHA256=${ROS2_REPOS_SHA256}" \
    --build-arg "RMW_HUMBLE_COMMIT=${RMW_HUMBLE_COMMIT}" \
    --build-arg "ROS1_BRIDGE_COMMIT=${ROS1_BRIDGE_COMMIT}" \
    --tag "${BRIDGE_IMAGE}" \
    --file "${SCRIPT_DIR}/Dockerfile.bridge" \
    "${SCRIPT_DIR}"

docker build --pull \
    --build-arg "NVIDIA_PYTORCH_IMAGE=${NVIDIA_PYTORCH_IMAGE}" \
    --tag "${GPU_IMAGE}" \
    --file "${SCRIPT_DIR}/Dockerfile.gpu" \
    "${AI_DRIVE_CONTEXT}"

for image in "${MOTOR_IMAGE}" "${BRIDGE_IMAGE}" "${GPU_IMAGE}"; do
    architecture=$(docker image inspect --format '{{.Architecture}}' "${image}")
    if [ "${architecture}" != "arm64" ]; then
        echo "[ERROR] ${image} architecture=${architecture}" >&2
        exit 1
    fi
done
