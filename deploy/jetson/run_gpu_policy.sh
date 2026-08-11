#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd)
source "${SCRIPT_DIR}/images.lock.env"

ARTIFACT_ID=${ARTIFACT_ID:-front-cam-policy-tiny-hflip-p05-patience5-e5-20260811}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/home/xytron/xycar_ws_mgw/artifacts/models}
SOCKET_DIR="/run/user/$(id -u)/xycar-ai"
SOCKET_PATH="${SOCKET_DIR}/policy.sock"
CONTAINER=xycar_ai_gpu

cleanup() {
    set +e
    docker stop --time 3 "${CONTAINER}" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

if [ ! -f "${ARTIFACT_ROOT}/${ARTIFACT_ID}/SHA256SUMS" ]; then
    echo "[ERROR] model artifact가 없습니다: ${ARTIFACT_ID}" >&2
    exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
    echo "[ERROR] 기존 ${CONTAINER}를 먼저 정상 종료하세요." >&2
    exit 1
fi
install -d -m 700 "${SOCKET_DIR}"

docker run --detach --rm \
    --name "${CONTAINER}" \
    --runtime nvidia \
    --network none \
    --user "$(id -u):$(id -g)" \
    --volume "${ARTIFACT_ROOT}:/artifacts:ro" \
    --volume "${SOCKET_DIR}:${SOCKET_DIR}" \
    "${GPU_IMAGE}" \
    --artifact-dir "/artifacts/${ARTIFACT_ID}" \
    --socket-path "${SOCKET_PATH}" \
    --device cuda \
    --warmup-count 3 \
    --history-reset-timeout-sec 0.25

for _ in $(seq 1 100); do
    if [ -S "${SOCKET_PATH}" ]; then
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
        docker logs "${CONTAINER}" >&2 || true
        exit 1
    fi
    sleep 0.1
done
if [ ! -S "${SOCKET_PATH}" ]; then
    echo "[ERROR] GPU policy socket이 준비되지 않았습니다." >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws_mgw/install/setup.bash
export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=xycar
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch xycar_ai_drive front_cam_policy.launch.py \
    artifact_id:="${ARTIFACT_ID}" \
    inference_backend:=unix \
    inference_device:=cuda \
    inference_socket_path:="${SOCKET_PATH}" \
    inference_rpc_timeout_sec:=0.20
