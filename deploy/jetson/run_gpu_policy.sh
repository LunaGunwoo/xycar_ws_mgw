#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd)
source "${SCRIPT_DIR}/images.lock.env"

ARTIFACT_ID=${ARTIFACT_ID:-front-cam-policy-tiny-hflip-p05-patience5-e5-20260811}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/home/xytron/xycar_ws_mgw/artifacts/models}
HOST_POLICY_LAUNCH=${HOST_POLICY_LAUNCH:-front_cam_policy.launch.py}
case "${HOST_POLICY_LAUNCH}" in
    front_cam_policy.launch.py|guided_policy_collection.launch.py|history_policy.launch.py|history_guided_collection.launch.py) ;;
    *)
        echo "[ERROR] 지원하지 않는 host policy launch: ${HOST_POLICY_LAUNCH}" >&2
        exit 1
        ;;
esac
SOCKET_DIR="/run/user/$(id -u)/xycar-ai"
SOCKET_PATH="${SOCKET_DIR}/policy.sock"
CONTAINER=xycar_ai_gpu
INNER_LAUNCH_PID=""

# Vehicle ROS packages use the Ubuntu/ROS Python ABI.  Ignore arbitrary
# per-user pip packages (notably NumPy 2.x) for this host-side launch while
# leaving the isolated CUDA container and AI training environments unchanged.
export PYTHONNOUSERSITE=1

cleanup() {
    set +e
    if [ -n "${INNER_LAUNCH_PID}" ] && kill -0 "${INNER_LAUNCH_PID}" 2>/dev/null; then
        kill -INT -- "-${INNER_LAUNCH_PID}" 2>/dev/null
        sleep 1
        kill -TERM -- "-${INNER_LAUNCH_PID}" 2>/dev/null
        wait "${INNER_LAUNCH_PID}" 2>/dev/null
    fi
    docker stop --time 3 "${CONTAINER}" >/dev/null 2>&1
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ROS/colcon setup files are not guaranteed to be nounset-clean.  Keep the
# wrapper strict, but suspend nounset only while loading the generated setup.
set +u
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws_mgw/install/setup.bash
set -u
export ROS_DOMAIN_ID=7
export ROS_LOCALHOST_ONLY=1
export ROS_NAMESPACE=xycar
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Humble cv_bridge and Ubuntu OpenCV are built against the NumPy 1.x ABI.
# Validate before opening the camera or starting the CUDA container so a
# user-site NumPy 2.x override fails without leaving hardware processes behind.
python3 - <<'PY'
import numpy

if int(numpy.__version__.split('.', maxsplit=1)[0]) >= 2:
    raise SystemExit(
        '[ERROR] ROS host requires NumPy 1.x; loaded '
        f'{numpy.__version__} from {numpy.__file__}'
    )

import cv2  # noqa: F401
from cv_bridge import CvBridge  # noqa: F401
PY

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

setsid ros2 launch xycar_ai_drive "${HOST_POLICY_LAUNCH}" \
    artifact_id:="${ARTIFACT_ID}" \
    inference_backend:=unix \
    inference_device:=cuda \
    inference_socket_path:="${SOCKET_PATH}" \
    inference_rpc_timeout_sec:=0.20 \
    "$@" &
INNER_LAUNCH_PID=$!

set +e
wait "${INNER_LAUNCH_PID}"
STATUS=$?
set -e
INNER_LAUNCH_PID=""
exit "${STATUS}"
