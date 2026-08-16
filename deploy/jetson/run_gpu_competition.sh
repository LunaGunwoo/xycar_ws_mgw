#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd)
source "${SCRIPT_DIR}/images.lock.env"

COMPETITION_BUNDLE_ID=${COMPETITION_BUNDLE_ID:-}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/home/xytron/xycar_ws_mgw/artifacts/models}
COMPETITION_RUN_MODE=${COMPETITION_RUN_MODE:-signal_shadow}
ALLOW_MOTION=${ALLOW_MOTION:-false}
USE_GAMEPAD=${USE_GAMEPAD:-true}
case "${COMPETITION_RUN_MODE}" in
    signal_shadow|shortcut_only|combined) ;;
    *)
        echo "[ERROR] 잘못된 COMPETITION_RUN_MODE: ${COMPETITION_RUN_MODE}" >&2
        exit 1
        ;;
esac
if [ -z "${COMPETITION_BUNDLE_ID}" ]; then
    echo "[ERROR] COMPETITION_BUNDLE_ID를 명시하세요." >&2
    exit 1
fi
if [ "${COMPETITION_RUN_MODE}" = signal_shadow ]; then
    ALLOW_MOTION=false
    USE_GAMEPAD=false
elif [ "${ALLOW_MOTION}" != true ]; then
    echo "[ERROR] 주행 mode는 ALLOW_MOTION=true를 명시해야 합니다." >&2
    exit 1
fi

SOCKET_DIR="/run/user/$(id -u)/xycar-ai"
SOCKET_PATH="${SOCKET_DIR}/competition.sock"
CONTAINER=xycar_ai_competition_gpu
INNER_LAUNCH_PID=""
export PYTHONNOUSERSITE=1

cleanup() {
    set +e
    if [ -n "${INNER_LAUNCH_PID}" ] \
        && kill -0 "${INNER_LAUNCH_PID}" 2>/dev/null; then
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

set +u
source /opt/ros/humble/setup.bash
source /home/xytron/xycar_ws_mgw/install/setup.bash
set -u
export ROS_DOMAIN_ID=7
export ROS_LOCALHOST_ONLY=1
export ROS_NAMESPACE=xycar
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

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

BUNDLE_PATH="${ARTIFACT_ROOT}/${COMPETITION_BUNDLE_ID}"
if [ ! -f "${BUNDLE_PATH}/SHA256SUMS" ]; then
    echo "[ERROR] competition bundle이 없습니다: ${BUNDLE_PATH}" >&2
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
    --entrypoint python3 \
    "${GPU_IMAGE}" \
    -m xycar_ai_drive.competition_ipc \
    --artifact-dir "/artifacts/${COMPETITION_BUNDLE_ID}" \
    --socket-path "${SOCKET_PATH}" \
    --device cuda \
    --warmup-count 3

for _ in $(seq 1 200); do
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
    echo "[ERROR] competition GPU socket이 준비되지 않았습니다." >&2
    docker logs "${CONTAINER}" >&2 || true
    exit 1
fi

setsid ros2 launch xycar_ai_drive competition_policy.launch.py \
    artifact_id:="${COMPETITION_BUNDLE_ID}" \
    artifact_root:="${ARTIFACT_ROOT}" \
    run_mode:="${COMPETITION_RUN_MODE}" \
    allow_motion:="${ALLOW_MOTION}" \
    use_gamepad:="${USE_GAMEPAD}" \
    inference_socket_path:="${SOCKET_PATH}" \
    "$@" &
INNER_LAUNCH_PID=$!

set +e
wait "${INNER_LAUNCH_PID}"
STATUS=$?
set -e
INNER_LAUNCH_PID=""
exit "${STATUS}"
