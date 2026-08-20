#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd)
source "${SCRIPT_DIR}/images.lock.env"

TRAFFIC_SHORTCUT_BUNDLE_ID=${TRAFFIC_SHORTCUT_BUNDLE_ID:-}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/home/xytron/xycar_ws_mgw/artifacts/models}
if [ -z "${TRAFFIC_SHORTCUT_BUNDLE_ID}" ]; then
    echo "[ERROR] TRAFFIC_SHORTCUT_BUNDLE_ID를 명시하세요." >&2
    exit 1
fi

BUNDLE_PATH="${ARTIFACT_ROOT}/${TRAFFIC_SHORTCUT_BUNDLE_ID}"
SOCKET_DIR="/run/user/$(id -u)/xycar-ai"
BASE_SOCKET="${SOCKET_DIR}/traffic-base.sock"
SHORTCUT_SOCKET="${SOCKET_DIR}/traffic-shortcut.sock"
CONTAINER=xycar_ai_traffic_shortcut_gpu
INNER_LAUNCH_PID=""

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

if [ ! -f "${BUNDLE_PATH}/SHA256SUMS" ]; then
    echo "[ERROR] traffic shortcut bundle이 없습니다: ${BUNDLE_PATH}" >&2
    exit 1
fi

# This mission intentionally uses the validated user-site ORT/NumPy pair.
# Do not set PYTHONNOUSERSITE: cv_bridge, OpenCV and ORT are checked together.
unset PYTHONNOUSERSITE
python3 - "${BUNDLE_PATH}" <<'PY'
import sys

import cv2
import numpy as np
import onnxruntime as ort
from cv_bridge import CvBridge  # noqa: F401
from xycar_ai_drive.traffic_shortcut_artifact import (
    load_traffic_shortcut_bundle,
)

if np.__version__ != '1.26.4':
    raise SystemExit(f'[ERROR] NumPy must be 1.26.4, got {np.__version__}')
if ort.__version__ != '1.24.0':
    raise SystemExit(
        f'[ERROR] ONNX Runtime must be 1.24.0, got {ort.__version__}'
    )
bundle = load_traffic_shortcut_bundle(sys.argv[1])
session = ort.InferenceSession(
    str(bundle.detector.model_path),
    providers=list(bundle.providers),
)
if tuple(session.get_providers()) != bundle.providers:
    raise SystemExit(
        '[ERROR] traffic ONNX providers must be CUDAExecutionProvider then '
        'CPUExecutionProvider'
    )
inputs = session.get_inputs()
outputs = session.get_outputs()
if len(inputs) != 1 or inputs[0].name != 'images' or list(inputs[0].shape) != [1, 3, 640, 640]:
    raise SystemExit('[ERROR] traffic ONNX input metadata mismatch')
if len(outputs) != 1 or outputs[0].name != 'output0' or list(outputs[0].shape) != [1, 5, 8400]:
    raise SystemExit('[ERROR] traffic ONNX output metadata mismatch')
image = np.zeros((1, 3, 640, 640), dtype=np.float32)
prediction = session.run(None, {'images': image})[0]
if prediction.shape != (1, 5, 8400) or not np.isfinite(prediction).all():
    raise SystemExit('[ERROR] traffic ONNX synthetic inference failed')
cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), cv2.COLOR_RGB2BGR)
PY

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
    -m xycar_ai_drive.dual_policy_ipc \
    --bundle-dir "/artifacts/${TRAFFIC_SHORTCUT_BUNDLE_ID}" \
    --base-socket-path "${BASE_SOCKET}" \
    --shortcut-socket-path "${SHORTCUT_SOCKET}" \
    --device cuda \
    --warmup-count 3 \
    --history-reset-timeout-sec 0.25

for _ in $(seq 1 200); do
    if [ -S "${BASE_SOCKET}" ] && [ -S "${SHORTCUT_SOCKET}" ]; then
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
        docker logs "${CONTAINER}" >&2 || true
        exit 1
    fi
    sleep 0.1
done
if [ ! -S "${BASE_SOCKET}" ] || [ ! -S "${SHORTCUT_SOCKET}" ]; then
    echo "[ERROR] dual GPU policy sockets are not ready." >&2
    docker logs "${CONTAINER}" >&2 || true
    exit 1
fi

setsid ros2 launch xycar_ai_drive traffic_shortcut_policy.launch.py \
    bundle_id:="${TRAFFIC_SHORTCUT_BUNDLE_ID}" \
    bundle_root:="${ARTIFACT_ROOT}" \
    base_socket_path:="${BASE_SOCKET}" \
    shortcut_socket_path:="${SHORTCUT_SOCKET}" \
    "$@" &
INNER_LAUNCH_PID=$!

set +e
wait "${INNER_LAUNCH_PID}"
STATUS=$?
set -e
INNER_LAUNCH_PID=""
exit "${STATUS}"
