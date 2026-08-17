#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
DEPS_ROOT="${REPO_ROOT}/_deps"
SOURCE_ROOT="${DEPS_ROOT}/src"
VESC_ROOT="${SOURCE_ROOT}/f1tenth_vesc"
EXPECTED_COMMIT=1bc8251296abb3936da5f30821b6311d67e861b7

mkdir -p "${SOURCE_ROOT}"
if [ ! -e "${VESC_ROOT}" ]; then
    vcs import --input "${SCRIPT_DIR}/f1tenth_vesc_humble.repos" \
        "${SOURCE_ROOT}"
fi
if [ ! -d "${VESC_ROOT}/.git" ]; then
    echo "[ERROR] expected Git checkout is missing: ${VESC_ROOT}" >&2
    exit 1
fi
actual_commit=$(git -C "${VESC_ROOT}" rev-parse HEAD)
if [ "${actual_commit}" != "${EXPECTED_COMMIT}" ]; then
    echo "[ERROR] F1TENTH VESC commit mismatch: ${actual_commit}" >&2
    exit 1
fi
if [ -n "$(git -C "${VESC_ROOT}" status --short)" ]; then
    echo "[ERROR] F1TENTH VESC checkout is dirty: ${VESC_ROOT}" >&2
    exit 1
fi
cmp --silent "${SCRIPT_DIR}/F1TENTH_VESC_LICENSE" \
    "${VESC_ROOT}/LICENSE" || {
    echo "[ERROR] F1TENTH VESC license mismatch" >&2
    exit 1
}
echo "Pinned F1TENTH VESC source ready: ${VESC_ROOT}@${actual_commit}"
