#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
DEPS_ROOT="${REPO_ROOT}/_deps"
SOURCE_ROOT="${DEPS_ROOT}/src"
VESC_ROOT="${SOURCE_ROOT}/f1tenth_vesc"
EXPECTED_COMMIT=c47fccbbd10fb66db3faaaa6e469f2eedba2586f
FW218_PATCH="${SCRIPT_DIR}/patches/f1tenth-vesc-fw218-values.patch"

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
cmp --silent "${SCRIPT_DIR}/F1TENTH_VESC_LICENSE" \
    "${VESC_ROOT}/LICENSE" || {
    echo "[ERROR] F1TENTH VESC license mismatch" >&2
    exit 1
}

expected_patched_status=$' M vesc_driver/CMakeLists.txt\n M vesc_driver/package.xml\n M vesc_driver/src/vesc_interface.cpp\n M vesc_driver/src/vesc_packet.cpp\n?? vesc_driver/test/test_vesc_packet_fw218.cpp'
actual_status=$(git -C "${VESC_ROOT}" status --short --untracked-files=all)
if [ -z "${actual_status}" ]; then
    git -C "${VESC_ROOT}" apply --check "${FW218_PATCH}"
    git -C "${VESC_ROOT}" apply "${FW218_PATCH}"
elif [ "${actual_status}" != "${expected_patched_status}" ]; then
    echo "[ERROR] unexpected F1TENTH VESC source changes: ${VESC_ROOT}" >&2
    printf '%s\n' "${actual_status}" >&2
    exit 1
fi

(
    cd "${VESC_ROOT}"
    sha256sum --check --strict <<'EOF'
b1af6807906278e1b2b7d7100fe2dd337f380ed9c37ede198c3ccc821bdfe776  vesc_driver/CMakeLists.txt
4c471129e37008f3bc8ab8d71eae8e1796cf82c6d735d2b63caaa41aed8047cd  vesc_driver/package.xml
dde929d710db722e09373203a5a448d5c9ef9b7cd3aefa3fdedaf70881670c07  vesc_driver/src/vesc_interface.cpp
0e3dad7747ba02447484b7d3e63c38b0b08b1a99018e0651182d6f0ba5379c52  vesc_driver/src/vesc_packet.cpp
c68e2d2b53ccdf726fedd079d9b7234a3a2940caa05503e8ba34655573477e73  vesc_driver/test/test_vesc_packet_fw218.cpp
EOF
)
git -C "${VESC_ROOT}" apply --reverse --check "${FW218_PATCH}"

echo "Pinned F1TENTH VESC source ready with FW 2.18 values patch: ${VESC_ROOT}@${actual_commit}"
