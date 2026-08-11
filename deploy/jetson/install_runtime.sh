#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
x27_path="${HOME}/xycar_ws/etc/gui-shell/x27.sh"
install -d -m 0755 "${HOME}/.local/bin"
backup_root="${HOME}/migration_backup/runtime-$(date +%Y%m%d-%H%M%S)"
install -d -m 0755 "${backup_root}"
if [ -e "${HOME}/.local/bin/motor" ] || [ -L "${HOME}/.local/bin/motor" ]; then
    cp -a "${HOME}/.local/bin/motor" "${backup_root}/motor"
fi
if [ ! -f "${x27_path}" ]; then
    echo "[ERROR] ${x27_path}를 찾을 수 없습니다." >&2
    exit 1
fi
cp -a "${x27_path}" "${backup_root}/x27.sh"
ln -sfn "${SCRIPT_DIR}/motor-jetson" "${HOME}/.local/bin/motor"
ln -sfn "${SCRIPT_DIR}/run_gpu_policy.sh" \
    "${HOME}/.local/bin/xycar-ai-gpu"
install -m 0755 "${SCRIPT_DIR}/x27-jetson.sh" "${x27_path}"
echo "Runtime wrapper와 x27 launcher 설치 완료; backup=${backup_root}"
