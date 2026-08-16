#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
x27_path="${HOME}/xycar_ws/etc/gui-shell/x27.sh"
gpu_runtime_dir="${HOME}/.local/lib/xycar-ai-gpu"
motor_runtime_dir="${HOME}/.local/lib/xycar-motor"
profile_dir="${HOME}/.config/xycar"
repo_root=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
install -d -m 0755 "${HOME}/.local/bin"
install -d -m 0755 "${profile_dir}"
backup_root="${HOME}/migration_backup/runtime-$(date +%Y%m%d-%H%M%S)"
install -d -m 0755 "${backup_root}"
if [ -e "${HOME}/.local/bin/motor" ] || [ -L "${HOME}/.local/bin/motor" ]; then
    cp -a "${HOME}/.local/bin/motor" "${backup_root}/motor"
fi
if [ -e "${HOME}/.local/bin/xycar-ai-gpu" ] \
    || [ -L "${HOME}/.local/bin/xycar-ai-gpu" ]; then
    cp -a "${HOME}/.local/bin/xycar-ai-gpu" \
        "${backup_root}/xycar-ai-gpu"
fi
if [ -e "${HOME}/.local/bin/xycar-ai-competition" ] \
    || [ -L "${HOME}/.local/bin/xycar-ai-competition" ]; then
    cp -a "${HOME}/.local/bin/xycar-ai-competition" \
        "${backup_root}/xycar-ai-competition"
fi
if [ ! -f "${x27_path}" ]; then
    echo "[ERROR] ${x27_path}를 찾을 수 없습니다." >&2
    exit 1
fi
cp -a "${x27_path}" "${backup_root}/x27.sh"
install -d -m 0755 "${motor_runtime_dir}"
install -m 0644 "${SCRIPT_DIR}/images.lock.env" \
    "${motor_runtime_dir}/images.lock.env"
install -m 0755 "${SCRIPT_DIR}/motor-jetson" \
    "${motor_runtime_dir}/motor-jetson"
ln -sfn "${motor_runtime_dir}/motor-jetson" "${HOME}/.local/bin/motor"
install -d -m 0755 "${gpu_runtime_dir}"
install -m 0644 "${SCRIPT_DIR}/images.lock.env" \
    "${gpu_runtime_dir}/images.lock.env"
install -m 0755 "${SCRIPT_DIR}/run_gpu_policy.sh" \
    "${gpu_runtime_dir}/run_gpu_policy.sh"
ln -sfn "${gpu_runtime_dir}/run_gpu_policy.sh" \
    "${HOME}/.local/bin/xycar-ai-gpu"
install -m 0755 "${SCRIPT_DIR}/run_gpu_competition.sh" \
    "${gpu_runtime_dir}/run_gpu_competition.sh"
ln -sfn "${gpu_runtime_dir}/run_gpu_competition.sh" \
    "${HOME}/.local/bin/xycar-ai-competition"
install -m 0755 "${SCRIPT_DIR}/x27-jetson.sh" "${x27_path}"
if [ ! -e "${profile_dir}/gamepad_stateless_manual.yaml" ] \
    && [ ! -L "${profile_dir}/gamepad_stateless_manual.yaml" ]; then
    install -m 0644 \
        "${repo_root}/src/xycar_data/config/gamepad_stateless_manual.yaml" \
        "${profile_dir}/gamepad_stateless_manual.yaml"
fi
if [ ! -e "${profile_dir}/guided_stateless_collection.yaml" ] \
    && [ ! -L "${profile_dir}/guided_stateless_collection.yaml" ]; then
    install -m 0644 \
        "${repo_root}/src/xycar_ai_drive/config/guided_stateless_collection.yaml" \
        "${profile_dir}/guided_stateless_collection.yaml"
fi
if [ ! -e "${profile_dir}/competition_mission_collection.yaml" ] \
    && [ ! -L "${profile_dir}/competition_mission_collection.yaml" ]; then
    install -m 0644 \
        "${repo_root}/src/xycar_data/config/competition_mission_collection.yaml" \
        "${profile_dir}/competition_mission_collection.yaml"
fi
echo "Runtime wrapper와 x27 launcher 설치 완료; backup=${backup_root}"
