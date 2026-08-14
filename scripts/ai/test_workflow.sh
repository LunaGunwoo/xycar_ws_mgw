#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

xycar_ai_require_command mktemp
xycar_ai_require_command mkfifo
xycar_ai_require_command rsync
[[ "${XYCAR_AI_DEFAULT_VEHICLE_SSH}" == "xytron@xycar-gpu" ]] ||
  xycar_ai_die "unexpected default vehicle Tailscale SSH target"
xycar_ai_validate_ssh_target \
  "${XYCAR_AI_DEFAULT_VEHICLE_SSH}" "default vehicle SSH target"

XYCAR_TEST_ROOT="$(mktemp -d)"
xycar_ai_cleanup_test() {
  if [[ -d "${XYCAR_TEST_ROOT}" ]]; then
    find "${XYCAR_TEST_ROOT}" -depth -mindepth 1 -delete
    rmdir "${XYCAR_TEST_ROOT}"
  fi
}
trap xycar_ai_cleanup_test EXIT

mkdir -p \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010101_001_session" \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010102_001_session_2" \
  "${XYCAR_TEST_ROOT}/dataset-source/20260724_010103_001_incomplete" \
  "${XYCAR_TEST_ROOT}/dataset-source/_recording_20260724_010104_001"
for directory in "${XYCAR_TEST_ROOT}"/dataset-source/*; do
  printf 'sample\n' >"${directory}/samples.csv"
done
printf 'partial\n' \
  >"${XYCAR_TEST_ROOT}/dataset-source/20260724_010101_001_session/frame.part"

XYCAR_TEST_ENV=(
  env
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_ALLOW_ROOT_FILESYSTEM_SHARED=1
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/dataset-source"
  XYCAR_AI_SHARED_DATASET_ROOT="${XYCAR_TEST_ROOT}/shared-dataset"
)

if "${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" --apply \
  >/dev/null 2>&1; then
  xycar_ai_die "publish accepted a missing SSD marker without --init"
fi
"${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" --init >/dev/null
[[ ! -e "${XYCAR_TEST_ROOT}/shared-dataset" ]]
"${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" --init --apply >/dev/null

XYCAR_SHARED_DESTINATION="${XYCAR_TEST_ROOT}/shared-dataset/teleop"
[[ -f "${XYCAR_TEST_ROOT}/shared-dataset/.xycar-ai-dataset-share" ]]
[[ -d "${XYCAR_SHARED_DESTINATION}/20260724_010101_001_session" ]]
[[ -d "${XYCAR_SHARED_DESTINATION}/20260724_010102_001_session_2" ]]
[[ -d "${XYCAR_SHARED_DESTINATION}/20260724_010103_001_incomplete" ]]
[[ ! -e "${XYCAR_SHARED_DESTINATION}/_recording_20260724_010104_001" ]]
[[ ! -e \
  "${XYCAR_SHARED_DESTINATION}/20260724_010101_001_session/frame.part" ]]

printf 'destination-only\n' \
  >"${XYCAR_SHARED_DESTINATION}/preserved-ssd-file"
printf 'new\n' >"${XYCAR_TEST_ROOT}/dataset-source/new-file"
"${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" >/dev/null
[[ ! -e "${XYCAR_SHARED_DESTINATION}/new-file" ]]
"${XYCAR_TEST_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/publish_dataset_ssd.sh" --apply >/dev/null
[[ -f "${XYCAR_SHARED_DESTINATION}/new-file" ]]
[[ -f "${XYCAR_SHARED_DESTINATION}/preserved-ssd-file" ]]

mkdir -p "${XYCAR_TEST_ROOT}/dataset-pulled"
XYCAR_PULL_ENV=(
  env
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_ALLOW_ROOT_FILESYSTEM_SHARED=1
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/dataset-pulled"
  XYCAR_AI_SHARED_DATASET_ROOT="${XYCAR_TEST_ROOT}/shared-dataset"
)
"${XYCAR_PULL_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" >/dev/null
[[ ! -e "${XYCAR_TEST_ROOT}/dataset-pulled/new-file" ]]
"${XYCAR_PULL_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --apply >/dev/null
[[ -f "${XYCAR_TEST_ROOT}/dataset-pulled/new-file" ]]
printf 'local-only\n' >"${XYCAR_TEST_ROOT}/dataset-pulled/preserved-local-file"
"${XYCAR_PULL_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --apply >/dev/null
[[ -f "${XYCAR_TEST_ROOT}/dataset-pulled/preserved-local-file" ]]

mkdir -p "${XYCAR_TEST_ROOT}/missing-marker"
if env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_ROOT_FILESYSTEM_SHARED=1 \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/dataset-pulled" \
  XYCAR_AI_SHARED_DATASET_ROOT="${XYCAR_TEST_ROOT}/missing-marker" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --apply \
  >/dev/null 2>&1; then
  xycar_ai_die "pull accepted a missing SSD marker"
fi

XYCAR_WINDOWS_SOURCE="${XYCAR_TEST_ROOT}/windows-source"
XYCAR_WINDOWS_DESTINATION="${XYCAR_TEST_ROOT}/windows-mirror"
mkdir -p "${XYCAR_WINDOWS_SOURCE}"
xycar_write_policy_session() {
  local root="$1"
  local name="$2"
  local complete="$3"
  local control_mode="$4"
  local max_forward_speed="$5"
  local session="${root}/${name}"
  mkdir -p "${session}/Images"
  printf 'source-%s\n' "${name}" >"${session}/Images/1.png"
  printf 'sample_index,image,angle,speed\n1,Images/1.png,0,20\n' \
    >"${session}/samples.csv"
  printf '%s\n' \
    'format_version: 1' \
    "complete: ${complete}" \
    'dataset_kind: camera_first_teleop_behavior_cloning' \
    "control_mode: ${control_mode}" \
    'sample_count: 1' \
    'gamepad:' \
    "  max_forward_speed: ${max_forward_speed}" \
    >"${session}/metadata.yaml"
}
xycar_write_policy_session "${XYCAR_WINDOWS_SOURCE}" \
  "20260811_010101_001_session" true gamepad 20.0
xycar_write_policy_session "${XYCAR_WINDOWS_SOURCE}" \
  "20260811_010102_001_session" true gamepad 25.0
xycar_write_policy_session "${XYCAR_WINDOWS_SOURCE}" \
  "20260811_010103_001_session" true gamepad 19.9
xycar_write_policy_session "${XYCAR_WINDOWS_SOURCE}" \
  "20260811_010104_001_session" false gamepad 25.0
xycar_write_policy_session "${XYCAR_WINDOWS_SOURCE}" \
  "20260811_010105_001_session" true terminal 25.0
xycar_write_policy_session "${XYCAR_WINDOWS_SOURCE}" \
  "_recording_20260811_010106_001" true gamepad 25.0
mkdir -p "${XYCAR_WINDOWS_SOURCE}/20260811_010107_001_session"
: >"${XYCAR_WINDOWS_SOURCE}/20260811_010107_001_session/metadata.yaml"
printf 'partial\n' \
  >"${XYCAR_WINDOWS_SOURCE}/20260811_010101_001_session/frame.part"
ln -s "${XYCAR_WINDOWS_SOURCE}" "${XYCAR_WINDOWS_DESTINATION}"

XYCAR_WINDOWS_ENV=(
  env
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1
  XYCAR_AI_WINDOWS_DATASET_ROOT="${XYCAR_WINDOWS_SOURCE}"
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_WINDOWS_DESTINATION}"
  XYCAR_AI_MIN_FORWARD_SPEED=20.0
)
"${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" --help |
  grep -q '^usage:'
if "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" --unknown \
  >/dev/null 2>&1; then
  xycar_ai_die "Windows sync accepted an unknown argument"
fi
if env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1 \
  XYCAR_AI_WINDOWS_DATASET_ROOT="${XYCAR_WINDOWS_SOURCE}" \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_WINDOWS_SOURCE}" \
  XYCAR_AI_MIN_FORWARD_SPEED=20.0 \
  "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" --init \
  >/dev/null 2>&1; then
  xycar_ai_die "Windows sync accepted identical source and destination paths"
fi
"${XYCAR_WINDOWS_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" --init >/dev/null
[[ -L "${XYCAR_WINDOWS_DESTINATION}" ]]
"${XYCAR_WINDOWS_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" --init --apply >/dev/null
[[ ! -L "${XYCAR_WINDOWS_DESTINATION}" ]]
[[ -f \
  "${XYCAR_WINDOWS_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" ]]
[[ -d "${XYCAR_WINDOWS_DESTINATION}/20260811_010101_001_session" ]]
[[ -d "${XYCAR_WINDOWS_DESTINATION}/20260811_010102_001_session" ]]
[[ ! -e "${XYCAR_WINDOWS_DESTINATION}/20260811_010103_001_session" ]]
[[ ! -e "${XYCAR_WINDOWS_DESTINATION}/20260811_010104_001_session" ]]
[[ ! -e "${XYCAR_WINDOWS_DESTINATION}/20260811_010105_001_session" ]]
[[ ! -e "${XYCAR_WINDOWS_DESTINATION}/_recording_20260811_010106_001" ]]
[[ ! -e "${XYCAR_WINDOWS_DESTINATION}/20260811_010107_001_session" ]]
[[ ! -e \
  "${XYCAR_WINDOWS_DESTINATION}/20260811_010101_001_session/frame.part" ]]

printf 'changed\n' \
  >"${XYCAR_WINDOWS_SOURCE}/20260811_010101_001_session/Images/1.png"
printf 'new\n' \
  >"${XYCAR_WINDOWS_SOURCE}/20260811_010101_001_session/Images/2.png"
printf 'destination-only\n' >"${XYCAR_WINDOWS_DESTINATION}/destination-only"
"${XYCAR_WINDOWS_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" >/dev/null
[[ "$(<"${XYCAR_WINDOWS_DESTINATION}/20260811_010101_001_session/Images/1.png")" != \
  "changed" ]]
[[ ! -e \
  "${XYCAR_WINDOWS_DESTINATION}/20260811_010101_001_session/Images/2.png" ]]
[[ -f "${XYCAR_WINDOWS_DESTINATION}/destination-only" ]]
"${XYCAR_WINDOWS_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" --checksum --apply >/dev/null
[[ "$(<"${XYCAR_WINDOWS_DESTINATION}/20260811_010101_001_session/Images/1.png")" == \
  "changed" ]]
[[ -f \
  "${XYCAR_WINDOWS_DESTINATION}/20260811_010101_001_session/Images/2.png" ]]
[[ ! -e "${XYCAR_WINDOWS_DESTINATION}/destination-only" ]]

find "${XYCAR_WINDOWS_SOURCE}/20260811_010102_001_session" \
  -depth -mindepth 1 -delete
rmdir "${XYCAR_WINDOWS_SOURCE}/20260811_010102_001_session"
"${XYCAR_WINDOWS_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" --apply >/dev/null
[[ ! -e "${XYCAR_WINDOWS_DESTINATION}/20260811_010102_001_session" ]]
[[ -f \
  "${XYCAR_WINDOWS_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" ]]

mkdir -p "${XYCAR_TEST_ROOT}/wrong-source" "${XYCAR_TEST_ROOT}/unmarked-dest"
ln -s "${XYCAR_TEST_ROOT}/wrong-source" "${XYCAR_TEST_ROOT}/wrong-link"
if env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1 \
  XYCAR_AI_WINDOWS_DATASET_ROOT="${XYCAR_WINDOWS_SOURCE}" \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/wrong-link" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" --init \
  >/dev/null 2>&1; then
  xycar_ai_die "Windows sync accepted a symlink to the wrong source"
fi
if env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1 \
  XYCAR_AI_WINDOWS_DATASET_ROOT="${XYCAR_WINDOWS_SOURCE}" \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/unmarked-dest" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_dataset_windows.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "Windows sync accepted an unmarked destination"
fi

XYCAR_DIRECT_SOURCE="${XYCAR_TEST_ROOT}/direct-ssd-teleop"
XYCAR_DIRECT_DESTINATION="${XYCAR_TEST_ROOT}/direct-local-teleop"
mkdir -p "${XYCAR_DIRECT_SOURCE}" "${XYCAR_DIRECT_DESTINATION}"
xycar_write_policy_session "${XYCAR_DIRECT_SOURCE}" \
  "20260811_020201_001_session" true gamepad 20.0
xycar_write_policy_session "${XYCAR_DIRECT_SOURCE}" \
  "20260811_020202_001_session" true gamepad 25.0
xycar_write_policy_session "${XYCAR_DIRECT_SOURCE}" \
  "20260811_020203_001_session" true gamepad 19.9
xycar_write_policy_session "${XYCAR_DIRECT_SOURCE}" \
  "20260811_020204_001_session" false gamepad 25.0
xycar_write_policy_session "${XYCAR_DIRECT_SOURCE}" \
  "20260811_020205_001_session" true terminal 25.0
xycar_write_policy_session "${XYCAR_DIRECT_SOURCE}" \
  "_recording_20260811_020206_001" true gamepad 25.0
mkdir -p "${XYCAR_DIRECT_SOURCE}/20260811_020207_001_session"
: >"${XYCAR_DIRECT_SOURCE}/20260811_020207_001_session/metadata.yaml"
printf 'partial\n' \
  >"${XYCAR_DIRECT_SOURCE}/20260811_020201_001_session/frame.part"
printf 'unexpected-source-marker\n' \
  >"${XYCAR_DIRECT_SOURCE}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"

printf '%s\n' "${XYCAR_AI_LOCAL_DATASET_MARKER_CONTENT}" \
  >"${XYCAR_DIRECT_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
xycar_write_policy_session "${XYCAR_DIRECT_DESTINATION}" \
  "20260811_020208_001_session" true gamepad 25.0
xycar_write_policy_session "${XYCAR_DIRECT_DESTINATION}" \
  "20260811_020209_001_session" true gamepad 15.0
printf 'root-local-only\n' \
  >"${XYCAR_DIRECT_DESTINATION}/root-local-only"

XYCAR_DIRECT_ENV=(
  env
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1
  XYCAR_AI_SSD_TELEOP_ROOT="${XYCAR_DIRECT_SOURCE}"
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_DIRECT_DESTINATION}"
  XYCAR_AI_MIN_FORWARD_SPEED=20.0
)
XYCAR_DIRECT_EMPTY_DESTINATION="${XYCAR_TEST_ROOT}/direct-empty-local-teleop"
mkdir -p "${XYCAR_DIRECT_EMPTY_DESTINATION}"
printf '%s\n' "${XYCAR_AI_LOCAL_DATASET_MARKER_CONTENT}" \
  >"${XYCAR_DIRECT_EMPTY_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1 \
  XYCAR_AI_SSD_TELEOP_ROOT="${XYCAR_DIRECT_SOURCE}" \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_DIRECT_EMPTY_DESTINATION}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct --mirror \
  >/dev/null
[[ -z "$(find "${XYCAR_DIRECT_EMPTY_DESTINATION}" -mindepth 1 \
  ! -name "${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" -print -quit)" ]]
"${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --help |
  grep -q -- '--direct'
"${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --help |
  grep -q -- '--all'
if "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --mirror \
  >/dev/null 2>&1; then
  xycar_ai_die "marker-based SSD pull accepted --mirror without --direct"
fi
if "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --all \
  >/dev/null 2>&1; then
  xycar_ai_die "marker-based SSD pull accepted --all without --direct"
fi
if "${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct --all --mirror \
  >/dev/null 2>&1; then
  xycar_ai_die "direct all-content pull accepted destructive mirror mode"
fi
if (xycar_ai_validate_external_windows_mount_values 9p /mnt/c) \
  >/dev/null 2>&1; then
  xycar_ai_die "direct SSD validation accepted the Windows C: drive"
fi
if (xycar_ai_validate_external_windows_mount_values ext4 /mnt/d) \
  >/dev/null 2>&1; then
  xycar_ai_die "direct SSD validation accepted a non-Windows filesystem"
fi
xycar_ai_validate_external_windows_mount_values 9p /mnt/d
if env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1 \
  XYCAR_AI_SSD_TELEOP_ROOT="${XYCAR_TEST_ROOT}/missing-direct-source" \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_DIRECT_DESTINATION}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct \
  >/dev/null 2>&1; then
  xycar_ai_die "direct SSD pull accepted a missing source"
fi
if env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1 \
  XYCAR_AI_SSD_TELEOP_ROOT="${XYCAR_DIRECT_SOURCE}" \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_DIRECT_SOURCE}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct \
  >/dev/null 2>&1; then
  xycar_ai_die "direct SSD pull accepted identical source and destination paths"
fi
mkdir -p "${XYCAR_TEST_ROOT}/direct-unmarked-destination"
if env \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1 \
  XYCAR_AI_SSD_TELEOP_ROOT="${XYCAR_DIRECT_SOURCE}" \
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_TEST_ROOT}/direct-unmarked-destination" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct \
  >/dev/null 2>&1; then
  xycar_ai_die "direct SSD pull accepted an unmarked destination"
fi

"${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct >/dev/null
[[ ! -e "${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session" ]]
[[ -d "${XYCAR_DIRECT_DESTINATION}/20260811_020208_001_session" ]]
"${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct --apply >/dev/null
[[ -d "${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session" ]]
[[ -d "${XYCAR_DIRECT_DESTINATION}/20260811_020202_001_session" ]]
[[ ! -e "${XYCAR_DIRECT_DESTINATION}/20260811_020203_001_session" ]]
[[ ! -e "${XYCAR_DIRECT_DESTINATION}/20260811_020204_001_session" ]]
[[ ! -e "${XYCAR_DIRECT_DESTINATION}/20260811_020205_001_session" ]]
[[ ! -e "${XYCAR_DIRECT_DESTINATION}/_recording_20260811_020206_001" ]]
[[ ! -e "${XYCAR_DIRECT_DESTINATION}/20260811_020207_001_session" ]]
[[ ! -e \
  "${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session/frame.part" ]]

XYCAR_DIRECT_ALL_DESTINATION="${XYCAR_TEST_ROOT}/direct-all-local-teleop"
mkdir -p "${XYCAR_DIRECT_ALL_DESTINATION}"
printf '%s\n' "${XYCAR_AI_LOCAL_DATASET_MARKER_CONTENT}" \
  >"${XYCAR_DIRECT_ALL_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
printf 'local-only\n' >"${XYCAR_DIRECT_ALL_DESTINATION}/local-only"
XYCAR_DIRECT_ALL_ENV=(
  env
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE=1
  XYCAR_AI_SSD_TELEOP_ROOT="${XYCAR_DIRECT_SOURCE}"
  XYCAR_AI_LOCAL_DATASET_ROOT="${XYCAR_DIRECT_ALL_DESTINATION}"
)
"${XYCAR_DIRECT_ALL_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct --all >/dev/null
[[ ! -e "${XYCAR_DIRECT_ALL_DESTINATION}/20260811_020203_001_session" ]]
"${XYCAR_DIRECT_ALL_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct --all --apply >/dev/null
[[ -d "${XYCAR_DIRECT_ALL_DESTINATION}/20260811_020203_001_session" ]]
[[ -d "${XYCAR_DIRECT_ALL_DESTINATION}/20260811_020204_001_session" ]]
[[ -d "${XYCAR_DIRECT_ALL_DESTINATION}/20260811_020205_001_session" ]]
[[ -f "${XYCAR_DIRECT_ALL_DESTINATION}/20260811_020207_001_session/metadata.yaml" ]]
[[ ! -e "${XYCAR_DIRECT_ALL_DESTINATION}/_recording_20260811_020206_001" ]]
[[ ! -e \
  "${XYCAR_DIRECT_ALL_DESTINATION}/20260811_020201_001_session/frame.part" ]]
[[ -f "${XYCAR_DIRECT_ALL_DESTINATION}/local-only" ]]
[[ "$(<"${XYCAR_DIRECT_ALL_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}")" \
  == "${XYCAR_AI_LOCAL_DATASET_MARKER_CONTENT}" ]]

XYCAR_DIRECT_SOURCE_IMAGE="${XYCAR_DIRECT_SOURCE}/20260811_020201_001_session/Images/1.png"
XYCAR_DIRECT_DESTINATION_IMAGE="${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session/Images/1.png"
printf 'AAAA\n' >"${XYCAR_DIRECT_SOURCE_IMAGE}"
"${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct --apply >/dev/null
[[ "$(<"${XYCAR_DIRECT_DESTINATION_IMAGE}")" == "AAAA" ]]
printf 'BBBB\n' >"${XYCAR_DIRECT_SOURCE_IMAGE}"
touch -r "${XYCAR_DIRECT_DESTINATION_IMAGE}" "${XYCAR_DIRECT_SOURCE_IMAGE}"
printf 'new\n' \
  >"${XYCAR_DIRECT_SOURCE}/20260811_020201_001_session/Images/2.png"
printf 'session-local-only\n' \
  >"${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session/local-only"
"${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" --direct --apply >/dev/null
[[ "$(<"${XYCAR_DIRECT_DESTINATION_IMAGE}")" == "AAAA" ]]
[[ -f \
  "${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session/Images/2.png" ]]
[[ -f \
  "${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session/local-only" ]]
[[ -d "${XYCAR_DIRECT_DESTINATION}/20260811_020208_001_session" ]]
"${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" \
  --direct --checksum >/dev/null
[[ "$(<"${XYCAR_DIRECT_DESTINATION_IMAGE}")" == "AAAA" ]]
"${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" \
  --direct --checksum --apply >/dev/null
[[ "$(<"${XYCAR_DIRECT_DESTINATION_IMAGE}")" == "BBBB" ]]

XYCAR_DIRECT_MIRROR_OUTPUT="${XYCAR_TEST_ROOT}/direct-mirror-output"
"${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" \
  --direct --mirror >"${XYCAR_DIRECT_MIRROR_OUTPUT}"
grep -Fq '*deleting managed session 20260811_020208_001_session' \
  "${XYCAR_DIRECT_MIRROR_OUTPUT}"
[[ -f \
  "${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session/local-only" ]]
[[ -d "${XYCAR_DIRECT_DESTINATION}/20260811_020208_001_session" ]]
"${XYCAR_DIRECT_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/pull_dataset_ssd.sh" \
  --direct --mirror --apply >/dev/null
[[ ! -e \
  "${XYCAR_DIRECT_DESTINATION}/20260811_020201_001_session/local-only" ]]
[[ ! -e "${XYCAR_DIRECT_DESTINATION}/20260811_020208_001_session" ]]
[[ -d "${XYCAR_DIRECT_DESTINATION}/20260811_020209_001_session" ]]
[[ -f "${XYCAR_DIRECT_DESTINATION}/root-local-only" ]]
[[ -f \
  "${XYCAR_DIRECT_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" ]]

XYCAR_LAN_FAKE_BIN="${XYCAR_TEST_ROOT}/lan-fake-bin"
XYCAR_LAN_SOURCE="${XYCAR_TEST_ROOT}/lan-source"
XYCAR_LAN_DESTINATION="${XYCAR_TEST_ROOT}/lan-destination"
mkdir -p \
  "${XYCAR_LAN_FAKE_BIN}" \
  "${XYCAR_LAN_SOURCE}/20260814_010101_001_session/Images" \
  "${XYCAR_LAN_SOURCE}/20260814_010102_001_incomplete/Images" \
  "${XYCAR_LAN_SOURCE}/_recording_20260814_010103_001/Images"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'if [[ "${XYCAR_TEST_POWERSHELL_FAIL:-0}" == "1" ]]; then' \
  '  exit 1' \
  'fi' \
  'printf "%s" "${XYCAR_TEST_POWERSHELL_OUTPUT:-100 Mbps|Ethernet|192.168.50.0/24|192.168.50.1}"' \
  >"${XYCAR_LAN_FAKE_BIN}/powershell.exe"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'while (($#)); do' \
  '  case "$1" in' \
  '    -o|-l|-p|-i|-F|-J)' \
  '      shift 2' \
  '      ;;' \
  '    --)' \
  '      shift' \
  '      break' \
  '      ;;' \
  '    -*)' \
  '      shift' \
  '      ;;' \
  '    *)' \
  '      shift' \
  '      break' \
  '      ;;' \
  '  esac' \
  'done' \
  '(($#)) || exit 2' \
  'if (($# == 1)); then' \
  '  exec bash -c "$1"' \
  'fi' \
  'exec "$@"' \
  >"${XYCAR_LAN_FAKE_BIN}/ssh"
chmod +x \
  "${XYCAR_LAN_FAKE_BIN}/powershell.exe" \
  "${XYCAR_LAN_FAKE_BIN}/ssh"
printf 'manual-one\n' \
  >"${XYCAR_LAN_SOURCE}/20260814_010101_001_session/Images/1.jpg"
printf 'sample_index,image,angle,speed\n1,Images/1.jpg,0,7\n' \
  >"${XYCAR_LAN_SOURCE}/20260814_010101_001_session/samples.csv"
printf 'incomplete-one\n' \
  >"${XYCAR_LAN_SOURCE}/20260814_010102_001_incomplete/Images/1.jpg"
printf 'active-one\n' \
  >"${XYCAR_LAN_SOURCE}/_recording_20260814_010103_001/Images/1.jpg"
printf 'partial\n' \
  >"${XYCAR_LAN_SOURCE}/20260814_010101_001_session/frame.part"
printf 'temporary\n' \
  >"${XYCAR_LAN_SOURCE}/20260814_010102_001_incomplete/frame.tmp"

XYCAR_LAN_ENV=(
  env
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}"
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)"
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_SOURCE}"
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_DESTINATION}"
)
"${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" --help |
  grep -q '^usage:'
if "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" --unknown \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted an unknown argument"
fi
"${XYCAR_LAN_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" >/dev/null
[[ -f \
  "${XYCAR_LAN_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" ]]
[[ -d "${XYCAR_LAN_DESTINATION}/20260814_010101_001_session" ]]
[[ -d "${XYCAR_LAN_DESTINATION}/20260814_010102_001_incomplete" ]]
[[ ! -e "${XYCAR_LAN_DESTINATION}/_recording_20260814_010103_001" ]]
[[ ! -e \
  "${XYCAR_LAN_DESTINATION}/20260814_010101_001_session/frame.part" ]]
[[ ! -e \
  "${XYCAR_LAN_DESTINATION}/20260814_010102_001_incomplete/frame.tmp" ]]

printf 'dry-run-only\n' \
  >"${XYCAR_LAN_SOURCE}/20260814_010101_001_session/Images/dry-run.jpg"
"${XYCAR_LAN_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" --dry-run >/dev/null
[[ ! -e \
  "${XYCAR_LAN_DESTINATION}/20260814_010101_001_session/Images/dry-run.jpg" ]]
find "${XYCAR_LAN_SOURCE}/20260814_010101_001_session/Images/dry-run.jpg" \
  -maxdepth 0 -type f -delete

printf 'manual-one-changed\n' \
  >"${XYCAR_LAN_SOURCE}/20260814_010101_001_session/Images/1.jpg"
printf 'manual-two\n' \
  >"${XYCAR_LAN_SOURCE}/20260814_010101_001_session/Images/2.jpg"
printf 'destination-only\n' >"${XYCAR_LAN_DESTINATION}/destination-only"
"${XYCAR_LAN_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" --checksum >/dev/null
[[ "$(<"${XYCAR_LAN_DESTINATION}/20260814_010101_001_session/Images/1.jpg")" == \
  "manual-one-changed" ]]
[[ -f \
  "${XYCAR_LAN_DESTINATION}/20260814_010101_001_session/Images/2.jpg" ]]
[[ ! -e "${XYCAR_LAN_DESTINATION}/destination-only" ]]

find "${XYCAR_LAN_SOURCE}/20260814_010102_001_incomplete" \
  -depth -mindepth 1 -delete
rmdir "${XYCAR_LAN_SOURCE}/20260814_010102_001_incomplete"
"${XYCAR_LAN_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" >/dev/null
[[ ! -e "${XYCAR_LAN_DESTINATION}/20260814_010102_001_incomplete" ]]
[[ -f \
  "${XYCAR_LAN_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" ]]

XYCAR_LAN_UNMARKED="${XYCAR_TEST_ROOT}/lan-unmarked"
mkdir -p "${XYCAR_LAN_UNMARKED}"
printf 'do-not-delete\n' >"${XYCAR_LAN_UNMARKED}/owned-file"
if env \
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}" \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1 \
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)" \
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_SOURCE}" \
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_UNMARKED}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted an unmarked non-empty destination"
fi
[[ -f "${XYCAR_LAN_UNMARKED}/owned-file" ]]

XYCAR_LAN_LINK="${XYCAR_TEST_ROOT}/lan-destination-link"
ln -s "${XYCAR_LAN_DESTINATION}" "${XYCAR_LAN_LINK}"
if env \
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}" \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1 \
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)" \
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_SOURCE}" \
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_LINK}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted a symlink destination"
fi

if env \
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}" \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1 \
  XYCAR_AI_LAN_EXPECTED_HOSTNAME=wrong-host \
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_SOURCE}" \
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_DESTINATION}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted the wrong vehicle hostname"
fi
if env \
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}" \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1 \
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)" \
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_TEST_ROOT}/missing-lan-source" \
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_DESTINATION}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted a missing vehicle root"
fi
if env \
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}" \
  XYCAR_TEST_POWERSHELL_OUTPUT='100 Mbps|Wi-Fi|192.168.50.0/24|192.168.50.1' \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1 \
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)" \
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_SOURCE}" \
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_DESTINATION}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted an unexpected Windows route"
fi
if env \
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}" \
  XYCAR_TEST_POWERSHELL_FAIL=1 \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1 \
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)" \
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_SOURCE}" \
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_DESTINATION}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted an unreachable vehicle SSH port"
fi

XYCAR_LAN_BAD_FS_BIN="${XYCAR_TEST_ROOT}/lan-bad-fs-bin"
mkdir -p "${XYCAR_LAN_BAD_FS_BIN}"
printf '%s\n' '#!/usr/bin/env bash' 'printf "xfs\\n"' \
  >"${XYCAR_LAN_BAD_FS_BIN}/findmnt"
chmod +x "${XYCAR_LAN_BAD_FS_BIN}/findmnt"
if env \
  PATH="${XYCAR_LAN_BAD_FS_BIN}:${XYCAR_LAN_FAKE_BIN}:${PATH}" \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)" \
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_SOURCE}" \
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_DESTINATION}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted a non-ext4 destination"
fi

(
  exec 8>"$(dirname -- "${XYCAR_LAN_DESTINATION}")/.stateless_manual_lan_sync.lock"
  flock -n 8
  if "${XYCAR_LAN_ENV[@]}" \
    "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
    >/dev/null 2>&1; then
    xycar_ai_die "LAN sync accepted a concurrent run"
  fi
)

XYCAR_LAN_EMPTY_SOURCE="${XYCAR_TEST_ROOT}/lan-empty-source"
mkdir -p "${XYCAR_LAN_EMPTY_SOURCE}"
printf 'remove-me\n' >"${XYCAR_LAN_DESTINATION}/remove-for-empty-mirror"
XYCAR_LAN_EMPTY_ENV=(
  env
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}"
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)"
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_EMPTY_SOURCE}"
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_DESTINATION}"
)
if "${XYCAR_LAN_EMPTY_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  >/dev/null 2>&1; then
  xycar_ai_die "LAN sync accepted an empty source without explicit approval"
fi
[[ -f "${XYCAR_LAN_DESTINATION}/remove-for-empty-mirror" ]]
"${XYCAR_LAN_EMPTY_ENV[@]}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" \
  --allow-empty-source >/dev/null
[[ ! -e "${XYCAR_LAN_DESTINATION}/remove-for-empty-mirror" ]]
[[ -f \
  "${XYCAR_LAN_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" ]]

XYCAR_LAN_EXISTING_EMPTY_DESTINATION="${XYCAR_TEST_ROOT}/lan-existing-empty"
mkdir -p "${XYCAR_LAN_EXISTING_EMPTY_DESTINATION}"
env \
  PATH="${XYCAR_LAN_FAKE_BIN}:${PATH}" \
  XYCAR_AI_ALLOW_ANY_CHECKOUT=1 \
  XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION=1 \
  XYCAR_AI_LAN_EXPECTED_HOSTNAME="$(hostname -s)" \
  XYCAR_AI_LAN_VEHICLE_DATASET_ROOT="${XYCAR_LAN_SOURCE}" \
  XYCAR_AI_LAN_LOCAL_DATASET_ROOT="${XYCAR_LAN_EXISTING_EMPTY_DESTINATION}" \
  "${XYCAR_AI_SCRIPT_DIR}/sync_stateless_manual_lan.sh" >/dev/null
[[ -f \
  "${XYCAR_LAN_EXISTING_EMPTY_DESTINATION}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" ]]

mkdir -p "${XYCAR_TEST_ROOT}/artifact"
printf 'schema_version: 1\nartifact_id: fixture\n' \
  >"${XYCAR_TEST_ROOT}/artifact/manifest.yaml"
printf 'model\n' >"${XYCAR_TEST_ROOT}/artifact/model.onnx"
(
  cd "${XYCAR_TEST_ROOT}/artifact"
  sha256sum manifest.yaml model.onnx >SHA256SUMS
)
xycar_ai_verify_sha_manifest "${XYCAR_TEST_ROOT}/artifact"
if (xycar_ai_validate_artifact_id '../unsafe') >/dev/null 2>&1; then
  xycar_ai_die "unsafe artifact id was accepted"
fi
if (xycar_ai_validate_absolute_path '/tmp/unsafe;path' 'test path') \
  >/dev/null 2>&1; then
  xycar_ai_die "unsafe absolute path was accepted"
fi
if (xycar_ai_validate_ssh_target '-oProxyCommand=unsafe' 'test SSH target') \
  >/dev/null 2>&1; then
  xycar_ai_die "unsafe SSH target was accepted"
fi
mkdir -p "${XYCAR_TEST_ROOT}/unsafe-artifact"
printf 'schema_version: 1\n' \
  >"${XYCAR_TEST_ROOT}/unsafe-artifact/manifest.yaml"
printf '%064d  ../outside.onnx\n' 0 \
  >"${XYCAR_TEST_ROOT}/unsafe-artifact/SHA256SUMS"
if (xycar_ai_verify_sha_manifest \
  "${XYCAR_TEST_ROOT}/unsafe-artifact") >/dev/null 2>&1; then
  xycar_ai_die "unsafe checksum path was accepted"
fi
printf 'unlisted\n' >"${XYCAR_TEST_ROOT}/artifact/unlisted.bin"
if (xycar_ai_verify_sha_manifest \
  "${XYCAR_TEST_ROOT}/artifact") >/dev/null 2>&1; then
  xycar_ai_die "unlisted artifact file was accepted"
fi
find "${XYCAR_TEST_ROOT}/artifact" -type f \
  -name unlisted.bin -delete
mkfifo "${XYCAR_TEST_ROOT}/artifact/unlisted.fifo"
if (xycar_ai_verify_sha_manifest \
  "${XYCAR_TEST_ROOT}/artifact") >/dev/null 2>&1; then
  xycar_ai_die "non-regular artifact entry was accepted"
fi

printf 'AI workflow fixture tests passed\n'
