#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

readonly XYCAR_LAN_WINDOWS_INTERFACE="${XYCAR_AI_LAN_WINDOWS_INTERFACE:-Ethernet}"
readonly XYCAR_LAN_WINDOWS_ADDRESS="${XYCAR_AI_LAN_WINDOWS_ADDRESS:-192.168.50.1}"
readonly XYCAR_LAN_SUBNET="${XYCAR_AI_LAN_SUBNET:-192.168.50.0/24}"
readonly XYCAR_LAN_VEHICLE_ADDRESS="${XYCAR_AI_LAN_VEHICLE_ADDRESS:-192.168.50.2}"
readonly XYCAR_LAN_VEHICLE_SSH="${XYCAR_AI_LAN_VEHICLE_SSH:-xytron@192.168.50.2}"
readonly XYCAR_LAN_EXPECTED_HOSTNAME="${XYCAR_AI_LAN_EXPECTED_HOSTNAME:-xycar-gpu}"
readonly XYCAR_LAN_SOURCE_ROOT="${XYCAR_AI_LAN_VEHICLE_DATASET_ROOT:-/home/xytron/xycar_data/stateless_manual}"
readonly XYCAR_LAN_DESTINATION_ROOT="${XYCAR_AI_LAN_LOCAL_DATASET_ROOT:-${XYCAR_AI_BUNDLE_ROOT}/datasets/stateless_manual}"

XYCAR_DRY_RUN=0
XYCAR_CHECKSUM=0
XYCAR_ALLOW_EMPTY_SOURCE=0
while (($#)); do
  case "$1" in
    --dry-run)
      XYCAR_DRY_RUN=1
      ;;
    --checksum)
      XYCAR_CHECKSUM=1
      ;;
    --allow-empty-source)
      XYCAR_ALLOW_EMPTY_SOURCE=1
      ;;
    -h|--help)
      printf 'usage: %s [--dry-run] [--checksum] [--allow-empty-source]\n' "$0"
      printf '%s\n' \
        'default: immediately mirror vehicle stateless_manual over the fixed direct LAN'
      exit 0
      ;;
    *)
      xycar_ai_die "unknown argument: $1"
      ;;
  esac
  shift
done

xycar_ai_require_authoring_checkout
xycar_ai_require_command findmnt
xycar_ai_require_command flock
xycar_ai_require_command mktemp
xycar_ai_require_command powershell.exe
xycar_ai_require_command realpath
xycar_ai_require_command rsync
xycar_ai_require_command ssh
xycar_ai_validate_ssh_target \
  "${XYCAR_LAN_VEHICLE_SSH}" "direct-LAN vehicle SSH target"
xycar_ai_validate_absolute_path \
  "${XYCAR_LAN_SOURCE_ROOT}" "direct-LAN vehicle dataset root"
xycar_ai_validate_absolute_path \
  "${XYCAR_LAN_DESTINATION_ROOT}" "direct-LAN local dataset root"

readonly XYCAR_EXPECTED_DESTINATION="${XYCAR_AI_BUNDLE_ROOT}/datasets/stateless_manual"
if [[ "${XYCAR_AI_ALLOW_ANY_CHECKOUT:-0}" != "1" ]]; then
  [[ "${XYCAR_LAN_WINDOWS_INTERFACE}" == "Ethernet" ]] ||
    xycar_ai_die "direct-LAN Windows interface must be Ethernet"
  [[ "${XYCAR_LAN_WINDOWS_ADDRESS}" == "192.168.50.1" ]] ||
    xycar_ai_die "direct-LAN Windows address must be 192.168.50.1"
  [[ "${XYCAR_LAN_SUBNET}" == "192.168.50.0/24" ]] ||
    xycar_ai_die "direct-LAN subnet must be 192.168.50.0/24"
  [[ "${XYCAR_LAN_VEHICLE_ADDRESS}" == "192.168.50.2" ]] ||
    xycar_ai_die "direct-LAN vehicle address must be 192.168.50.2"
  [[ "${XYCAR_LAN_VEHICLE_SSH}" == "xytron@192.168.50.2" ]] ||
    xycar_ai_die "direct-LAN SSH target must be xytron@192.168.50.2"
  [[ "${XYCAR_LAN_EXPECTED_HOSTNAME}" == "xycar-gpu" ]] ||
    xycar_ai_die "direct-LAN vehicle hostname must be xycar-gpu"
  [[ "${XYCAR_LAN_SOURCE_ROOT}" == \
      "/home/xytron/xycar_data/stateless_manual" ]] ||
    xycar_ai_die \
      "direct-LAN vehicle source must be /home/xytron/xycar_data/stateless_manual"
  [[ "$(realpath -m -- "${XYCAR_LAN_DESTINATION_ROOT}")" == \
      "$(realpath -m -- "${XYCAR_EXPECTED_DESTINATION}")" ]] ||
    xycar_ai_die \
      "direct-LAN destination must be ${XYCAR_EXPECTED_DESTINATION}"
fi

readonly XYCAR_DESTINATION_PARENT="$(dirname -- "${XYCAR_LAN_DESTINATION_ROOT}")"
[[ -d "${XYCAR_DESTINATION_PARENT}" ]] ||
  xycar_ai_die "local dataset parent is missing: ${XYCAR_DESTINATION_PARENT}"
XYCAR_DESTINATION_FS="$(
  findmnt -n -o FSTYPE -T "${XYCAR_DESTINATION_PARENT}"
)" || xycar_ai_die "cannot resolve local dataset filesystem"
readonly XYCAR_DESTINATION_FS
if [[ "${XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION:-0}" != "1" ]] &&
   [[ "${XYCAR_DESTINATION_FS}" != "ext4" ]]; then
  xycar_ai_die \
    "direct-LAN destination must be on ext4: ${XYCAR_DESTINATION_FS}"
fi
[[ ! -L "${XYCAR_LAN_DESTINATION_ROOT}" ]] ||
  xycar_ai_die "direct-LAN destination must not be a symlink"
if [[ -e "${XYCAR_LAN_DESTINATION_ROOT}" && \
      ! -d "${XYCAR_LAN_DESTINATION_ROOT}" ]]; then
  xycar_ai_die \
    "direct-LAN destination is not a directory: ${XYCAR_LAN_DESTINATION_ROOT}"
fi

readonly XYCAR_LOCK_PATH="${XYCAR_DESTINATION_PARENT}/.stateless_manual_lan_sync.lock"
exec 9>"${XYCAR_LOCK_PATH}"
flock -n 9 || xycar_ai_die "another stateless manual LAN sync is already running"

XYCAR_WINDOWS_STATE="$(
  powershell.exe -NoProfile -NonInteractive -Command \
    '& {
      param($InterfaceName, $WindowsAddress, $Subnet, $VehicleAddress)
      $ErrorActionPreference = "Stop"
      $adapter = Get-NetAdapter -Name $InterfaceName -ErrorAction Stop
      if ($adapter.Status -ne "Up") { throw "Ethernet adapter is not Up" }
      $address = Get-NetIPAddress -InterfaceAlias $InterfaceName -AddressFamily IPv4 |
        Where-Object IPAddress -eq $WindowsAddress
      if ($null -eq $address) { throw "fixed Ethernet IPv4 address is missing" }
      $route = Get-NetRoute -AddressFamily IPv4 -DestinationPrefix $Subnet |
        Where-Object InterfaceIndex -eq $adapter.ifIndex
      if ($null -eq $route) { throw "direct Ethernet route is missing" }
      $reachable = Test-NetConnection -ComputerName $VehicleAddress -Port 22 -InformationLevel Quiet -WarningAction SilentlyContinue
      if (-not $reachable) { throw "vehicle LAN SSH port is unreachable" }
      [Console]::Out.Write(
        $adapter.LinkSpeed.ToString() + "|" +
        $InterfaceName + "|" +
        $Subnet + "|" +
        $WindowsAddress
      )
    }' \
    "${XYCAR_LAN_WINDOWS_INTERFACE}" \
    "${XYCAR_LAN_WINDOWS_ADDRESS}" \
    "${XYCAR_LAN_SUBNET}" \
    "${XYCAR_LAN_VEHICLE_ADDRESS}"
)" || xycar_ai_die \
  "Windows direct-LAN preflight failed; verify the fixed Ethernet configuration"
XYCAR_WINDOWS_STATE="${XYCAR_WINDOWS_STATE//$'\r'/}"
IFS='|' read -r \
  XYCAR_LINK_SPEED \
  XYCAR_ROUTE_INTERFACE \
  XYCAR_ROUTE_SUBNET \
  XYCAR_ROUTE_ADDRESS <<<"${XYCAR_WINDOWS_STATE}"
[[ "${XYCAR_ROUTE_INTERFACE}" == "${XYCAR_LAN_WINDOWS_INTERFACE}" &&
   "${XYCAR_ROUTE_SUBNET}" == "${XYCAR_LAN_SUBNET}" &&
   "${XYCAR_ROUTE_ADDRESS}" == "${XYCAR_LAN_WINDOWS_ADDRESS}" ]] ||
  xycar_ai_die "Windows direct-LAN preflight returned an unexpected route"

XYCAR_SSH_ARGS=(
  -o BatchMode=yes
  -o Compression=no
  -o ConnectTimeout=8
  -o HostKeyAlias=xycar-gpu
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  -o StrictHostKeyChecking=yes
)
XYCAR_SOURCE_STATE="$(
  ssh "${XYCAR_SSH_ARGS[@]}" "${XYCAR_LAN_VEHICLE_SSH}" \
    bash -s -- "${XYCAR_LAN_SOURCE_ROOT}" "${XYCAR_LAN_EXPECTED_HOSTNAME}" <<'REMOTE'
set -euo pipefail
source_root="$1"
expected_hostname="$2"
[[ "$(hostname -s)" == "${expected_hostname}" ]] || {
  printf 'unexpected vehicle hostname: %s\n' "$(hostname -s)" >&2
  exit 1
}
[[ -d "${source_root}" ]] || {
  printf 'vehicle dataset root is missing: %s\n' "${source_root}" >&2
  exit 1
}
[[ ! -L "${source_root}" && "$(realpath -e -- "${source_root}")" == "${source_root}" ]] || {
  printf 'vehicle dataset root must be the exact non-symlink path: %s\n' \
    "${source_root}" >&2
  exit 1
}
first_closed_entry="$(
  find "${source_root}" -mindepth 1 -maxdepth 1 \
    ! -name '_recording_*' \
    ! -name '.rsync-partial' \
    ! -name '*.partial' \
    ! -name '*.part' \
    ! -name '*.tmp' \
    -print -quit
)"
active_count="$(
  find "${source_root}" -mindepth 1 -maxdepth 1 \
    -type d -name '_recording_*' -printf '.' | wc -c
)"
if [[ -n "${first_closed_entry}" ]]; then
  printf 'nonempty|%s\n' "${active_count}"
else
  printf 'empty|%s\n' "${active_count}"
fi
REMOTE
)" || xycar_ai_die "vehicle identity or dataset preflight failed over direct LAN"
IFS='|' read -r XYCAR_SOURCE_CONTENT XYCAR_ACTIVE_COUNT <<<"${XYCAR_SOURCE_STATE}"
[[ "${XYCAR_SOURCE_CONTENT}" == "empty" || \
   "${XYCAR_SOURCE_CONTENT}" == "nonempty" ]] ||
  xycar_ai_die "vehicle dataset preflight returned an unexpected source state"
[[ "${XYCAR_ACTIVE_COUNT}" =~ ^[0-9]+$ ]] ||
  xycar_ai_die "vehicle dataset preflight returned an invalid active count"
if [[ "${XYCAR_SOURCE_CONTENT}" == "empty" ]] &&
   ((!XYCAR_ALLOW_EMPTY_SOURCE)); then
  xycar_ai_die \
    "vehicle dataset has no closed entries; use --allow-empty-source only for an intentional empty mirror"
fi

XYCAR_PREVIEW_ROOT=""
xycar_cleanup_lan_sync() {
  if [[ -n "${XYCAR_PREVIEW_ROOT}" && -d "${XYCAR_PREVIEW_ROOT}" ]]; then
    find "${XYCAR_PREVIEW_ROOT}" -depth -mindepth 1 -delete
    rmdir "${XYCAR_PREVIEW_ROOT}"
  fi
}
trap xycar_cleanup_lan_sync EXIT

XYCAR_SYNC_DESTINATION="${XYCAR_LAN_DESTINATION_ROOT}"
XYCAR_MARKER="${XYCAR_LAN_DESTINATION_ROOT}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
if [[ -d "${XYCAR_LAN_DESTINATION_ROOT}" && -f "${XYCAR_MARKER}" ]]; then
  [[ "$(<"${XYCAR_MARKER}")" == "${XYCAR_AI_LOCAL_DATASET_MARKER_CONTENT}" ]] ||
    xycar_ai_die "local dataset marker has unexpected content: ${XYCAR_MARKER}"
elif [[ -d "${XYCAR_LAN_DESTINATION_ROOT}" ]] &&
     [[ -n "$(find "${XYCAR_LAN_DESTINATION_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  xycar_ai_die \
    "refusing to mirror-delete an unmarked non-empty destination: ${XYCAR_LAN_DESTINATION_ROOT}"
elif ((XYCAR_DRY_RUN)); then
  XYCAR_PREVIEW_ROOT="$(
    mktemp -d "${XYCAR_DESTINATION_PARENT}/.xycar-lan-sync-preview.XXXXXX"
  )"
  XYCAR_SYNC_DESTINATION="${XYCAR_PREVIEW_ROOT}"
  printf 'dry-run: would initialize marked dataset destination %s\n' \
    "${XYCAR_LAN_DESTINATION_ROOT}"
else
  mkdir -p -- "${XYCAR_LAN_DESTINATION_ROOT}"
  printf '%s\n' "${XYCAR_AI_LOCAL_DATASET_MARKER_CONTENT}" >"${XYCAR_MARKER}"
fi

XYCAR_RSYNC_ARGS=(
  -a
  --human-readable
  --itemize-changes
  --info=progress2
  --partial
  --partial-dir=.rsync-partial
  --protect-args
  --delete-delay
  --stats
  --filter="protect /${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
  --filter="merge ${XYCAR_AI_SCRIPT_DIR}/dataset-rsync-filter.rules"
)
if ((XYCAR_DRY_RUN)); then
  XYCAR_RSYNC_ARGS+=(--dry-run)
fi
if ((XYCAR_CHECKSUM)); then
  XYCAR_RSYNC_ARGS+=(--checksum)
fi

readonly XYCAR_RSH='ssh -o BatchMode=yes -o Compression=no -o ConnectTimeout=8 -o HostKeyAlias=xycar-gpu -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o StrictHostKeyChecking=yes'
printf 'Direct LAN: %s (%s, %s) -> %s\n' \
  "${XYCAR_LAN_WINDOWS_ADDRESS}" \
  "${XYCAR_LAN_WINDOWS_INTERFACE}" \
  "${XYCAR_LINK_SPEED}" \
  "${XYCAR_LAN_VEHICLE_ADDRESS}"
printf 'Dataset mirror: %s:%s -> %s; active recording directories skipped: %s\n' \
  "${XYCAR_LAN_VEHICLE_SSH}" \
  "${XYCAR_LAN_SOURCE_ROOT}" \
  "${XYCAR_LAN_DESTINATION_ROOT}" \
  "${XYCAR_ACTIVE_COUNT}"
rsync "${XYCAR_RSYNC_ARGS[@]}" \
  -e "${XYCAR_RSH}" \
  "${XYCAR_LAN_VEHICLE_SSH}:${XYCAR_LAN_SOURCE_ROOT}/" \
  "${XYCAR_SYNC_DESTINATION}/"

if ((XYCAR_DRY_RUN)); then
  printf 'Direct-LAN mirror dry-run complete; no dataset files were changed.\n'
else
  printf 'Direct-LAN stateless manual mirror applied: %s\n' \
    "${XYCAR_LAN_DESTINATION_ROOT}"
fi
