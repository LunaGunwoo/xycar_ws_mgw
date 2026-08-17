#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

readonly XYCAR_LAN_WINDOWS_INTERFACE="${XYCAR_AI_LAN_WINDOWS_INTERFACE:-Ethernet}"
readonly XYCAR_LAN_WINDOWS_ADDRESS="${XYCAR_AI_LAN_WINDOWS_ADDRESS:-192.168.50.1}"
readonly XYCAR_LAN_SUBNET="${XYCAR_AI_LAN_SUBNET:-192.168.50.0/24}"
readonly XYCAR_LAN_VEHICLE_ADDRESS="${XYCAR_AI_LAN_VEHICLE_ADDRESS:-192.168.50.2}"
readonly XYCAR_LAN_VEHICLE_SSH="${XYCAR_AI_LAN_VEHICLE_SSH:-xytron@192.168.50.2}"
readonly XYCAR_LAN_EXPECTED_HOSTNAME="${XYCAR_AI_LAN_EXPECTED_HOSTNAME:-xycar-gpu}"
readonly XYCAR_LAN_SOURCE_BASE="${XYCAR_AI_LAN_GUIDED_VEHICLE_ROOT:-/home/xytron/xycar_data/stateless_guided}"
readonly XYCAR_LAN_DESTINATION_BASE="${XYCAR_AI_LAN_GUIDED_LOCAL_ROOT:-${XYCAR_AI_BUNDLE_ROOT}/datasets/stateless_guided}"

XYCAR_GENERATION=""
XYCAR_COLLECTION_ID=""
XYCAR_DRY_RUN=0
XYCAR_CHECKSUM=0
while (($#)); do
  case "$1" in
    --generation)
      (($# >= 2)) || xycar_ai_die "--generation requires a value"
      XYCAR_GENERATION="$2"
      shift
      ;;
    --collection-id)
      (($# >= 2)) || xycar_ai_die "--collection-id requires a value"
      XYCAR_COLLECTION_ID="$2"
      shift
      ;;
    --dry-run)
      XYCAR_DRY_RUN=1
      ;;
    --checksum)
      XYCAR_CHECKSUM=1
      ;;
    -h|--help)
      printf 'usage: %s --generation N --collection-id ID [--dry-run] [--checksum]\n' "$0"
      printf '%s\n' \
        'default: incrementally copy one Guided cohort over the fixed direct LAN without deletion'
      exit 0
      ;;
    *)
      xycar_ai_die "unknown argument: $1"
      ;;
  esac
  shift
done

[[ "${XYCAR_GENERATION}" =~ ^(0|[1-9][0-9]*)$ ]] ||
  xycar_ai_die "generation must be a non-negative integer"
[[ "${XYCAR_COLLECTION_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
  xycar_ai_die "collection ID contains unsupported characters"

readonly XYCAR_RELATIVE_COHORT="generation_${XYCAR_GENERATION}/${XYCAR_COLLECTION_ID}"
readonly XYCAR_LAN_SOURCE_ROOT="${XYCAR_LAN_SOURCE_BASE}/${XYCAR_RELATIVE_COHORT}"
readonly XYCAR_LAN_DESTINATION_ROOT="${XYCAR_LAN_DESTINATION_BASE}/${XYCAR_RELATIVE_COHORT}"

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
  "${XYCAR_LAN_SOURCE_BASE}" "direct-LAN Guided vehicle base"
xycar_ai_validate_absolute_path \
  "${XYCAR_LAN_DESTINATION_BASE}" "direct-LAN Guided local base"
xycar_ai_validate_absolute_path \
  "${XYCAR_LAN_SOURCE_ROOT}" "direct-LAN Guided vehicle cohort"
xycar_ai_validate_absolute_path \
  "${XYCAR_LAN_DESTINATION_ROOT}" "direct-LAN Guided local cohort"

readonly XYCAR_EXPECTED_SOURCE_BASE="/home/xytron/xycar_data/stateless_guided"
readonly XYCAR_EXPECTED_DESTINATION_BASE="${XYCAR_AI_BUNDLE_ROOT}/datasets/stateless_guided"
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
  [[ "${XYCAR_LAN_SOURCE_BASE}" == "${XYCAR_EXPECTED_SOURCE_BASE}" ]] ||
    xycar_ai_die \
      "direct-LAN Guided source base must be ${XYCAR_EXPECTED_SOURCE_BASE}"
  [[ "$(realpath -m -- "${XYCAR_LAN_DESTINATION_BASE}")" == \
      "$(realpath -m -- "${XYCAR_EXPECTED_DESTINATION_BASE}")" ]] ||
    xycar_ai_die \
      "direct-LAN Guided destination base must be ${XYCAR_EXPECTED_DESTINATION_BASE}"
fi

readonly XYCAR_DESTINATION_PARENT="$(dirname -- "${XYCAR_LAN_DESTINATION_ROOT}")"
readonly XYCAR_DESTINATION_BASE_PARENT="$(dirname -- "${XYCAR_LAN_DESTINATION_BASE}")"
[[ -d "${XYCAR_LAN_DESTINATION_BASE}" ]] ||
  xycar_ai_die \
    "local Guided dataset base is missing; run bootstrap_env.sh: ${XYCAR_LAN_DESTINATION_BASE}"
readonly XYCAR_EXISTING_DESTINATION_ANCESTOR="$({
  probe="${XYCAR_LAN_DESTINATION_ROOT}"
  while [[ ! -e "${probe}" ]]; do
    probe="$(dirname -- "${probe}")"
  done
  printf '%s\n' "${probe}"
})"
XYCAR_DESTINATION_FS="$(
  findmnt -n -o FSTYPE -T "${XYCAR_EXISTING_DESTINATION_ANCESTOR}"
)" || xycar_ai_die "cannot resolve local Guided dataset filesystem"
readonly XYCAR_DESTINATION_FS
if [[ "${XYCAR_AI_LAN_ALLOW_NON_EXT4_DESTINATION:-0}" != "1" ]] &&
   [[ "${XYCAR_DESTINATION_FS}" != "ext4" ]]; then
  xycar_ai_die \
    "direct-LAN Guided destination must be on ext4: ${XYCAR_DESTINATION_FS}"
fi
for destination_path in \
  "${XYCAR_LAN_DESTINATION_BASE}" \
  "${XYCAR_DESTINATION_PARENT}" \
  "${XYCAR_LAN_DESTINATION_ROOT}"; do
  [[ ! -L "${destination_path}" ]] ||
    xycar_ai_die "direct-LAN Guided destination path must not be a symlink: ${destination_path}"
done
if [[ -e "${XYCAR_LAN_DESTINATION_ROOT}" && \
      ! -d "${XYCAR_LAN_DESTINATION_ROOT}" ]]; then
  xycar_ai_die \
    "direct-LAN Guided destination is not a directory: ${XYCAR_LAN_DESTINATION_ROOT}"
fi

readonly XYCAR_LOCK_PATH="${XYCAR_DESTINATION_BASE_PARENT}/.stateless_guided_lan_sync.lock"
exec 9>"${XYCAR_LOCK_PATH}"
flock -n 9 || xycar_ai_die "another stateless Guided LAN sync is already running"

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
  printf 'vehicle Guided cohort is missing: %s\n' "${source_root}" >&2
  exit 1
}
[[ ! -L "${source_root}" && "$(realpath -e -- "${source_root}")" == "${source_root}" ]] || {
  printf 'vehicle Guided cohort must be the exact non-symlink path: %s\n' \
    "${source_root}" >&2
  exit 1
}
closed_count="$(
  find "${source_root}" -mindepth 1 -maxdepth 1 \
    ! -name '_recording_*' \
    ! -name '.rsync-partial' \
    ! -name '*.partial' \
    ! -name '*.part' \
    ! -name '*.tmp' \
    -printf '.' | wc -c
)"
active_count="$(
  find "${source_root}" -mindepth 1 -maxdepth 1 \
    -type d -name '_recording_*' -printf '.' | wc -c
)"
printf '%s|%s\n' "${closed_count}" "${active_count}"
REMOTE
)" || xycar_ai_die "vehicle identity or Guided cohort preflight failed over direct LAN"
IFS='|' read -r XYCAR_CLOSED_COUNT XYCAR_ACTIVE_COUNT <<<"${XYCAR_SOURCE_STATE}"
[[ "${XYCAR_CLOSED_COUNT}" =~ ^[0-9]+$ && "${XYCAR_CLOSED_COUNT}" -gt 0 ]] ||
  xycar_ai_die "vehicle Guided cohort has no closed entries"
[[ "${XYCAR_ACTIVE_COUNT}" =~ ^[0-9]+$ ]] ||
  xycar_ai_die "vehicle Guided cohort preflight returned an invalid active count"

XYCAR_PREVIEW_ROOT=""
xycar_cleanup_guided_lan_sync() {
  if [[ -n "${XYCAR_PREVIEW_ROOT}" && -d "${XYCAR_PREVIEW_ROOT}" ]]; then
    find "${XYCAR_PREVIEW_ROOT}" -depth -mindepth 1 -delete
    rmdir "${XYCAR_PREVIEW_ROOT}"
  fi
}
trap xycar_cleanup_guided_lan_sync EXIT

XYCAR_SYNC_DESTINATION="${XYCAR_LAN_DESTINATION_ROOT}"
if [[ ! -d "${XYCAR_LAN_DESTINATION_ROOT}" ]] && ((XYCAR_DRY_RUN)); then
  XYCAR_PREVIEW_ROOT="$(
    mktemp -d "${XYCAR_LAN_DESTINATION_BASE}/.xycar-guided-lan-preview.XXXXXX"
  )"
  XYCAR_SYNC_DESTINATION="${XYCAR_PREVIEW_ROOT}"
  printf 'dry-run: would initialize Guided cohort destination %s\n' \
    "${XYCAR_LAN_DESTINATION_ROOT}"
elif ((!XYCAR_DRY_RUN)); then
  mkdir -p -- "${XYCAR_LAN_DESTINATION_ROOT}"
fi

XYCAR_RSYNC_ARGS=(
  -a
  --human-readable
  --itemize-changes
  --info=progress2
  --partial
  --partial-dir=.rsync-partial
  --protect-args
  --stats
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
printf 'Guided cohort copy: %s:%s -> %s; closed entries: %s; active recordings skipped: %s\n' \
  "${XYCAR_LAN_VEHICLE_SSH}" \
  "${XYCAR_LAN_SOURCE_ROOT}" \
  "${XYCAR_LAN_DESTINATION_ROOT}" \
  "${XYCAR_CLOSED_COUNT}" \
  "${XYCAR_ACTIVE_COUNT}"
rsync "${XYCAR_RSYNC_ARGS[@]}" \
  -e "${XYCAR_RSH}" \
  "${XYCAR_LAN_VEHICLE_SSH}:${XYCAR_LAN_SOURCE_ROOT}/" \
  "${XYCAR_SYNC_DESTINATION}/"

if ((XYCAR_DRY_RUN)); then
  printf 'Direct-LAN Guided dry-run complete; no dataset files were changed.\n'
else
  printf 'Direct-LAN Guided no-delete copy applied: %s\n' \
    "${XYCAR_LAN_DESTINATION_ROOT}"
fi
