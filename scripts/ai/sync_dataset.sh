#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

XYCAR_DRY_RUN=0
XYCAR_CHECKSUM=0
while (($#)); do
  case "$1" in
    --dry-run)
      XYCAR_DRY_RUN=1
      ;;
    --checksum)
      XYCAR_CHECKSUM=1
      ;;
    -h|--help)
      printf 'usage: %s [--dry-run] [--checksum]\n' "$0"
      exit 0
      ;;
    *)
      xycar_ai_die "unknown argument: $1"
      ;;
  esac
  shift
done

xycar_ai_require_authoring_checkout
xycar_ai_require_command base64
xycar_ai_require_command mktemp
xycar_ai_require_command ssh
xycar_ai_validate_ssh_target "${XYCAR_AI_TRAIN_SSH}" "training SSH target"
xycar_ai_validate_ssh_target "${XYCAR_AI_VEHICLE_SSH}" "vehicle SSH target"
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_VEHICLE_DATASET_ROOT}" "vehicle dataset root"
xycar_ai_validate_absolute_path "${XYCAR_AI_TRAIN_ROOT}" "training root"

XYCAR_VEHICLE_USER="${XYCAR_AI_VEHICLE_SSH%@*}"
XYCAR_VEHICLE_HOST="${XYCAR_AI_VEHICLE_SSH#*@}"
[[ "${XYCAR_VEHICLE_USER}" != "${XYCAR_AI_VEHICLE_SSH}" ]] ||
  xycar_ai_die "vehicle SSH target must use user@host form"
XYCAR_TUNNEL_PORT="${XYCAR_AI_TUNNEL_PORT:-10022}"
[[ "${XYCAR_TUNNEL_PORT}" =~ ^[0-9]+$ ]] ||
  xycar_ai_die "tunnel port must be numeric"
((XYCAR_TUNNEL_PORT >= 1024 && XYCAR_TUNNEL_PORT <= 65535)) ||
  xycar_ai_die "tunnel port must be between 1024 and 65535"

XYCAR_TUNNEL_DIR="$(mktemp -d)"
XYCAR_CONTROL_SOCKET="${XYCAR_TUNNEL_DIR}/ssh-control"
XYCAR_TUNNEL_OPEN=0
xycar_ai_cleanup_tunnel() {
  if ((XYCAR_TUNNEL_OPEN)); then
    ssh -S "${XYCAR_CONTROL_SOCKET}" -O exit \
      "${XYCAR_AI_TRAIN_SSH}" >/dev/null 2>&1 || true
  fi
  if [[ -S "${XYCAR_CONTROL_SOCKET}" ]]; then
    unlink "${XYCAR_CONTROL_SOCKET}"
  fi
  rmdir "${XYCAR_TUNNEL_DIR}" 2>/dev/null || true
}
trap xycar_ai_cleanup_tunnel EXIT

ssh -M -S "${XYCAR_CONTROL_SOCKET}" \
  -o ConnectTimeout=8 \
  -o ExitOnForwardFailure=yes \
  -fNT \
  -R "127.0.0.1:${XYCAR_TUNNEL_PORT}:${XYCAR_VEHICLE_HOST}:22" \
  "${XYCAR_AI_TRAIN_SSH}"
XYCAR_TUNNEL_OPEN=1

XYCAR_REMOTE_DRY_RUN=0
XYCAR_REMOTE_CHECKSUM=0
if ((XYCAR_DRY_RUN)); then
  XYCAR_REMOTE_DRY_RUN=1
fi
if ((XYCAR_CHECKSUM)); then
  XYCAR_REMOTE_CHECKSUM=1
fi

read -r -d '' XYCAR_REMOTE_SCRIPT <<'REMOTE_RSYNC' || true
set -euo pipefail
training_root="$1"
vehicle_dataset_root="$2"
vehicle_user="$3"
tunnel_port="$4"
dry_run="$5"
checksum="$6"
destination="${training_root}/datasets/teleop"

[[ -f "${training_root}/.xycar-ai-workspace" ]] ||
  { printf 'training workspace is not initialized\n' >&2; exit 1; }
[[ -d "${destination}" ]] ||
  { printf 'dataset destination is missing: %s\n' "${destination}" >&2; exit 1; }
command -v rsync >/dev/null 2>&1 ||
  { printf 'rsync is unavailable on the training host\n' >&2; exit 1; }

rsync_args=(
  -a
  --human-readable
  --itemize-changes
  --info=progress2
  --partial
  --partial-dir=.rsync-partial
  --protect-args
  '--exclude=/_recording_*/'
  '--exclude=/.rsync-partial/'
)
if [[ "${dry_run}" == "1" ]]; then
  rsync_args+=(--dry-run)
fi
if [[ "${checksum}" == "1" ]]; then
  rsync_args+=(--checksum)
fi

rsync "${rsync_args[@]}" \
  -e "ssh -p ${tunnel_port} -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o HostKeyAlias=xycar-vehicle-via-dev -o StrictHostKeyChecking=accept-new" \
  "${vehicle_user}@127.0.0.1:${vehicle_dataset_root}/" \
  "${destination}/"
REMOTE_RSYNC

XYCAR_REMOTE_SCRIPT_B64="$(
  printf '%s' "${XYCAR_REMOTE_SCRIPT}" | base64 -w0
)"
printf -v XYCAR_REMOTE_COMMAND \
  'printf %%s %q | base64 -d | bash -s -- %q %q %q %q %q %q' \
  "${XYCAR_REMOTE_SCRIPT_B64}" \
  "${XYCAR_AI_TRAIN_ROOT}" \
  "${XYCAR_AI_VEHICLE_DATASET_ROOT}" \
  "${XYCAR_VEHICLE_USER}" \
  "${XYCAR_TUNNEL_PORT}" \
  "${XYCAR_REMOTE_DRY_RUN}" \
  "${XYCAR_REMOTE_CHECKSUM}"

if ! ssh -tt -o ConnectTimeout=8 \
  "${XYCAR_AI_TRAIN_SSH}" \
  "${XYCAR_REMOTE_COMMAND}"; then
  xycar_ai_die \
    "dataset sync failed; verify that the development PC can reach the vehicle and retry"
fi

printf 'Dataset sync complete: %s -> %s:%s/datasets/teleop\n' \
  "${XYCAR_AI_VEHICLE_SSH}" \
  "${XYCAR_AI_TRAIN_SSH}" \
  "${XYCAR_AI_TRAIN_ROOT}"
