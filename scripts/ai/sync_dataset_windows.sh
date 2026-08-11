#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

XYCAR_APPLY=0
XYCAR_CHECKSUM=0
XYCAR_INIT=0
while (($#)); do
  case "$1" in
    --apply)
      XYCAR_APPLY=1
      ;;
    --checksum)
      XYCAR_CHECKSUM=1
      ;;
    --init)
      XYCAR_INIT=1
      ;;
    -h|--help)
      printf 'usage: %s [--init] [--apply] [--checksum]\n' "$0"
      printf 'default: dry-run; mirror eligible Windows sessions into WSL ext4\n'
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
xycar_ai_require_command mktemp
xycar_ai_require_command realpath
xycar_ai_require_command rsync
xycar_ai_require_command uv
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_WINDOWS_DATASET_ROOT}" "Windows dataset root"
xycar_ai_validate_absolute_path \
  "${XYCAR_AI_LOCAL_DATASET_ROOT}" "local dataset root"

[[ -d "${XYCAR_AI_WINDOWS_DATASET_ROOT}" ]] ||
  xycar_ai_die \
    "Windows dataset root is missing: ${XYCAR_AI_WINDOWS_DATASET_ROOT}"
XYCAR_SOURCE_REAL="$(realpath -e -- "${XYCAR_AI_WINDOWS_DATASET_ROOT}")"
XYCAR_DESTINATION_PARENT="$(dirname -- "${XYCAR_AI_LOCAL_DATASET_ROOT}")"
[[ -d "${XYCAR_DESTINATION_PARENT}" ]] ||
  xycar_ai_die "local dataset parent is missing: ${XYCAR_DESTINATION_PARENT}"

if [[ "${XYCAR_AI_ALLOW_NON_WINDOWS_SOURCE:-0}" != "1" ]]; then
  XYCAR_SOURCE_FS="$(
    findmnt -n -o FSTYPE -T "${XYCAR_SOURCE_REAL}"
  )" || xycar_ai_die "cannot resolve Windows dataset filesystem"
  [[ "${XYCAR_SOURCE_FS}" == "9p" || "${XYCAR_SOURCE_FS}" == "drvfs" ]] ||
    xycar_ai_die \
      "Windows dataset source must be on a WSL Windows mount: ${XYCAR_SOURCE_FS}"
fi

XYCAR_DESTINATION_FS="$(
  findmnt -n -o FSTYPE -T "${XYCAR_DESTINATION_PARENT}"
)" || xycar_ai_die "cannot resolve local dataset filesystem"
[[ "${XYCAR_DESTINATION_FS}" == "ext4" ]] ||
  xycar_ai_die \
    "local dataset destination must be on ext4: ${XYCAR_DESTINATION_FS}"

XYCAR_DESTINATION_IS_LINK=0
if [[ -L "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]]; then
  XYCAR_DESTINATION_IS_LINK=1
  ((XYCAR_INIT)) ||
    xycar_ai_die \
      "local dataset is still a symlink; use --init first: ${XYCAR_AI_LOCAL_DATASET_ROOT}"
  XYCAR_LINK_TARGET="$(realpath -e -- "${XYCAR_AI_LOCAL_DATASET_ROOT}")"
  [[ "${XYCAR_LINK_TARGET}" == "${XYCAR_SOURCE_REAL}" ]] ||
    xycar_ai_die \
      "local dataset symlink points outside the configured Windows source: ${XYCAR_LINK_TARGET}"
elif [[ -e "${XYCAR_AI_LOCAL_DATASET_ROOT}" && \
        ! -d "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]]; then
  xycar_ai_die \
    "local dataset destination is not a directory: ${XYCAR_AI_LOCAL_DATASET_ROOT}"
fi
if ((!XYCAR_DESTINATION_IS_LINK)) && \
   [[ -e "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]] && \
   [[ "$(realpath -e -- "${XYCAR_AI_LOCAL_DATASET_ROOT}")" == "${XYCAR_SOURCE_REAL}" ]]; then
  xycar_ai_die "Windows source and local destination must differ"
fi

XYCAR_SELECTION_OUTPUT="$(
  cd -- "${XYCAR_AI_BUNDLE_ROOT}"
  uv run --locked python -m xycar_ai.select_policy_sessions \
    --root "${XYCAR_SOURCE_REAL}" \
    --min-forward-speed "${XYCAR_AI_MIN_FORWARD_SPEED}"
)" || xycar_ai_die "failed to select eligible Windows dataset sessions"
mapfile -t XYCAR_SELECTED_SESSIONS <<<"${XYCAR_SELECTION_OUTPUT}"
((${#XYCAR_SELECTED_SESSIONS[@]} > 0)) ||
  xycar_ai_die "no eligible Windows dataset sessions were selected"

XYCAR_TEMP_ROOT=""
XYCAR_FILTER_FILE="$(mktemp)"
xycar_ai_cleanup_windows_sync() {
  if [[ -n "${XYCAR_TEMP_ROOT}" && -d "${XYCAR_TEMP_ROOT}" ]]; then
    find "${XYCAR_TEMP_ROOT}" -depth -mindepth 1 -delete
    rmdir "${XYCAR_TEMP_ROOT}"
  fi
  if [[ -f "${XYCAR_FILTER_FILE}" ]]; then
    find "${XYCAR_FILTER_FILE}" -maxdepth 0 -type f -delete
  fi
}
trap xycar_ai_cleanup_windows_sync EXIT

{
  printf 'P /%s\n' "${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
  printf '%s\n' 'P /.rsync-partial/***'
  printf '%s\n' '- /.rsync-partial/***'
  printf '%s\n' '- /_recording_*/***'
  printf '%s\n' '- *.partial'
  printf '%s\n' '- *.part'
  printf '%s\n' '- *.tmp'
  for session in "${XYCAR_SELECTED_SESSIONS[@]}"; do
    printf '+ /%s/***\n' "${session}"
  done
  printf '%s\n' '- /*'
} >"${XYCAR_FILTER_FILE}"

XYCAR_SYNC_DESTINATION="${XYCAR_AI_LOCAL_DATASET_ROOT}"
if ((XYCAR_INIT)); then
  if ((XYCAR_APPLY)); then
    if ((XYCAR_DESTINATION_IS_LINK)); then
      unlink -- "${XYCAR_AI_LOCAL_DATASET_ROOT}"
      mkdir -p -- "${XYCAR_AI_LOCAL_DATASET_ROOT}"
    elif [[ ! -e "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]]; then
      mkdir -p -- "${XYCAR_AI_LOCAL_DATASET_ROOT}"
    elif [[ ! -f "${XYCAR_AI_LOCAL_DATASET_ROOT}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}" ]] &&
         [[ -n "$(find "${XYCAR_AI_LOCAL_DATASET_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      xycar_ai_die \
        "refusing to initialize a non-empty unmarked destination: ${XYCAR_AI_LOCAL_DATASET_ROOT}"
    fi
    printf '%s\n' "${XYCAR_AI_LOCAL_DATASET_MARKER_CONTENT}" \
      >"${XYCAR_AI_LOCAL_DATASET_ROOT}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
  else
    XYCAR_TEMP_ROOT="$(
      mktemp -d "${XYCAR_DESTINATION_PARENT}/.xycar-windows-sync-preview.XXXXXX"
    )"
    XYCAR_SYNC_DESTINATION="${XYCAR_TEMP_ROOT}"
    printf 'dry-run: would replace or initialize %s as an ext4 dataset directory\n' \
      "${XYCAR_AI_LOCAL_DATASET_ROOT}"
  fi
else
  [[ ! -L "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]] ||
    xycar_ai_die "local dataset destination must not be a symlink"
  [[ -d "${XYCAR_AI_LOCAL_DATASET_ROOT}" ]] ||
    xycar_ai_die "local dataset destination is missing; use --init first"
  XYCAR_MARKER="${XYCAR_AI_LOCAL_DATASET_ROOT}/${XYCAR_AI_LOCAL_DATASET_MARKER_NAME}"
  [[ -f "${XYCAR_MARKER}" ]] ||
    xycar_ai_die "local dataset marker is missing: ${XYCAR_MARKER}"
  [[ "$(<"${XYCAR_MARKER}")" == "${XYCAR_AI_LOCAL_DATASET_MARKER_CONTENT}" ]] ||
    xycar_ai_die "local dataset marker has unexpected content: ${XYCAR_MARKER}"
fi

if [[ "$(realpath -m -- "${XYCAR_SYNC_DESTINATION}")" == "${XYCAR_SOURCE_REAL}" ]]; then
  xycar_ai_die "Windows source and local destination must differ"
fi

XYCAR_RSYNC_ARGS=(
  -rt
  --modify-window=1
  --omit-dir-times
  --human-readable
  --itemize-changes
  --info=progress2
  --partial
  --partial-dir=.rsync-partial
  --protect-args
  --delete-delay
  --delete-excluded
  --filter="merge ${XYCAR_FILTER_FILE}"
)
if ((!XYCAR_APPLY)); then
  XYCAR_RSYNC_ARGS+=(--dry-run)
fi
if ((XYCAR_CHECKSUM)); then
  XYCAR_RSYNC_ARGS+=(--checksum)
fi

printf 'Selected %d completed gamepad session(s) with max_forward_speed >= %s\n' \
  "${#XYCAR_SELECTED_SESSIONS[@]}" "${XYCAR_AI_MIN_FORWARD_SPEED}"
rsync "${XYCAR_RSYNC_ARGS[@]}" \
  "${XYCAR_SOURCE_REAL}/" \
  "${XYCAR_SYNC_DESTINATION}/"

if ((XYCAR_APPLY)); then
  printf 'Windows dataset mirror applied: %s -> %s\n' \
    "${XYCAR_SOURCE_REAL}" "${XYCAR_AI_LOCAL_DATASET_ROOT}"
else
  printf 'Windows dataset dry-run complete; rerun with --apply to mirror changes.\n'
fi
